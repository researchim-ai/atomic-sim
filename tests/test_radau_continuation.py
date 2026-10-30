import numpy as np
import pytest

from core.continuation import ImplicitContinuation
from core.status import SolverError
from sdk.api import create_session, export_snapshot
from sdk.experiment import validate_experiment


def test_radau_history_and_dense_output_match_stiff_analytic_solution():
    solver = ImplicitContinuation(
        lambda t, y: -np.array([1.0, 100.0]) * y,
        [1.0, 2.0],
        0.0,
        1.0,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1,
        jacobian_relative_step=1e-5,
    )
    engine = solver.solver
    for end in (0.3, 1.0):
        result = solver.advance_to(end)
        assert solver.solver is engine
        assert result.final == pytest.approx(
            np.exp(-np.array([1.0, 100.0]) * end) * [1.0, 2.0], abs=1e-9
        )
        assert result.dense(end) == pytest.approx(result.final)
        assert result.message == "Radau continuation accepted"


def test_method_change_restarts_history_and_invalid_method_rolls_back():
    plant = create_session()
    plant.enable_continuation(1.0)
    plant.advance(0.1, method="BDF", jacobian_relative_step=1e-5)
    old = plant._continuation
    plant.advance(0.1, method="Radau", jacobian_relative_step=1e-5)
    assert plant._continuation is not old
    before = export_snapshot(plant)
    with pytest.raises(SolverError):
        plant.advance(0.1, method="unknown")
    assert export_snapshot(plant) == before


def test_radau_localizes_protection_and_restarts_after_trip():
    plant = create_session()
    plant.enable_continuation(0.5)
    plant.power_protection.setpoint = 0.5 * plant.fission_power_nom
    plant.power_protection.delay_s = 0.15
    plant.advance(0.3, method="Radau", jacobian_relative_step=1e-5)
    assert plant.last_protection_event_s == pytest.approx(0.15, abs=1e-6)
    assert plant.scram_latched
    assert plant.rod == pytest.approx(plant.rod_ref + 0.15 * plant.scram_speed, abs=1e-5)


def test_experiment_method_is_validated_and_recorded_in_normalized_input():
    spec = dict(
        name="method", schema_version=1, engine="vver", duration_s=1, solver={"method": "Radau"}
    )
    assert validate_experiment(spec)["solver"]["method"] == "Radau"
    spec["solver"]["method"] = "LSODA"
    with pytest.raises(ValueError):
        validate_experiment(spec)
