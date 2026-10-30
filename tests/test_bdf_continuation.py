"""Сохранение истории BDF, контроль ошибки, события и инвалидирование кэша."""

import numpy as np
import pytest

from core.continuation import BDFContinuation
from core.solvers import integrate
from core.status import DomainError, SolverError
from sdk.api import create_session, export_snapshot
from sdk.experiment import validate_experiment


def test_accepted_endpoints_are_exact_even_when_dense_polynomial_rounds():
    from types import SimpleNamespace

    values = np.array([[1e12, 1e12 + 1.0], [1.0, 2.0]])
    result = SimpleNamespace(
        y=values, dense=lambda t: np.nextafter(values[:, int(t)], np.inf)
    )
    plant = create_session()
    for index in (0, 1):
        state = plant._state_at(result, 0.0, 1.0, float(index))
        assert np.array_equal(state, values[:, index])
        state[0] = 0.0
        assert values[0, index] != 0.0


def test_explicit_first_step_is_a_hint_and_error_control_still_refines():
    result = integrate(
        lambda t, y: -100 * y, np.array([1.0]), (0.0, 0.1), rtol=1e-7, atol=1e-10, first_step=0.1
    )
    assert result.n_accepted_steps > 1
    assert result.final[0] == pytest.approx(np.exp(-10), abs=1e-9)
    assert result.t[1] < 0.1
    with pytest.raises(SolverError):
        integrate(lambda t, y: -y, np.array([1.0]), (0.0, 0.1), first_step=np.nan)


def test_continuation_preserves_solver_history_and_dense_solution():
    session = BDFContinuation(
        lambda t, y: -np.array([1.0, 10.0]) * y,
        [1.0, 2.0],
        0.0,
        1.0,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.2,
        first_step=0.1,
    )
    engine = session.solver
    for end in (0.1, 0.3, 1.0):
        result = session.advance_to(end)
        assert session.solver is engine
        assert result.t[-1] == end
        assert result.final == pytest.approx(
            np.exp(-np.array([1.0, 10.0]) * end) * [1.0, 2.0], rel=2e-6, abs=1e-9
        )
        assert result.dense(end) == pytest.approx(result.final)
        assert result.dense(np.array([result.t[0], end])).shape == (2, 2)
        assert result.n_accepted_steps == len(result.t) - 1
        with pytest.raises(SolverError):
            result.dense(end + 1.0)


def test_failed_continuation_cannot_be_reused():
    def broken(t, y):
        if t > 0.1:
            raise DomainError("boundary")
        return -y

    session = BDFContinuation(broken, [1.0], 0.0, 1.0, rtol=1e-6, atol=1e-9, max_step=0.1)
    with pytest.raises(DomainError):
        session.advance_to(0.3)
    with pytest.raises(SolverError):
        session.advance_to(0.4)


def test_plant_restarts_history_on_command_and_restores_on_failure():
    plant = create_session()
    plant.enable_continuation(1.0)
    plant.advance(0.1, first_step_s=0.1)
    first = plant._continuation
    plant.advance(0.1, first_step_s=0.1)
    assert plant._continuation is first
    plant.throttle = 0.9
    plant.advance(0.1, first_step_s=0.1)
    assert plant._continuation is not first
    snapshot = export_snapshot(plant)

    def failed(t, y):
        raise SolverError("injected failure")

    plant.derivatives = failed
    with pytest.raises(SolverError):
        plant.advance(0.1, first_step_s=0.1)
    assert export_snapshot(plant) == snapshot
    assert plant._continuation is None


def test_protection_discontinuity_is_localized_with_continuation():
    plant = create_session()
    plant.enable_continuation(0.5)
    plant.power_protection.setpoint = 0.5 * plant.fission_power_nom
    plant.power_protection.delay_s = 0.15
    plant.advance(0.3, first_step_s=0.1)
    assert plant.last_protection_event_s == pytest.approx(0.15, abs=1e-6)
    assert plant.scram_latched
    assert plant.rod == pytest.approx(plant.rod_ref + plant.scram_speed * 0.15, abs=1e-4)


def test_experiment_continuation_and_first_step_are_explicit_inputs():
    spec = {
        "name": "test",
        "schema_version": 1,
        "engine": "vver",
        "duration_s": 0.2,
        "solver": {"continuation": True, "first_step_s": 1.0},
    }
    result = validate_experiment(spec)
    assert result["solver"]["continuation"] is True
    assert result["solver"]["first_step_s"] == 1.0
    spec["solver"]["continuation"] = "true"
    with pytest.raises(ValueError):
        validate_experiment(spec)
