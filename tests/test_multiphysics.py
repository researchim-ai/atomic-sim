"""Independent conservation, exact references, source restart and strict domains."""

import json

import numpy as np
import pytest
from scipy.linalg import expm

from atomic_model_multiphysics.absorber import insert, study
from atomic_model_multiphysics.agent import SpatialAgent
from atomic_model_multiphysics.coupled import map_power, map_temperature, solve, thermal
from atomic_model_multiphysics.depletion import advance, operator
from atomic_model_multiphysics.feedback import AXES, FeedbackTable, numerical_table
from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_multiphysics.qualification import qualify_depletion, uncertainty_report
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab


def test_nonmatching_mesh_power_conservation_and_constant_temperature():
    a = np.array([0, 0.1, 0.4, 1.0])
    b = np.linspace(0, 1, 8)
    q = np.array([4.0, 5.0, 6.0])
    assert sum(map_power(a, b, q)) == pytest.approx(sum(q), abs=1e-13)
    np.testing.assert_allclose(map_temperature(a, b, [560.0] * 3), 560.0)
    with pytest.raises(ValueError):
        map_power(a, b * 2, q)
    with pytest.raises(ValueError):
        map_temperature(a, b, [560.0, float("nan"), 560.0])


@pytest.mark.parametrize("name", AXES)
def test_feedback_axes_refuse_extrapolation(name):
    table = FeedbackTable(numerical_table())
    state = dict(
        fuel_temperature_k=600.0,
        coolant_temperature_k=560.0,
        density_kg_m3=750.0,
        boron_ppm=0.0,
        composition_fraction=0.0,
    )
    assert table(**state) == pytest.approx(1.0)
    state[name] = 1e10
    with pytest.raises(ValueError):
        table(**state)


def test_feedback_tensor_and_provenance_strictness():
    spec = numerical_table()
    spec["values"][0][0][0][0][0] = -1
    with pytest.raises(ValueError):
        FeedbackTable(spec)
    spec = numerical_table()
    spec["provenance"] = ""
    with pytest.raises(ValueError):
        FeedbackTable(spec)


@pytest.mark.parametrize("cells", [1, 5, 12])
def test_channel_and_rod_energy_independent_enthalpy(cells):
    q = np.linspace(100, 400, cells)
    result = thermal(np.linspace(0, 1, cells + 1), q)
    assert result["energy_relative"] < 1e-10
    assert 0.3 * 4200 * (result["channel_outlet_k"][-1] - 560) == pytest.approx(sum(q), rel=1e-12)
    np.testing.assert_allclose(result["boundary_heat_w"], q, rtol=1e-10)


def test_coupled_iteration_residual_and_failure_transaction():
    config = slab(6)
    spec = numerical_table()
    result = solve(config, spec)
    assert result["iterations"][-1]["temperature_error_k"] <= 1e-5
    assert sum(result["cell_power_w"]) == pytest.approx(5000.0, rel=1e-14)
    with pytest.raises(ValueError, match="converge"):
        solve(config, spec, max_iterations=1)
    with pytest.raises(ValueError):
        solve(config, spec, total_power_w=1e8)


def test_spatial_agent_full_restart_and_rejected_action_is_transactional():
    agent = SpatialAgent(slab(5), numerical_table())
    agent.step(dict(thermal_power_w=5000), expected_revision=0)
    saved = json.loads(json.dumps(agent.checkpoint()))
    assert SpatialAgent.restore(saved).checkpoint() == saved
    with pytest.raises(ValueError):
        agent.step(dict(thermal_power_w=10001), expected_revision=1)
    with pytest.raises(ValueError):
        agent.step(dict(thermal_power_w=5000), expected_revision=0)
    assert agent.checkpoint() == saved
    saved["payload"]["session"]["payload"]["source"] = {}
    with pytest.raises(ValueError):
        SpatialAgent.restore(saved)


@pytest.mark.parametrize("n", [[1.0, 0.0, 0.0], [2.0, 3.0, 4.0], [1e9, 2e8, 4e7]])
def test_cram_chain_and_released_energy_against_exponential(n):
    a = operator([0.1, 0.03, 0], [0, 0, 0], 0)
    result = advance(a, n, 10, [2.0, 3.0, 0.0])
    exact = expm(a * 10) @ n
    np.testing.assert_allclose(result["composition"], exact, rtol=1e-10, atol=1e-12)
    # Energy for two-step chain equals weighted completed transitions.
    energy = 2 * (n[0] - exact[0]) + 3 * (exact[2] - n[2])
    assert result["released_energy_j"] == pytest.approx(energy, rel=1e-10)


def test_cecm_order_and_multiple_composition_qualification():
    assert qualify_depletion()["accepted"]


def test_absorber_overlap_no_mesh_snap_and_manufactured_peaks():
    config = slab(8)
    model = Model(config)
    changed = Model(insert(config, 0.371, 0.2))
    assert np.sum(
        (changed.absorption_per_m[0] - model.absorption_per_m[0]) * model.width
    ) == pytest.approx(0.2 * 0.371)
    assert study()["accepted"]


def test_uncertainty_does_not_invent_measurement_covariance_or_cover():
    result = uncertainty_report()
    assert result["validation_rows"] == 24
    assert not result["experimental_coverage_accepted"]
    assert not result["parameter_fitting_performed"]
    assert set(result["calibration_cases"]).isdisjoint(result["validation_cases"])


@pytest.mark.parametrize(
    "mutation", [lambda p: p.update(sha256="x"), lambda p: p["payload"].update(x=2)]
)
def test_envelope_tampering(mutation):
    value = seal(dict(x=1))
    mutation(value)
    with pytest.raises(ValueError):
        unseal(value)


@pytest.mark.parametrize(
    "change",
    [
        {"particles": True},
        {"seed": 0},
        {"batches": 501},
        {"temperature_k": 901.0},
        {"density_kg_m3": float("nan")},
        {"width_m": 0.0},
        {"energy_bounds_ev_ascending": [2e7, 0.625, 1e-5]},
        {"nuclides": {"U235": 1.0}},
        {"extra": "not allowed"},
    ],
)
def test_transport_strict_admission(change):
    from atomic_model_multiphysics.nuclear import specification
    from atomic_model_multiphysics.transport import admit

    spec = specification()
    spec.update(change)
    with pytest.raises(ValueError):
        admit(spec)


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[-1.0, 0.0], [-1.0, 0.0]]),
        np.array([[float("nan"), 0.0], [0.0, 0.0]]),
    ],
)
def test_cecm_rejects_invalid_transition_operator(invalid):
    from atomic_model_multiphysics.depletion import cecm

    with pytest.raises(ValueError):
        cecm([1.0, 0.0], 0.1, lambda state: invalid)


def test_evaluated_model_coupling_requires_explicit_heat_response():
    config = slab(4)
    config["provenance"] = "Evaluated input, intentionally no heat-response data"
    with pytest.raises(ValueError, match="kappa-fission"):
        solve(config, numerical_table())
    result = solve(config, numerical_table(), heat_response_per_m=Model(config).nu_fission_per_m)
    assert result["thermal"]["energy_relative"] < 1e-9


def test_real_evaluated_mgxs_reference_axis_and_unit_conversion():
    from atomic_model_multiphysics.nuclear import model_from_mgxs
    from atomic_model_multiphysics.resources import reference_mgxs

    source = reference_mgxs()
    model = model_from_mgxs(source, cells=4)
    p = source["payload"]
    assert model.absorption_per_m[0, 0] == pytest.approx(p["means"]["absorption"][0] * 100)
    assert model.scatter[1, 0, 0] == pytest.approx(p["means"]["scatter matrix"][0][1] * 100)
    assert model.diffusion_m[0, 0] == pytest.approx(p["means"]["diffusion-coefficient"][0] * 0.01)
    assert model.speed_m_s[0, 0] == pytest.approx(1 / (p["means"]["inverse-velocity"][0] * 100))
    assert p["source_family"] == "evaluated" and p["k_std"] > 0
    source["payload"]["source_family"] = "numerical"
    source = seal(source["payload"])
    with pytest.raises(ValueError):
        model_from_mgxs(source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inlet_k": float("nan")},
        {"mass_flow_kg_s": float("inf")},
        {"cp_j_kg_k": float("nan")},
        {"channel_faces": [0.0, float("inf")]},
    ],
)
def test_thermal_nonfinite_boundaries_refused(kwargs):
    values = dict(channel_faces=[0.0, 1.0], powers=[100.0])
    values.update(kwargs)
    with pytest.raises(ValueError):
        thermal(**values)


def test_wrong_feedback_quantity_and_negative_physical_axis_refused():
    spec = numerical_table()
    spec["quantity"] = "fission power"
    with pytest.raises(ValueError):
        FeedbackTable(spec)
    spec = numerical_table()
    spec["axes"]["fuel_temperature_k"] = [-100.0, 1600.0]
    with pytest.raises(ValueError):
        FeedbackTable(spec)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(k_std=float("nan")),
        lambda p: p["axes"].update(diffusion_unit="mm"),
        lambda p: p["standard_errors"].pop("chi"),
        lambda p: p["means"]["scatter matrix"][0].__setitem__(0, -1.0),
    ],
)
def test_evaluated_mgxs_uncertainty_inventory_and_units_refused(mutation):
    from atomic_model_multiphysics.nuclear import model_from_mgxs
    from atomic_model_multiphysics.resources import reference_mgxs

    payload = reference_mgxs()["payload"]
    mutation(payload)
    # Nonfinite envelope is itself inadmissible under canonical serialization.
    with pytest.raises(ValueError):
        model_from_mgxs(seal(payload))
