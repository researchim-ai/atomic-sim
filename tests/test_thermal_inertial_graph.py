"""Conservative two-way coupling, heat recovery and exact joint restart."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.inertial_graph import InertialGraph, digest
from scripts.inertial_graph_references import cases as hydraulic_cases
from scripts.thermal_inertial_graph import LEDGERS, ThermalInertialGraph
from scripts.thermal_inertial_references import cases, request


def spec(name="single_pipe"):
    return deepcopy(next(r for n, r, _ in cases() if n == name))


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def closed(recovery=1.0, beta=0.0):
    h = next(s for n, s, _ in hydraulic_cases() if n == "free_cycle_decay")
    r = request(h, beta=beta, recovery=recovery, heat=0, volume=0.001)
    for n in r["thermal"]["nodes"].values():
        n["ambient_ua_w_k"] = 0
        n["supply_temperature_k"] = 300
    return r


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.6, 1.0])
def test_closed_decay_distributes_initial_kinetic_energy_once(fraction):
    a = ThermalInertialGraph(closed(fraction))
    r = a.advance(20)["state"]
    lost = 60 * (1 - math.exp(-4))
    assert r["kinetic_energy_j"] == pytest.approx(60 * math.exp(-4), abs=1e-7)
    assert r["friction_to_fluid_j"] == pytest.approx(fraction * lost, abs=1e-7)
    assert r["exported_friction_heat_j"] == pytest.approx((1 - fraction) * lost, abs=1e-7)
    assert r["thermal_energy_change_j"] == pytest.approx(fraction * lost, abs=1e-7)
    for T in r["temperatures_k"].values():
        assert T == pytest.approx(300 + fraction * lost / 3000, abs=1e-9)
    assert abs(r["total_energy_residual_j"]) < 1e-7
    assert abs(r["external_thermal_advection_j"]) < 1e-7


def test_zero_feedback_and_exported_friction_recover_mechanical_backend():
    r = spec()
    a = ThermalInertialGraph(r)
    b = InertialGraph(r["hydraulic"])
    out = a.advance(12)["state"]
    ref = b.advance(12)["state"]
    assert out["flows_kg_s"] == pytest.approx(ref["flows_kg_s"], abs=1e-9)
    assert out["pressure_work_j"] == pytest.approx(ref["pressure_work_j"], abs=1e-7)
    assert out["exported_friction_heat_j"] == pytest.approx(ref["exported_heat_j"], abs=1e-7)
    assert out["temperatures_k"] == dict(a=300.0, b=300.0)


def test_isolated_heated_inventory_matches_exact_temperature_ramp():
    r = closed(0)
    r["hydraulic"]["initial_flows_kg_s"] = dict(ab=0.0, bc=0.0, ca=0.0)
    r["thermal"]["nodes"]["a"]["heat_power_w"] = 2000
    a = ThermalInertialGraph(r)
    out = a.advance(7)["state"]
    assert out["temperatures_k"] == pytest.approx(dict(a=314.0, b=300.0, c=300.0), abs=1e-9)
    assert out["thermal_energy_change_j"] == pytest.approx(14000, abs=1e-6)
    assert out["external_heat_j"] == pytest.approx(14000, abs=1e-6)


def test_internal_upwind_transport_preserves_closed_total_thermal_energy():
    r = closed(0)
    r["thermal"]["nodes"]["a"]["initial_temperature_k"] = 330
    a = ThermalInertialGraph(r)
    out = a.advance(10)["state"]
    assert abs(out["thermal_energy_change_j"]) < 1e-6
    assert out["temperatures_k"]["a"] < 330 and out["temperatures_k"]["b"] > 300
    assert abs(out["external_thermal_advection_j"]) < 1e-6


def test_heating_changes_resistance_and_inertial_flow_without_mutating_base_coefficients():
    r = spec("series_junction")
    a = ThermalInertialGraph(r)
    r0 = a.system.R.copy()
    reference = spec("series_junction")
    for law in reference["edge_laws"].values():
        law["temperature_coefficient_per_k"] = 0
    b = ThermalInertialGraph(reference)
    a.advance(20)
    b.advance(20)
    assert max(a.snapshot()["state"]["resistance_multipliers"].values()) > 1.01
    assert a.snapshot()["state"]["flows_kg_s"]["aj"] < b.snapshot()["state"]["flows_kg_s"]["aj"]
    assert np.array_equal(a.system.R, r0)


def test_constant_density_heated_hydrostatic_column_does_not_invent_buoyancy():
    a = ThermalInertialGraph(spec("hydrostatic_column"))
    out = a.advance(20)["state"]
    assert out["temperatures_k"]["a"] > 300
    assert out["flows_kg_s"] == dict(aj=0.0, jb=0.0) and out["gravity_work_j"] == 0


@pytest.mark.parametrize("name", [n for n, _, _ in cases()])
def test_joint_json_checkpoint_exact_continuation_all_topologies(name):
    r = spec(name)
    a = ThermalInertialGraph(r, max_step_s=0.1)
    b = ThermalInertialGraph(r, max_step_s=0.1)
    a.advance(3.7)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(2.3) == b.advance(2.3)


def test_boundary_command_preserves_temperatures_kinetic_energy_and_all_ledgers():
    a = ThermalInertialGraph(spec("boundary_reversal"))
    a.advance(10)
    result = a.step(dict(a=198000.0, b=200000.0), 5, expected_revision=1, command_id="reverse")
    left, right = result["boundary_transition"]["left"], result["boundary_transition"]["right"]
    for field in (
        "flows_kg_s",
        "temperatures_k",
        "temperature_increments_k",
        "kinetic_energy_j",
        "thermal_energy_change_j",
        *LEDGERS,
    ):
        assert left[field] == right[field]
    assert result["state"]["flows_kg_s"]["aj"] < 0
    b = ThermalInertialGraph(spec("boundary_reversal"))
    assert b.restore(a.checkpoint()) == a.snapshot()
    assert a.advance(2) == b.advance(2)


def test_enthalpy_reference_change_leaves_physics_and_net_energy_unchanged():
    r = spec("series_junction")
    a = ThermalInertialGraph(r)
    r["thermal"]["reference_temperature_k"] = 200
    b = ThermalInertialGraph(r)
    x, y = a.advance(10)["state"], b.advance(10)["state"]
    assert x["temperatures_k"] == pytest.approx(y["temperatures_k"], abs=1e-9)
    assert x["flows_kg_s"] == pytest.approx(y["flows_kg_s"], abs=1e-9)
    assert x["external_thermal_advection_j"] == pytest.approx(
        y["external_thermal_advection_j"], abs=1e-5
    )


def test_dimensional_residual_is_pure_and_handles_off_manifold_trial():
    a = ThermalInertialGraph(spec("parallel_loop"))
    a.advance(1)
    s = a.system
    rate, point = s.evaluate(a._state, a._boundaries)
    n = len(s.names)
    y = np.r_[point["q"], point["p"], point["T"], a._state[-6:]]
    dy = np.r_[point["qd"], np.zeros(n), rate[s.dimension : s.dimension + n], rate[-6:]]
    original = y.copy()
    before = a.checkpoint()
    assert max(abs(a.residual(y, dy))) < 1e-6
    y[0] += 0.01
    assert max(abs(a.residual(y, dy))) > 0
    assert a.checkpoint() == before and np.array_equal(
        original, np.r_[point["q"], point["p"], point["T"], a._state[-6:]]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("volume_m3", 0),
        ("initial_temperature_k", 600),
        ("heat_power_w", True),
        ("ambient_ua_w_k", -1),
        ("supply_temperature_k", 600),
    ],
)
def test_invalid_node_contract_rejected(field, value):
    r = spec()
    r["thermal"]["nodes"]["a"][field] = value
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fluid_heat_fraction", -0.1),
        ("fluid_heat_fraction", 1.1),
        ("minimum_multiplier", 0),
        ("maximum_multiplier", 0.01),
        ("reference_temperature_k", 0),
        ("temperature_coefficient_per_k", float("nan")),
    ],
)
def test_invalid_loss_or_recovery_law_rejected(field, value):
    r = spec()
    r["edge_laws"]["ab"][field] = value
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r)


@pytest.mark.parametrize("case", ["temperature", "multiplier", "rhs", "steps", "commands"])
def test_late_coupled_failure_rolls_back_all_thermal_and_mechanical_stocks(case):
    r = spec("series_junction")
    if case == "temperature":
        r["thermal"]["max_temperature_k"] = 330.01
        r["thermal"]["nodes"]["a"]["heat_power_w"] = 1e8
    if case == "multiplier":
        for law in r["edge_laws"].values():
            law["maximum_multiplier"] = 1.001
    a = ThermalInertialGraph(r)
    if case == "rhs":
        a.budget["maximum_rhs_evaluations"] = 0
    if case == "steps":
        a.budget["maximum_internal_steps"] = 0
    if case == "commands":
        a.budget["maximum_commands"] = 0
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=203000.0, b=200000.0), 10, expected_revision=0, command_id="retry")
    assert a.checkpoint() == before


@pytest.fixture(scope="module")
def checkpoint():
    a = ThermalInertialGraph(spec("series_junction"))
    a.advance(2)
    a.step(dict(a=198000.0, b=200000.0), 2, expected_revision=1, command_id="r")
    return a.checkpoint()


@pytest.mark.parametrize(
    "change",
    [
        "temperature",
        "fluid_heat",
        "bath_heat",
        "total_heat",
        "history",
        "basis",
        "model",
        "signature",
    ],
)
def test_resealed_checkpoint_corruption_is_atomic(checkpoint, change):
    cp = deepcopy(checkpoint)
    p = cp["payload"]
    if change == "temperature":
        p["coordinates"][1] += 0.1
    if change == "fluid_heat":
        p["coordinates"][-2] = -100
    if change == "bath_heat":
        p["coordinates"][-1] = -100
    if change == "total_heat":
        p["coordinates"][-4] += 100
    if change == "history":
        p["snapshot"]["history"] = []
    if change == "basis":
        p["snapshot"]["contract"]["basis_sha256"] = "bad"
    if change == "model":
        p["model"] = "inertial-pipe-radau-v1"
    if change == "signature":
        p["signature"]["python"] = "foreign"
    a = ThermalInertialGraph(spec("series_junction"))
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value", [("reference_temperature_k", 290), ("heat_capacity_j_kg_k", 2000)]
)
def test_checkpoint_binds_thermal_frame(checkpoint, field, value):
    r = spec("series_junction")
    r["thermal"][field] = value
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r).restore(checkpoint)


def test_input_output_and_checkpoint_are_detached():
    r = spec()
    a = ThermalInertialGraph(r)
    r["thermal"]["nodes"]["a"]["heat_power_w"] = 1e9
    out = a.advance(1)
    out["state"]["temperatures_k"].clear()
    cp = a.checkpoint()
    cp["payload"]["coordinates"].clear()
    assert a.snapshot()["state"]["temperatures_k"] == dict(a=300.0, b=300.0)


def test_stale_revision_duplicate_id_and_source_fault_never_commit(monkeypatch):
    a = ThermalInertialGraph(spec())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=202000.0, b=200000.0), 1, expected_revision=True, command_id="id")
    assert a.checkpoint() == before
    import scripts.thermal_inertial_graph as module

    original = module.signature
    calls = []

    def drift():
        v = original()
        calls.append(True)
        if len(calls) > 1:
            v["drift"] = True
        return v

    with monkeypatch.context() as patch:
        patch.setattr(module, "signature", drift)
        with pytest.raises(NetworkError):
            a.advance(1)
    assert a.checkpoint() == before


def test_final_observation_and_restore_fault_leave_original_intact(checkpoint, monkeypatch):
    a = ThermalInertialGraph(spec("series_junction"))
    before = a.checkpoint()
    original = ThermalInertialGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("observation fault")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(ThermalInertialGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.advance(2)
        with pytest.raises(NetworkError):
            a.restore(checkpoint)
    assert a.checkpoint() == before


def test_imposed_reverse_flow_uses_incoming_supply_and_outgoing_node_temperature():
    h = next(s for n, s, _ in hydraulic_cases() if n == "injection_tree")
    r = request(h, beta=0, recovery=0, heat=0, volume=0.001)
    for n in r["thermal"]["nodes"].values():
        n["ambient_ua_w_k"] = 0
        n["supply_temperature_k"] = 330
    # Node j receives 1 kg/s from its declared injection and sends it towards a.
    a = ThermalInertialGraph(r)
    out = a.advance(2)["state"]
    assert out["temperatures_k"]["j"] == pytest.approx(330 - 30 * math.exp(-2), abs=1e-8)
    assert out["temperatures_k"]["a"] > 300
    assert abs(out["thermal_energy_residual_j"]) < 1e-6


def test_closed_hot_cycle_with_feedback_preserves_total_kinetic_plus_thermal_energy():
    r = closed(1.0, beta=0.02)
    r["thermal"]["nodes"]["a"]["initial_temperature_k"] = 310
    a = ThermalInertialGraph(r)
    out = a.advance(20)["state"]
    assert out["thermal_energy_change_j"] + out["kinetic_energy_j"] == pytest.approx(60, abs=1e-6)
    assert out["exported_friction_heat_j"] == 0


@pytest.mark.parametrize(
    "field,value", [("volume_m3", 0.04), ("heat_power_w", 123), ("supply_temperature_k", 320)]
)
def test_checkpoint_binds_all_thermal_inputs(checkpoint, field, value):
    r = spec("series_junction")
    r["thermal"]["nodes"]["a"][field] = value
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r).restore(checkpoint)


@pytest.mark.parametrize(
    "field,value", [("fluid_heat_fraction", 0.3), ("temperature_coefficient_per_k", 0.02)]
)
def test_checkpoint_binds_feedback_and_recovery(checkpoint, field, value):
    r = spec("series_junction")
    r["edge_laws"]["aj"][field] = value
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r).restore(checkpoint)


@pytest.mark.parametrize("missing", ["thermal_node", "edge_law", "edge_endpoint"])
def test_incomplete_coupled_contract_is_rejected(missing):
    r = spec()
    if missing == "thermal_node":
        r["thermal"]["nodes"].pop("a")
    if missing == "edge_law":
        r["edge_laws"].clear()
    if missing == "edge_endpoint":
        r["hydraulic"]["edges"][0].pop("from")
    with pytest.raises(NetworkError):
        ThermalInertialGraph(r)


def test_pressure_trial_affects_pressure_work_residual_and_boundary_equation():
    a = ThermalInertialGraph(spec())
    s = a.system
    rate, point = s.evaluate(a._state, a._boundaries)
    # Use a valid nonzero flow and an off-manifold trial pressure.
    a.advance(1)
    rate, point = s.evaluate(a._state, a._boundaries)
    n = len(s.names)
    m = len(s.edges)
    y = np.r_[point["q"], point["p"], point["T"], a._state[-6:]]
    dy = np.r_[point["qd"], np.zeros(n), rate[s.dimension : s.dimension + n], rate[-6:]]
    base = a.residual(y, dy)
    y[m] += 1000
    changed = a.residual(y, dy)
    assert changed[m] - base[m] == pytest.approx(1000)
    assert abs(changed[-6] - base[-6]) > 0


def test_checkpoint_initial_zero_stocks_restore_after_nonzero_thermal_history():
    a = ThermalInertialGraph(spec("parallel_loop"))
    cp = a.checkpoint()
    a.advance(3)
    assert a.restore(cp) == cp["payload"]["snapshot"]
