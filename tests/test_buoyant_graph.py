"""Heated gravity-loop startup, signed energy exchange and strict domain refusal."""

import json
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.buoyant_graph_references import IndependentBuoyant, cases, loop
from scripts.buoyant_inertial_graph import LEDGERS, BuoyantInertialGraph
from scripts.inertial_graph import InertialGraphError, digest
from scripts.thermal_inertial_graph import ThermalInertialGraph


def spec(name="heated_startup"):
    return deepcopy(next(r for n, r, _ in cases() if n == name))


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def test_hot_leg_initial_acceleration_matches_analytic_loop_head():
    r = loop(heated=False, hot_start=True)
    a = BuoyantInertialGraph(r)
    s = a.system
    rate, point = s.evaluate(a._state, a._boundaries)
    # Hot left leg rises10m, cold right leg falls10m; no pump pressure source.
    head = s.rho * s.gravity * 0.001 * 10 * 10
    assert point["qd"] == pytest.approx(np.full(6, head / sum(s.L)), abs=1e-12)
    assert rate[-1] == 0 and point["q"] == pytest.approx(np.zeros(6))


@pytest.mark.parametrize(
    "name,sign", [("heated_startup", 1), ("mirror_startup", -1), ("eight_nodes", 1)]
)
def test_coupled_heater_cooler_starts_circulation_from_rest(name, sign):
    a = BuoyantInertialGraph(spec(name))
    out = a.advance(40)["state"]
    assert sign * out["flows_kg_s"]["e0"] > 0.01
    assert out["buoyancy_work_j"] > 0 and out["kinetic_energy_j"] > 0
    assert abs(out["pressure_work_j"]) < 1e-6
    assert abs(out["total_energy_residual_j"]) < 1e-5


@pytest.mark.parametrize(
    "name", ["hydrostatic_rest", "uniform_heating", "gravity_off", "expansion_off"]
)
def test_controls_never_invent_a_circulation_source(name):
    a = BuoyantInertialGraph(spec(name))
    out = a.advance(40)["state"]
    assert max(abs(q) for q in out["flows_kg_s"].values()) < 1e-10
    assert abs(out["buoyancy_work_j"]) < 1e-8


def test_uniform_warm_hydrostatics_use_gravity_density_without_mass_change():
    r = loop(heated=False)
    for node in r["base"]["thermal"]["nodes"].values():
        node["initial_temperature_k"] = 310
    a = BuoyantInertialGraph(r)
    out = a.advance(5)["state"]
    assert out["pressures_pa"]["n2"] == pytest.approx(200000 - 990 * 9.81 * 10, abs=1e-7)
    assert out["gravity_density_kg_m3"] == pytest.approx({e: 990 for e in a.system.edges})
    assert max(abs(q) for q in out["flows_kg_s"].values()) < 1e-10
    assert np.array_equal(a.system.capacity, 1000 * 1000 * a.system.volumes)


def test_buoyancy_work_is_signed_internal_exchange_not_external_heat():
    r = loop(heated=False, hot_start=True, initial_flow=0.01)
    a = BuoyantInertialGraph(r)
    out = a.advance(5)["state"]
    assert out["buoyancy_work_j"] > 0
    assert out["external_heat_j"] == out["external_thermal_advection_j"] == 0
    assert out["thermal_energy_change_j"] + out["kinetic_energy_j"] == pytest.approx(
        a.system.k0, abs=1e-6
    )
    assert out["thermal_energy_change_j"] == pytest.approx(
        out["friction_to_fluid_j"] - out["buoyancy_work_j"], abs=1e-6
    )


def test_counterflow_has_negative_buoyancy_power_and_heats_internal_energy():
    a = BuoyantInertialGraph(loop(heated=False, hot_start=True, initial_flow=-0.01))
    s = a.system
    rate, point = s.evaluate(a._state, a._boundaries)
    assert rate[-1] < 0
    assert s.capacity @ rate[s.dimension : s.dimension + len(s.names)] == pytest.approx(
        rate[-3] - rate[-1], abs=1e-8
    )
    assert float(np.dot(s.L * point["q"], point["qd"]) / s.rho) < 0


def test_opposed_hot_start_reverses_smoothly_without_resetting_stocks():
    a = BuoyantInertialGraph(spec("opposed_hot_start"))
    assert a.snapshot()["state"]["flows_kg_s"]["e0"] < 0
    out = a.advance(10)["state"]
    assert out["flows_kg_s"]["e0"] > 0
    assert abs(out["total_energy_residual_j"]) < 1e-5


def test_zero_expansion_reduces_to_previous_thermal_backend():
    r = spec("expansion_off")
    a = BuoyantInertialGraph(r)
    b = ThermalInertialGraph(r["base"])
    x, y = a.advance(10)["state"], b.advance(10)["state"]
    assert x["temperatures_k"] == pytest.approx(y["temperatures_k"], abs=1e-9)
    assert x["flows_kg_s"] == pytest.approx(y["flows_kg_s"], abs=1e-9)
    assert x["external_heat_j"] == pytest.approx(y["external_heat_j"], abs=1e-6)


@pytest.mark.parametrize("name", [n for n, _, _ in cases()])
def test_joint_json_restart_every_closed_topology(name):
    r = spec(name)
    a = BuoyantInertialGraph(r, max_step_s=0.1)
    a.advance(3.7)
    b = BuoyantInertialGraph(r, max_step_s=0.1)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(2.3) == b.advance(2.3)


def test_pressure_gauge_step_preserves_thermal_and_mechanical_stocks():
    a = BuoyantInertialGraph(spec("branched_gauge"))
    a.advance(3)
    out = a.step(dict(n0=210000.0), 2, expected_revision=1, command_id="gauge")
    left, right = out["boundary_transition"]["left"], out["boundary_transition"]["right"]
    for field in (
        "flows_kg_s",
        "temperatures_k",
        "kinetic_energy_j",
        "thermal_energy_change_j",
        *LEDGERS,
    ):
        assert left[field] == right[field]
    assert {
        n: right["pressures_pa"][n] - left["pressures_pa"][n] for n in left["pressures_pa"]
    } == pytest.approx({n: 10000 for n in left["pressures_pa"]}, abs=1e-7)


def test_absolute_elevation_shift_changes_neither_head_nor_trajectory():
    r = spec()
    a = BuoyantInertialGraph(r)
    for n in r["base"]["hydraulic"]["nodes"]:
        n["elevation_m"] += 100
    b = BuoyantInertialGraph(r)
    x, y = a.advance(10)["state"], b.advance(10)["state"]
    assert x["flows_kg_s"] == pytest.approx(y["flows_kg_s"], abs=1e-9)
    assert x["temperatures_k"] == pytest.approx(y["temperatures_k"], abs=1e-8)
    assert x["pressures_pa"] == pytest.approx(y["pressures_pa"], abs=1e-6)


def test_buoyant_dimensional_residual_is_pure_for_off_mass_trial():
    a = BuoyantInertialGraph(spec("branched_gauge"))
    a.advance(1)
    s = a.system
    rate, point = s.evaluate(a._state, a._boundaries)
    n = len(s.names)
    y = np.r_[point["q"], point["p"], point["T"], a._state[-7:]]
    dy = np.r_[point["qd"], np.zeros(n), rate[s.dimension : s.dimension + n], rate[-7:]]
    before = a.checkpoint()
    assert max(abs(a.residual(y, dy))) < 1e-6
    y[0] += 0.01
    assert max(abs(a.residual(y, dy))) > 0.001
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("expansion_per_k", -1),
        ("expansion_per_k", True),
        ("reference_temperature_k", 0),
        ("maximum_relative_density_change", 0),
        ("maximum_relative_density_change", 0.1),
    ],
)
def test_invalid_boussinesq_contract_rejected(field, value):
    r = spec()
    r["buoyancy"][field] = value
    with pytest.raises(NetworkError):
        BuoyantInertialGraph(r)


@pytest.mark.parametrize("case", ["mass_injection", "second_anchor", "no_cycle"])
def test_open_or_acyclic_network_is_not_silently_accepted(case):
    r = spec()
    h = r["base"]["hydraulic"]
    if case == "mass_injection":
        h["nodes"][1]["injection_kg_s"] = 1
    if case == "second_anchor":
        h["nodes"][3] = dict(
            id="n3", kind="pressure_boundary_v1", elevation_m=10.0, pressure_pa=101900.0
        )
    if case == "no_cycle":
        h["edges"].pop()
        h["initial_flows_kg_s"].pop("e5")
        r["base"]["edge_laws"].pop("e5")
    with pytest.raises(NetworkError):
        BuoyantInertialGraph(r)


@pytest.mark.parametrize("limit", ["density", "temperature", "rhs", "steps", "commands"])
def test_late_failure_rolls_back_every_buoyancy_and_thermal_stock(limit):
    r = spec("uniform_heating")
    if limit == "density":
        r["buoyancy"]["maximum_relative_density_change"] = 0.001
    if limit == "temperature":
        r["buoyancy"]["expansion_per_k"] = 0
    a = BuoyantInertialGraph(r)
    if limit == "rhs":
        a.budget["maximum_rhs_evaluations"] = 0
    if limit == "steps":
        a.budget["maximum_internal_steps"] = 0
    if limit == "commands":
        a.budget["maximum_commands"] = 0
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(n0=200000.0), 100, expected_revision=0, command_id="retry")
    assert a.checkpoint() == before


@pytest.fixture(scope="module")
def checkpoint():
    a = BuoyantInertialGraph(spec())
    a.advance(3)
    a.step(dict(n0=210000.0), 2, expected_revision=1, command_id="gauge")
    return a.checkpoint()


@pytest.mark.parametrize(
    "change",
    [
        "buoyancy_stock",
        "temperature_stock",
        "thermal_stock",
        "history",
        "density",
        "model",
        "basis",
        "signature",
    ],
)
def test_resealed_checkpoint_corruption_is_atomic(checkpoint, change):
    cp = deepcopy(checkpoint)
    p = cp["payload"]
    if change == "buoyancy_stock":
        p["coordinates"][-1] += 1
    if change == "temperature_stock":
        p["coordinates"][1] += 0.1
    if change == "thermal_stock":
        p["coordinates"][-5] += 1
    if change == "history":
        p["snapshot"]["history"] = []
    if change == "density":
        p["snapshot"]["state"]["gravity_density_kg_m3"]["e0"] = 999
    if change == "model":
        p["model"] = "thermal-inertial-radau-v1"
    if change == "basis":
        p["snapshot"]["contract"]["basis_sha256"] = "bad"
    if change == "signature":
        p["signature"]["python"] = "foreign"
    a = BuoyantInertialGraph(spec())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("expansion_per_k", 0.002),
        ("reference_temperature_k", 301),
        ("maximum_relative_density_change", 0.04),
    ],
)
def test_checkpoint_binds_buoyancy_parameters(checkpoint, field, value):
    r = spec()
    r["buoyancy"][field] = value
    with pytest.raises(NetworkError):
        BuoyantInertialGraph(r).restore(checkpoint)


def test_source_and_final_observation_failures_are_atomic(monkeypatch):
    a = BuoyantInertialGraph(spec())
    before = a.checkpoint()
    original = BuoyantInertialGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("observation fault")
        return original(self)

    with monkeypatch.context() as p:
        p.setattr(BuoyantInertialGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.advance(2)
    assert a.checkpoint() == before
    import scripts.buoyant_inertial_graph as module

    signature = module.signature
    calls = []

    def drift():
        v = signature()
        calls.append(True)
        if len(calls) > 1:
            v["drift"] = True
        return v

    with monkeypatch.context() as p:
        p.setattr(module, "signature", drift)
        with pytest.raises(NetworkError):
            a.advance(2)
    assert a.checkpoint() == before


def test_input_output_checkpoint_are_detached_and_initial_restore_is_exact():
    r = spec()
    a = BuoyantInertialGraph(r)
    initial = a.checkpoint()
    r["buoyancy"]["expansion_per_k"] = 1
    out = a.advance(2)
    out["state"]["temperatures_k"].clear()
    assert a.restore(initial) == initial["payload"]["snapshot"]


@pytest.mark.parametrize(
    "name,seed", [("heated_startup", 0.13), ("mirror_startup", -0.13), ("eight_nodes", 0.13)]
)
def test_independent_nontrivial_stationary_branch_is_held_by_production(name, seed):
    r = spec(name)
    steady = IndependentBuoyant(r).stationary(seed)
    r["base"]["hydraulic"]["initial_flows_kg_s"] = steady["flows_kg_s"]
    for node, T in steady["temperatures_k"].items():
        r["base"]["thermal"]["nodes"][node]["initial_temperature_k"] = T
    a = BuoyantInertialGraph(r)
    before = a.snapshot()["state"]
    out = a.advance(10)["state"]
    assert out["flows_kg_s"] == pytest.approx(before["flows_kg_s"], abs=1e-8)
    assert out["temperatures_k"] == pytest.approx(before["temperatures_k"], abs=1e-7)


def test_excessive_dissipation_number_is_refused_as_outside_reduced_closure():
    r = spec()
    r["base"]["thermal"]["heat_capacity_j_kg_k"] = 0.01
    with pytest.raises(NetworkError):
        BuoyantInertialGraph(r)


def test_missing_thermal_work_compensation_is_detected_and_rolled_back(monkeypatch):
    from scripts.buoyant_inertial_graph import BuoyantSystem
    from scripts.thermal_inertial_graph import ThermalInertialSystem

    a = BuoyantInertialGraph(loop(heated=False, hot_start=True, initial_flow=0.01))
    before = a.checkpoint()

    def broken(self, q, T, p):
        rate, ledger, loss, multipliers = ThermalInertialSystem.thermal_terms(self, q, T, p)
        force, _ = self.buoyancy(T)
        return rate, np.r_[ledger, q @ force / self.rho], loss, multipliers

    with monkeypatch.context() as patch:
        patch.setattr(BuoyantSystem, "thermal_terms", broken)
        with pytest.raises(InertialGraphError) as error:
            a.advance(5)
    assert error.value.report["criterion"] == "energy" and a.checkpoint() == before


@pytest.mark.parametrize("revision", [True, 0, 2, 1.0])
def test_revision_checked_before_pressure_gauge_transaction(revision):
    a = BuoyantInertialGraph(spec())
    a.advance(1)
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(n0=210000.0), 1, expected_revision=revision, command_id="stale")
    assert a.checkpoint() == before


def test_invalid_pressure_gauge_cannot_partially_reinitialize_loop():
    a = BuoyantInertialGraph(spec())
    a.advance(1)
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(n0=50000.0), 1, expected_revision=1, command_id="pressure-domain")
    assert a.checkpoint() == before


def test_restore_final_observation_fault_is_atomic(checkpoint, monkeypatch):
    a = BuoyantInertialGraph(spec())
    before = a.checkpoint()
    original = BuoyantInertialGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("restore fault")
        return original(self)

    with monkeypatch.context() as p:
        p.setattr(BuoyantInertialGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.restore(checkpoint)
    assert a.checkpoint() == before
