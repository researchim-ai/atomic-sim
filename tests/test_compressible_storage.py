"""EOS/Gibbs identities, real mass/volume storage and first/second-law guards."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.compressible_storage_graph import CompressibleStorageGraph, StiffenedCaloricEOS
from scripts.inertial_graph import InertialGraphError, digest
from scripts.storage_graph_references import cases, fluid, link, node, request


def spec(name="mass_energy_equalization"):
    return deepcopy(next(r for n, r, _ in cases() if n == name))


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


@pytest.mark.parametrize("shift", [0.0, 1e5, 1e8])
def test_eos_round_trip_rho_u_and_pressure_temperature(shift):
    eos = StiffenedCaloricEOS(fluid(shift))
    a = eos.p_T(200000.0, 320.0)
    b = eos.rho_u(a["density_kg_m3"], a["internal_energy_j_kg"])
    assert b["pressure_pa"] == pytest.approx(200000.0, abs=1e-6)
    assert b["temperature_k"] == pytest.approx(320, abs=1e-10)
    assert a["enthalpy_j_kg"] == pytest.approx(
        a["internal_energy_j_kg"] + a["pressure_pa"] / a["density_kg_m3"], abs=1e-7
    )
    assert a["sound_speed_squared_m2_s2"] > 0


def test_gibbs_identity_and_isothermal_density_derivatives():
    eos = StiffenedCaloricEOS(fluid(1e8))
    a = eos.p_T(2e5, 300)
    rho = a["density_kg_m3"]
    dr = rho * 1e-5
    dt = 0.001
    up = eos.rho_T(rho + dr, 300 + dt)
    down = eos.rho_T(rho - dr, 300 - dt)
    du = (up["internal_energy_j_kg"] - down["internal_energy_j_kg"]) / 2
    ds = (up["entropy_j_kg_k"] - down["entropy_j_kg_k"]) / 2
    dv = (1 / (rho + dr) - 1 / (rho - dr)) / 2
    assert du == pytest.approx(300 * ds - a["pressure_pa"] * dv, rel=1e-6, abs=1e-6)
    dp = 100
    derivative = (
        eos.p_T(2e5 + dp, 300)["density_kg_m3"] - eos.p_T(2e5 - dp, 300)["density_kg_m3"]
    ) / (2 * dp)
    assert derivative == pytest.approx(a["drho_dp_at_T"], rel=1e-9)


def test_isochoric_heating_matches_analytic_pressure_temperature_and_entropy():
    r = spec("sealed_heat")
    a = CompressibleStorageGraph(r)
    initial = a.snapshot()["state"]
    out = a.advance(20)["state"]
    m = initial["masses_kg"]["a"]
    cv = 1000.0
    T = 300 + 2000 / (m * cv)
    assert out["temperatures_k"]["a"] == pytest.approx(T, abs=1e-9)
    assert out["pressures_pa"]["a"] == pytest.approx(208000, abs=1e-7)
    assert out["entropy_change_j_k"] == pytest.approx(m * cv * math.log(T / 300), abs=1e-8)
    assert out["generated_entropy_j_k"] == 0 and abs(out["energy_residual_j"]) < 1e-7


@pytest.mark.parametrize("name", ["adiabatic_expansion", "adiabatic_compression"])
def test_piston_adiabatic_isentrope_has_true_density_and_pressure_work(name):
    a = CompressibleStorageGraph(spec(name))
    initial = a.snapshot()["state"]
    out = a.advance(20)["state"]
    V = out["volumes_m3"]["a"]
    V0 = 0.1
    assert out["masses_kg"] == initial["masses_kg"]
    assert out["densities_kg_m3"]["a"] == pytest.approx(initial["masses_kg"]["a"] / V)
    assert out["temperatures_k"]["a"] == pytest.approx(300 * (V0 / V) ** 0.4, abs=1e-8)
    assert out["pressures_pa"]["a"] + 1e5 == pytest.approx(3e5 * (V0 / V) ** 1.4, abs=1e-5)
    assert out["internal_energy_change_j"] == pytest.approx(-out["piston_work_j"], abs=1e-6)
    assert abs(out["entropy_change_j_k"]) < 1e-7 and out["generated_entropy_j_k"] == 0


def test_isobaric_heated_expansion_matches_cp_and_mass_conservation():
    r = request([node("a", velocity=0.001, heat=1050.0)])
    a = CompressibleStorageGraph(r)
    out = a.advance(20)["state"]
    assert out["pressures_pa"]["a"] == pytest.approx(200000, abs=1e-6)
    assert out["temperatures_k"]["a"] == pytest.approx(360, abs=1e-8)
    assert out["densities_kg_m3"]["a"] == pytest.approx(2.5 / 1.2, abs=1e-10)
    assert out["internal_energy_change_j"] == pytest.approx(17000, abs=1e-6)
    assert out["piston_work_j"] == pytest.approx(4000, abs=1e-6)


def test_two_volumes_exchange_mass_energy_and_produce_entropy():
    a = CompressibleStorageGraph(spec())
    before = a.snapshot()["state"]
    after = a.advance(20)["state"]
    assert after["masses_kg"] != before["masses_kg"]
    assert abs(after["mass_change_kg"]) < 1e-10 and abs(after["internal_energy_change_j"]) < 1e-6
    assert after["generated_entropy_j_k"] > 0
    assert after["entropy_change_j_k"] == pytest.approx(after["generated_entropy_j_k"], abs=1e-7)
    assert abs(after["pressures_pa"]["a"] - after["pressures_pa"]["b"]) < 1000


def test_heating_high_stiffness_storage_uses_actual_eos_density():
    a = CompressibleStorageGraph(spec("stiffened_storage"))
    before = a.snapshot()["state"]
    after = a.advance(20)["state"]
    assert after["densities_kg_m3"] != before["densities_kg_m3"]
    assert after["pressures_pa"] != before["pressures_pa"]
    assert abs(after["energy_residual_j"]) < 1e-5 and abs(after["entropy_residual_j_k"]) < 1e-7


def test_heat_conduction_produces_entropy_even_at_initial_equal_pressure():
    r = request([node("a", T=340), node("b", T=280)], [link("ab", "a", "b")])
    a = CompressibleStorageGraph(r)
    rate, point = a.system.evaluate(0, a._state, a._heaters)
    assert point["q"][0] == 0 and point["generation"] > 0
    assert sum(rate[:2]) == 0 and sum(rate[2:4]) == 0


def test_energy_and_entropy_reference_changes_leave_physics_unchanged():
    r = spec()
    a = CompressibleStorageGraph(r)
    r["fluid"]["energy_offset_j_kg"] = 1e6
    r["fluid"]["entropy_reference_density_kg_m3"] = 10
    r["fluid"]["entropy_reference_temperature_k"] = 500
    b = CompressibleStorageGraph(r)
    x, y = a.advance(10)["state"], b.advance(10)["state"]
    for field in ("pressures_pa", "temperatures_k", "densities_kg_m3", "masses_kg"):
        assert x[field] == y[field]
    assert x["entropy_change_j_k"] == pytest.approx(y["entropy_change_j_k"], abs=1e-8)
    assert x["internal_energy_change_j"] == pytest.approx(y["internal_energy_change_j"], abs=1e-7)


@pytest.mark.parametrize("name", [n for n, _, _ in cases()])
def test_json_checkpoint_continuation_every_storage_program(name):
    r = spec(name)
    a = CompressibleStorageGraph(r)
    a.advance(3.7)
    b = CompressibleStorageGraph(r)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(2.3) == b.advance(2.3)


def test_heater_switch_changes_no_inventory_or_derived_eos_state():
    a = CompressibleStorageGraph(spec())
    a.advance(2)
    result = a.step(dict(a=100, b=-50), 1, expected_revision=1, command_id="heat")
    left, right = result["heater_transition"]["left"], result["heater_transition"]["right"]
    for field in (
        "masses_kg",
        "volumes_m3",
        "pressures_pa",
        "temperatures_k",
        "internal_energies_j",
        "internal_energy_change_j",
        "entropy_change_j_k",
    ):
        assert left[field] == right[field]
    b = CompressibleStorageGraph(spec())
    assert b.restore(a.checkpoint()) == a.snapshot()
    assert a.advance(1) == b.advance(1)


def test_dimensional_residual_is_pure_and_detects_mass_rate_error():
    a = CompressibleStorageGraph(spec())
    a.advance(1)
    before = a.checkpoint()
    y = a._state.copy()
    rate = a.system.evaluate(a._time, y, a._heaters)[0]
    assert np.array_equal(a.residual(a._time, y, rate), np.zeros(len(y)))
    rate[0] += 1
    assert a.residual(a._time, y, rate)[0] == 1
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("cv_j_kg_k", 0),
        ("gamma", 1),
        ("gamma", True),
        ("pressure_shift_pa", -1),
        ("entropy_reference_temperature_k", 0),
        ("entropy_reference_density_kg_m3", 0),
        ("kind", "water-calibrated"),
    ],
)
def test_invalid_eos_contract(field, value):
    r = spec()
    r["fluid"][field] = value
    with pytest.raises(NetworkError):
        CompressibleStorageGraph(r)


@pytest.mark.parametrize(
    "case",
    [
        "negative_density",
        "negative_temperature",
        "invalid_shifted_pressure",
        "invalid_internal_energy",
    ],
)
def test_eos_inverse_rejects_invalid_state(case):
    e = StiffenedCaloricEOS(fluid())
    with pytest.raises(NetworkError):
        if case == "negative_density":
            e.rho_T(-1, 300)
        elif case == "negative_temperature":
            e.p_T(2e5, 0)
        elif case == "invalid_shifted_pressure":
            e.p_T(-1e5, 300)
        else:
            e.rho_u(1, 1)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "disconnected",
        "negative_resistance",
        "unknown_endpoint",
        "unknown_kind",
        "bad_volume",
        "negative_UA",
    ],
)
def test_invalid_storage_topology_and_node_parameters(case):
    r = spec()
    if case == "duplicate":
        r["nodes"][1]["id"] = "a"
    if case == "disconnected":
        r["edges"] = []
    if case == "negative_resistance":
        r["edges"][0]["linear_pa_s_kg"] = -1
    if case == "unknown_endpoint":
        r["edges"][0]["to"] = "unknown"
    if case == "unknown_kind":
        r["nodes"][0]["kind"] = "incompressible-junction"
    if case == "bad_volume":
        r["nodes"][0]["min_volume_m3"] = 1
    if case == "negative_UA":
        r["nodes"][0]["ambient_ua_w_k"] = -1
    with pytest.raises(NetworkError):
        CompressibleStorageGraph(r)


@pytest.mark.parametrize("case", ["mass", "temperature", "volume", "rhs", "steps", "commands"])
def test_late_storage_refusal_is_fully_transactional(case):
    r = spec("adiabatic_compression") if case == "volume" else spec()
    if case == "mass":
        r["nodes"][0]["initial_temperature_k"] = 600
    a = CompressibleStorageGraph(r)
    if case == "rhs":
        a.budget["maximum_rhs_evaluations"] = 0
    if case == "steps":
        a.budget["maximum_internal_steps"] = 0
    if case == "commands":
        a.budget["maximum_commands"] = 0
    if case == "mass":
        a.system.domain["min_density_kg_m3"] = min(a.system.rho0) * 0.9999
    before = a.checkpoint()
    powers = {n: (-1e6 if case == "temperature" else 0.0) for n in a.system.names}
    with pytest.raises(NetworkError):
        a.step(powers, 100 if case == "volume" else 10, expected_revision=0, command_id="retry")
    assert a.checkpoint() == before


@pytest.fixture(scope="module")
def checkpoint():
    a = CompressibleStorageGraph(spec())
    a.advance(2)
    a.step(dict(a=100, b=0), 2, expected_revision=1, command_id="heat")
    return a.checkpoint()


@pytest.mark.parametrize(
    "change", ["mass", "energy", "entropy", "history", "model", "contract", "signature", "heater"]
)
def test_resealed_checkpoint_corruption_is_atomic(checkpoint, change):
    cp = deepcopy(checkpoint)
    p = cp["payload"]
    if change == "mass":
        p["coordinates"][0] += 0.01
    if change == "energy":
        p["coordinates"][2] += 100
    if change == "entropy":
        p["coordinates"][-1] += 1
    if change == "history":
        p["snapshot"]["history"] = []
    if change == "model":
        p["model"] = "thermal-inertial-radau-v1"
    if change == "contract":
        p["snapshot"]["contract"]["eos"] = "foreign"
    if change == "signature":
        p["signature"]["python"] = "foreign"
    if change == "heater":
        p["snapshot"]["active_heat_power_w"]["a"] = 0
    a = CompressibleStorageGraph(spec())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value", [("gamma", 1.5), ("energy_offset_j_kg", 1000), ("pressure_shift_pa", 0)]
)
def test_checkpoint_binds_eos_parameters(checkpoint, field, value):
    r = spec()
    r["fluid"][field] = value
    with pytest.raises(NetworkError):
        CompressibleStorageGraph(r).restore(checkpoint)


@pytest.mark.parametrize("revision", [True, 0, 2, 1.0])
def test_stale_revision_never_changes_heaters(revision):
    a = CompressibleStorageGraph(spec())
    a.advance(1)
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=1, b=2), 1, expected_revision=revision, command_id="stale")
    assert a.checkpoint() == before


def test_inputs_outputs_and_initial_checkpoint_are_detached():
    r = spec()
    a = CompressibleStorageGraph(r)
    cp = a.checkpoint()
    r["fluid"]["gamma"] = 8
    result = a.advance(1)
    result["state"]["masses_kg"].clear()
    assert a.restore(cp) == cp["payload"]["snapshot"]


def test_final_observation_and_restore_failure_are_atomic(checkpoint, monkeypatch):
    a = CompressibleStorageGraph(spec())
    before = a.checkpoint()
    original = CompressibleStorageGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("final observation fault")
        return original(self)

    with monkeypatch.context() as p:
        p.setattr(CompressibleStorageGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.advance(2)
        with pytest.raises(NetworkError):
            a.restore(checkpoint)
    assert a.checkpoint() == before


def test_final_source_drift_is_atomic(monkeypatch):
    import scripts.compressible_storage_graph as module

    a = CompressibleStorageGraph(spec())
    before = a.checkpoint()
    original = module.signature
    calls = []

    def drift():
        value = original()
        calls.append(True)
        if len(calls) > 1:
            value["drift"] = True
        return value

    with monkeypatch.context() as p:
        p.setattr(module, "signature", drift)
        with pytest.raises(NetworkError):
            a.advance(2)
    assert a.checkpoint() == before


def test_isolated_connected_volumes_reach_analytic_mass_energy_equilibrium():
    a = CompressibleStorageGraph(spec())
    s = a.system
    expectedT = sum(s.A0) / (sum(s.m0) * s.eos.cv)
    expectedP = sum(s.p0 * s.V0) / sum(s.V0)
    out = a.advance(200)["state"]
    for T in out["temperatures_k"].values():
        assert T == pytest.approx(expectedT, abs=1e-6)
    for p in out["pressures_pa"].values():
        assert p == pytest.approx(expectedP, abs=0.001)
    assert out["generated_entropy_j_k"] > 0


def test_missing_entropy_production_is_detected_and_rolled_back(monkeypatch):
    from scripts.compressible_storage_graph import StorageSystem

    a = CompressibleStorageGraph(spec())
    before = a.checkpoint()
    original = StorageSystem.evaluate

    def broken(self, t, y, heaters):
        rate, point = original(self, t, y, heaters)
        rate[-1] = 0
        return rate, point

    with monkeypatch.context() as patch:
        patch.setattr(StorageSystem, "evaluate", broken)
        with pytest.raises(InertialGraphError) as error:
            a.advance(1)
    assert error.value.report["criterion"] == "entropy" and a.checkpoint() == before


def test_missing_pressure_shift_in_piston_work_fails_first_law(monkeypatch):
    from scripts.compressible_storage_graph import StorageSystem

    a = CompressibleStorageGraph(spec("adiabatic_expansion"))
    before = a.checkpoint()
    original = StorageSystem.evaluate

    def broken(self, t, y, heaters):
        rate, point = original(self, t, y, heaters)
        rate[self.n : 2 * self.n] += self.eos.pinf * self.vdot
        return rate, point

    with monkeypatch.context() as patch:
        patch.setattr(StorageSystem, "evaluate", broken)
        with pytest.raises(InertialGraphError) as error:
            a.advance(1)
    assert error.value.report["criterion"] == "energy" and a.checkpoint() == before
