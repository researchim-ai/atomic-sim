"""Coupled pump, kinetic inertance and gas cushions: power, Jacobian and rollback."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.hydraulics.pump_storage_pair import PumpStoragePair
from sdk.pump_storage_experiment import prepare, run_pump_storage_experiment


def example():
    return json.loads(Path("scenarios/ai05_pump_storage_trip.json").read_text())


def make(flow=0, omega=20, torque=0.2):
    s = example()
    s["timeline"] = []
    s["model"]["hydraulic_configuration"]["initial_mass_flow_kg_s"] = flow
    s["model"].update(initial_omega_rad_s=omega, motor_torque_nm=torque)
    return s, prepare(s)[1]


@pytest.mark.parametrize(
    "flow,omega", [(1, 20), (-1, -20), (1, -20), (-1, 20), (0, 0), (0, 20), (1, 0)]
)
def test_reciprocal_passive_power_and_fixed_leak_thermal_port(flow, omega):
    _, m = make(flow, omega)
    o = m.observation()
    ports = o["ports"]
    assert o["shaft_to_pump_power_w"] == pytest.approx(
        o["hydraulic_power_w"] + o["leakage_power_w"], abs=1e-10
    )
    assert o["leakage_power_w"] >= 0 and o["bearing_power_w"] >= 0
    a, b = ports["pump.leak_heat"], ports["left.pump_heat"]
    assert a["temperature_k"] == b["temperature_k"] and a["heat_flow_w"] == -b["heat_flow_w"]
    a, b = ports["shaft.to_hydraulics"], ports["hydraulics.shaft"]
    assert a["omega_rad_s"] == b["omega_rad_s"] and a["torque_nm"] == -b["torque_nm"]
    rate = m.derivatives(0, m._state)
    q = flow / m.rho
    kinetic_rate = m._hydraulic._inertia * q * rate[8]
    flux = sum(ports["connection." + end]["energy_flow_w"] for end in ("left", "right"))
    assert flux + o["shaft_to_pump_power_w"] - o["leakage_power_w"] == pytest.approx(
        kinetic_rate, abs=1e-8
    )


@pytest.mark.parametrize("flow,omega", [(1, 20), (-1, -20), (1, -20), (-1, 20), (0, 0)])
def test_analytic_jacobian_and_nonlinear_total_energy_identity(flow, omega):
    _, m = make(flow, omega)
    y = m._state.copy()
    j = m.analytic_jacobian(0, y).toarray()
    numeric = np.zeros_like(j)
    for col in range(len(y)):
        h = (1e-8 if flow == 0 and col == 8 else 1e-5) * max(1, abs(y[col]))
        a = y.copy()
        b = y.copy()
        a[col] += h
        b[col] -= h
        numeric[:, col] = (m.derivatives(0, a) - m.derivatives(0, b)) / (2 * h)
    assert np.max(abs(j - numeric)) < 1e-3
    gradient = np.r_[
        0,
        np.ones(4),
        -1,
        0,
        0,
        m._hydraulic._inertia * flow / m.rho,
        m.configuration["pump"]["inertia_kg_m2"] * omega,
        -1,
        1,
        1,
        1,
    ]
    extra = np.zeros(14)
    extra[8] = m._hydraulic._inertia / m.rho * m.derivatives(0, y)[8]
    extra[9] = m.configuration["pump"]["inertia_kg_m2"] * m.derivatives(0, y)[9]
    assert np.max(abs(gradient @ j + extra)) < 1e-7


def test_no_leak_heat_destination_switch_at_zero_flow():
    _, m = make(flow=0, omega=20)
    a = m._state.copy()
    b = a.copy()
    a[8] = 1e-10
    b[8] = -1e-10
    assert np.max(abs(m.derivatives(0, a) - m.derivatives(0, b))) < 1e-3
    assert m.observation(a)["ports"]["left.pump_heat"]["heat_flow_w"] > 0
    assert m.observation(b)["ports"]["left.pump_heat"]["heat_flow_w"] > 0


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize("sign", [1, -1])
def test_unpowered_energy_and_adiabatic_gas_invariant(method, sign):
    _, m = make(flow=sign, omega=20 * sign, torque=0)
    m.enable_continuation(10)
    o = m.advance(10, method=method, rtol=1e-11, max_step_s=0.02, atol_scale=0.01)
    s = m._hydraulic._storage
    for i, name in enumerate(m.names):
        exact = s._ug0[i] * (s._gas_volume0[i] / o["gas_volume_m3"][name]) ** (s._gamma[i] - 1)
        assert s._ug0[i] + m._state[3 + i] == pytest.approx(exact, abs=1e-5)
    assert o["motor_work_j"] == 0 and o["bearing_heat_j"] >= 0 and o["leakage_heat_j"] >= 0
    assert abs(o["energy_residual_j"]) < 1e-5 and abs(o["shaft_energy_residual_j"]) < 1e-5
    assert sum(o["liquid_mass_kg"].values()) == 20


def test_exact_equilibrium_and_pure_evaluation():
    s = example()
    s["timeline"] = []
    s["model"].update(initial_omega_rad_s=0, motor_torque_nm=0)
    for v in s["model"]["hydraulic_configuration"]["storage_configuration"]["vessels"]:
        v["initial_gas_pressure_pa"] = 100000
    m = prepare(s)[1]
    before = m.snapshot()
    assert np.array_equal(m.derivatives(0, m._state), np.zeros(14))
    m.analytic_jacobian(0, m._state)
    m.observation()
    assert m.snapshot() == before
    m.advance(100)
    assert np.array_equal(m._state, np.zeros(14)) and m.events == []


def test_atomic_controls_checkpoint_and_failed_operations(monkeypatch):
    _, m = make()
    m.advance(0.7)
    before = m.observation()
    state = m._state.copy()
    m.set_controls(dict(resistance_pa_s2_kg2=10000, motor_torque_nm=-0.2))
    assert (
        np.array_equal(m._state, state)
        and m.observation()["kinetic_energy_j"] == before["kinetic_energy_j"]
    )
    clone = PumpStoragePair.from_snapshot(json.loads(json.dumps(m.snapshot())))
    assert clone.snapshot() == m.snapshot()
    assert clone.advance(1)["mass_flow_kg_s"] == pytest.approx(
        m.advance(1)["mass_flow_kg_s"], abs=1e-9
    )
    old = m.snapshot()
    cache = m._continuation
    with pytest.raises(NetworkError):
        m.set_controls(dict(motor_torque_nm=0))
    assert m.snapshot() == old and m._continuation is cache

    def fail(*a, **kw):
        raise NetworkError("synthetic observer failure")

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.set_controls(dict(resistance_pa_s2_kg2=0, motor_torque_nm=0))
    assert m.snapshot() == old and m._continuation is cache
    with pytest.raises(NetworkError):
        m.advance(0.1)
    assert m.snapshot() == old and m._continuation is None


@pytest.mark.parametrize("change", ["hash", "state", "control", "previous", "model", "bool"])
def test_bad_checkpoint_rejected(change):
    _, m = make()
    m.advance(0.1)
    m.set_controls(dict(resistance_pa_s2_kg2=10000, motor_torque_nm=0))
    p = m.snapshot()
    if change == "hash":
        p["configuration"]["pump"]["inertia_kg_m2"] *= 2
    if change == "state":
        p["state"][9] += 1
    if change == "control":
        p["active_controls"]["motor_torque_nm"] = 1
    if change == "previous":
        p["events"][0]["previous"]["motor_torque_nm"] = 1
    if change == "model":
        p["model_version"] = "2"
    if change == "bool":
        p["state"][0] = True
    with pytest.raises(NetworkError):
        PumpStoragePair.from_snapshot(p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("displacement_m3_rad", 0),
        ("leakage_m3_s_pa", 0),
        ("inertia_kg_m2", -1),
        ("drag_nm_s_rad", -1),
        ("leak_heat_vessel", "missing"),
    ],
)
def test_invalid_pump_rejected_before_output(key, value, tmp_path):
    s = example()
    s["model"]["pump"][key] = value
    with pytest.raises(NetworkError):
        run_pump_storage_experiment(s, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_sdk_atomic_timeline_and_deadline(tmp_path):
    s = example()
    s.update(
        duration_s=2,
        step_s=0.3,
        timeline=[dict(t=0.7, controls=dict(resistance_pa_s2_kg2=10000, motor_torque_nm=0))],
    )
    r = run_pump_storage_experiment(s, tmp_path / "run")
    assert r["numerical_acceptance"] and r["accepted_commands"] == 1
    rows = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    pair = [x for x in rows if x["time_s"] == 0.7]
    assert (
        len(pair) == 2
        and pair[0]["omega_rad_s"] == pair[1]["omega_rad_s"]
        and pair[0]["mass_flow_kg_s"] == pair[1]["mass_flow_kg_s"]
    )
    assert pair[0]["liquid_energy_change_j"] == pair[1]["liquid_energy_change_j"]
    r = run_pump_storage_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0
    PumpStoragePair.from_snapshot(
        json.loads((tmp_path / "deadline" / "checkpoint.json").read_text())
    )
