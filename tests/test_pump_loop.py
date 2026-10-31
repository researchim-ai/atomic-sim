"""Pump/shaft reciprocal power, signed modes, solver rollback and restart."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.network import NetworkError
from physics.hydraulics.pump_loop import PumpCirculationLoop, steady_pump_loop
from sdk.pump_experiment import prepare, run_pump_experiment


def example():
    return json.loads(Path("scenarios/ai05_displacement_pump_trip.json").read_text())


def model_for(flow=0.5, omega=20.0, torque=0.2):
    s = example()
    s["loop"]["initialization"]["mass_flow_kg_s"] = flow
    s["initial_omega_rad_s"], s["motor_torque_nm"] = omega, torque
    return prepare(s)[1]


@pytest.mark.parametrize(
    "flow,omega", [(1, 20), (-1, -20), (1, -20), (-1, 20), (0, 0), (0, 20), (1, 0)]
)
def test_signed_power_transfer_is_passive_and_division_free(flow, omega):
    m = model_for(flow, omega)
    o = m.observation()
    assert o["shaft_to_pump_power_w"] == pytest.approx(
        o["hydraulic_power_w"] + o["leakage_power_w"], abs=1e-10
    )
    assert o["leakage_power_w"] >= 0 and o["bearing_power_w"] >= 0
    rate = m.derivatives(0, m._state)
    loop = m._loop
    k = m._size
    total_rate = (
        loop.capacity @ rate[1 : loop.n + 1]
        + loop.wall_capacity @ rate[loop.n + 1 : 2 * loop.n + 1]
        + loop.inertia_total * flow * rate[0] / loop.rho
        + m.pump["inertia_kg_m2"] * omega * rate[k]
    )
    assert total_rate == pytest.approx(o["motor_power_w"] - o["bearing_power_w"], abs=1e-10)


@pytest.mark.parametrize("flow,omega", [(1, 20), (-1, -20), (1, -20), (-1, 20), (0, 0)])
def test_analytic_jacobian_matches_independent_finite_difference_and_energy_identity(flow, omega):
    m = model_for(flow, omega)
    y = m._state.copy()
    k = m._size
    loop = m._loop
    j = m.analytic_jacobian(0, y).toarray()
    numerical = np.zeros_like(j)
    for c in range(len(y)):
        h = (1e-8 if flow == 0 and c == 0 else 1e-5) * max(1, abs(y[c]))
        a = y.copy()
        b = y.copy()
        a[c] += h
        b[c] -= h
        # At zero flow uniform temperatures make both directional derivatives coincide.
        numerical[:, c] = (m.derivatives(0, a) - m.derivatives(0, b)) / (2 * h)
    assert np.max(abs(j - numerical)) < 1e-7
    gradient = np.r_[
        loop.inertia_total * flow / loop.rho,
        loop.capacity,
        loop.wall_capacity,
        0,
        -1,
        m.pump["inertia_kg_m2"] * omega,
        -1,
        1,
        1,
    ]
    # Efluid+Ewall+Kflow+Kshaft-Wmotor+Qbearing-Qloop+Qleak is invariant.
    hessian = np.zeros_like(j)
    hessian[0, 0] = loop.inertia_total / loop.rho
    hessian[k, k] = m.pump["inertia_kg_m2"]
    assert np.max(abs(gradient @ j + m.derivatives(0, y) @ hessian)) < 1e-8


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize("sign", [1, -1])
def test_independent_mechanical_reference_and_heat_inventories(method, sign):
    m = model_for(0.5 * sign, 20.0 * sign, 0.2 * sign)
    p = m.pump
    loop = m._loop

    def reference(t, z):
        flow, omega = z[:2]
        q = flow / loop.rho
        head = (p["displacement_m3_rad"] * omega - q) / p["leakage_m3_s_pa"]
        return [
            (head - loop.resistance_total * flow * abs(flow)) / loop.inertia_total,
            (0.2 * sign - p["displacement_m3_rad"] * head - p["drag_nm_s_rad"] * omega)
            / p["inertia_kg_m2"],
            0.2 * sign * omega,
            p["drag_nm_s_rad"] * omega**2,
            p["leakage_m3_s_pa"] * head**2,
            loop.resistance_total * abs(flow) ** 3 / loop.rho,
        ]

    ref = solve_ivp(
        reference,
        (0, 10),
        [0.5 * sign, 20.0 * sign, 0, 0, 0, 0],
        method="DOP853",
        rtol=2e-13,
        atol=1e-13,
        dense_output=True,
        max_step=0.02,
    )
    assert ref.success
    for t in (1, 3, 10):
        o = m.advance(t - m.time_s, method=method, rtol=1e-10, atol_scale=0.01)
        exact = ref.sol(t)
        assert o["loop"]["mass_flow_kg_s"] == pytest.approx(exact[0], abs=1e-8)
        assert o["omega_rad_s"] == pytest.approx(exact[1], abs=1e-8)
        assert [o["motor_work_j"], o["bearing_heat_j"], o["leakage_heat_j"]] == pytest.approx(
            exact[2:5], abs=1e-7
        )
        assert loop.capacity @ m._state[1 : loop.n + 1] == pytest.approx(exact[5], abs=1e-7)
        assert loop.wall_capacity @ m._state[loop.n + 1 : 2 * loop.n + 1] == pytest.approx(
            exact[4], abs=1e-7
        )
        assert abs(o["total_energy_residual_j"]) < 1e-6


@pytest.mark.parametrize("torque", [0.2, -0.2, 0])
def test_steady_initializer_solves_actual_dimensional_residual(torque):
    s = example()
    for n in s["loop"]["model"]["topology"]["nodes"]:
        n["parameters"].update(fluid_wall_ua_w_k=100, ambient_ua_w_k=20)
    m = steady_pump_loop(s["loop"]["model"], s["pump"], torque)
    rate = m.derivatives(0, m._state)
    assert abs(rate[0] * m._loop.inertia_total) < 1e-7
    assert abs(rate[m._size] * m.pump["inertia_kg_m2"]) < 1e-9
    assert (
        max(abs(np.r_[m._loop.capacity, m._loop.wall_capacity] * rate[1 : 2 * m._loop.n + 1]))
        < 1e-6
    )
    initial = m.observation()
    final = m.advance(10)
    assert final["omega_rad_s"] == pytest.approx(initial["omega_rad_s"], abs=1e-9)
    assert final["loop"]["fluid_temperature_k"] == pytest.approx(
        initial["loop"]["fluid_temperature_k"], abs=1e-9
    )


def test_no_sink_rejects_thermal_steady():
    s = example()
    with pytest.raises(NetworkError, match="anchor"):
        steady_pump_loop(s["loop"]["model"], s["pump"], 0.2)


def test_torque_switch_rollback_and_checkpoint_roundtrip(monkeypatch):
    m = model_for()
    m.advance(1)
    old = m.snapshot()
    energy = m.observation()["loop"]["stored_energy_j"]
    m.set_motor_torque(-0.2)
    assert m.observation()["loop"]["stored_energy_j"] == energy
    m.advance(15)
    assert {e["kind"] for e in m.events} == {"motor_torque_changed", "flow_zero", "shaft_zero"}
    clone = PumpCirculationLoop.from_snapshot(json.loads(json.dumps(m.snapshot())))
    assert clone.snapshot() == m.snapshot()
    assert clone.advance(1)["omega_rad_s"] == pytest.approx(m.advance(1)["omega_rad_s"], abs=1e-8)
    before = m.snapshot()

    def fail(*args, **kwargs):
        raise NetworkError("synthetic failed observation")

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.advance(0.1)
    assert m.snapshot() == before and m._continuation is None
    with pytest.raises(NetworkError):
        m.set_motor_torque(0.2)
    assert m.snapshot() == before
    assert old["active_motor_torque_nm"] == 0.2


@pytest.mark.parametrize(
    "key,value",
    [
        ("leakage_m3_s_pa", 0),
        ("displacement_m3_rad", -1),
        ("inertia_kg_m2", 0),
        ("drag_nm_s_rad", -1),
        ("segment", "missing"),
    ],
)
def test_invalid_pump_rejected_before_output(key, value, tmp_path):
    s = example()
    s["pump"][key] = value
    with pytest.raises(NetworkError):
        run_pump_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("corruption", ["hash", "state", "torque", "direction", "model"])
def test_checkpoint_tampering_rejected(corruption):
    m = model_for()
    m.set_motor_torque(-0.2)
    m.advance(15)
    p = m.snapshot()
    if corruption == "hash":
        p["pump"]["inertia_kg_m2"] *= 2
    if corruption == "state":
        p["state"][-3] += 1
    if corruption == "torque":
        p["active_motor_torque_nm"] += 1
    if corruption == "direction":
        next(e for e in p["events"] if e["kind"] == "flow_zero")["direction"] = True
    if corruption == "model":
        p["model_version"] = "2.0"
    with pytest.raises(NetworkError):
        PumpCirculationLoop.from_snapshot(p)


def test_runtime_parameters_detached_and_noninitial_loop_rejected():
    m = model_for()
    p = m.pump
    p["inertia_kg_m2"] = 99
    assert m.pump["inertia_kg_m2"] == 0.1
    loop = m._loop
    loop.advance(0.1)
    with pytest.raises(NetworkError, match="initial loop"):
        PumpCirculationLoop(loop.snapshot(), m.pump, 20, 0.2)


def test_sdk_torque_timeline_and_deadline_preserve_prefix(tmp_path):
    s = example()
    s["duration_s"] = 2
    s["step_s"] = 0.3
    s["timeline"] = [dict(t=0.7, motor_torque_nm=0)]
    r = run_pump_experiment(s, tmp_path / "normal")
    assert r["numerical_acceptance"] and r["accepted_commands"] == 1
    rows = json.loads((tmp_path / "normal" / "trajectory.json").read_text())
    pair = [r for r in rows if r["time_s"] == 0.7]
    assert (
        len(pair) == 2 and pair[0]["loop"]["stored_energy_j"] == pair[1]["loop"]["stored_energy_j"]
    )
    r = run_pump_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0
    PumpCirculationLoop.from_snapshot(
        json.loads((tmp_path / "deadline" / "checkpoint.json").read_text())
    )


def test_ideal_heads_cannot_be_double_counted():
    s = example()
    s["loop"]["model"]["topology"]["nodes"][0]["parameters"]["drive_pressure_pa"] = 100
    with pytest.raises(NetworkError, match="ideal heads"):
        prepare(s)


def test_failed_huge_torque_restores_control_and_history():
    m = model_for(omega=20)
    m.advance(0.1)
    before = m.snapshot()
    cache = m._continuation
    with pytest.raises(NetworkError):
        m.set_motor_torque(1e308)
    assert m.snapshot() == before and m._continuation is cache
