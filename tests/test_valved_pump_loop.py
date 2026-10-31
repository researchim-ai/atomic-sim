"""Finite-leakage valve: exact actuator, passive heat, pure evaluation and rollback."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.hydraulics.valved_pump_loop import ValvedPumpLoop
from sdk.pump_experiment import prepare, run_pump_experiment


def example():
    return json.loads(Path("scenarios/ai05_valve_close_reopen.json").read_text())


def model_for(sign=1):
    s = example()
    s["initial_omega_rad_s"] *= sign
    s["motor_torque_nm"] *= sign
    s["loop"]["initialization"]["mass_flow_kg_s"] *= sign
    return prepare(s)[1]


@pytest.mark.parametrize("target", [0, 0.3, 1])
def test_exact_actuator_and_continuous_resistance_after_commands(target):
    m = model_for()
    initial = m.observation()
    m.set_valve_opening(target)
    assert m.observation()["valve_opening"] == initial["valve_opening"]
    assert m.observation()["total_energy_residual_j"] == initial["total_energy_residual_j"]
    for t in (0.1, 1, 3):
        o = m.advance(t - m.time_s)
        assert o["valve_opening"] == pytest.approx(
            target + (1 - target) * np.exp(-t / 0.5), abs=1e-15
        )
    a = m.opening()
    m.set_valve_opening(0.8)
    m.advance(0.2)
    assert m.opening() == pytest.approx(0.8 + (a - 0.8) * np.exp(-0.2 / 0.5), abs=1e-15)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("opening", [0, 0.4, 1])
def test_signed_passive_valve_heat_and_pressure(sign, opening):
    m = model_for(sign)
    m._epoch_opening = m._target = opening
    o = m.observation()
    flow = o["loop"]["mass_flow_kg_s"]
    assert o["valve_dissipation_w"] >= 0
    assert o["valve_pressure_loss_pa"] * flow / m._loop.rho == pytest.approx(
        o["valve_dissipation_w"]
    )
    rate = m.derivatives(0, m._state)
    base = m._loop.derivatives(0, m._state[: m._size])
    # Extra valve friction heats the fluid inventory directly, not the wall or exterior.
    assert m._loop.capacity @ (rate[1 : m._loop.n + 1] - base[1 : m._loop.n + 1]) == pytest.approx(
        o["valve_dissipation_w"]
    )
    initial = m.snapshot()
    m.analytic_jacobian(0, m._state)
    m.observation(m._state, 2)
    assert m.snapshot() == initial
    assert all(n["parameters"]["loss_coefficient"] == 10 for n in m._loop.topology["nodes"])


@pytest.mark.parametrize("sign", [1, -1])
def test_non_autonomous_analytic_jacobian(sign):
    m = model_for(sign)
    m.set_valve_opening(0)
    y = m._state.copy()
    y[1:4] = [1, 2, 3]
    for t in (0.1, 1, 5):
        analytic = m.analytic_jacobian(t, y).toarray()
        numerical = np.zeros_like(analytic)
        for col in range(len(y)):
            h = 1e-5 * max(1, abs(y[col]))
            a = y.copy()
            b = y.copy()
            a[col] += h
            b[col] -= h
            numerical[:, col] = (m.derivatives(t, a) - m.derivatives(t, b)) / (2 * h)
        assert np.max(abs(analytic - numerical)) < 1e-7


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_restart_midstroke_and_reverse_flow(method):
    m = model_for()
    m.set_valve_opening(0)
    m.advance(0.3, method=method)
    clone = ValvedPumpLoop.from_snapshot(json.loads(json.dumps(m.snapshot())))
    assert clone.snapshot() == m.snapshot()
    a = m.advance(2, method=method)
    b = clone.advance(2, method=method)
    assert a["loop"]["mass_flow_kg_s"] == pytest.approx(b["loop"]["mass_flow_kg_s"], abs=1e-8)
    m.set_motor_torque(-0.2)
    m.advance(20, method=method)
    assert m.observation()["loop"]["mass_flow_kg_s"] < 0
    assert {e["kind"] for e in m.events} >= {"flow_zero", "shaft_zero", "valve_target_changed"}
    assert abs(m.observation()["total_energy_residual_j"]) < 1e-5


@pytest.mark.parametrize(
    "key,value",
    [
        ("segment", "missing"),
        ("initial_opening", 1.1),
        ("open_loss_coefficient", -1),
        ("closed_loss_coefficient", 0),
        ("actuator_time_s", 0),
    ],
)
def test_invalid_configuration_before_output(key, value, tmp_path):
    s = example()
    s["valve"][key] = value
    with pytest.raises(NetworkError):
        run_pump_experiment(s, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_failed_valve_command_and_step_are_atomic(monkeypatch):
    m = model_for()
    m.advance(0.1)
    before = m.snapshot()
    cache = m._continuation
    with pytest.raises(NetworkError):
        m.set_valve_opening(1.1)
    assert m.snapshot() == before and m._continuation is cache

    def fail(*args, **kwargs):
        raise NetworkError("synthetic observation failure")

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.set_valve_opening(0)
    assert m.snapshot() == before and m._continuation is cache
    with pytest.raises(NetworkError):
        m.advance(0.1)
    assert m.snapshot() == before and m._continuation is None


@pytest.mark.parametrize("change", ["hash", "previous", "opening", "target", "extra", "state"])
def test_checkpoint_tampering_rejected(change):
    m = model_for()
    m.set_valve_opening(0)
    m.advance(0.3)
    m.set_motor_torque(0)
    p = m.snapshot()
    if change == "hash":
        p["valve"]["actuator_time_s"] *= 2
    if change == "previous":
        p["events"][0]["previous_target"] = 0.5
    if change == "opening":
        p["events"][0]["opening"] = 0.5
    if change == "target":
        p["events"][0]["target"] = 2
    if change == "extra":
        p["extra"] = 0
    if change == "state":
        p["state"][-3] += 1
    with pytest.raises(NetworkError):
        ValvedPumpLoop.from_snapshot(p)


def test_sdk_mixed_controls_prefix_and_deadline(tmp_path):
    s = example()
    s.update(
        duration_s=2,
        step_s=0.3,
        timeline=[dict(t=0.7, valve_opening=0), dict(t=1.1, motor_torque_nm=0)],
    )
    report = run_pump_experiment(s, tmp_path / "run")
    assert report["numerical_acceptance"] and report["accepted_commands"] == 2
    rows = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    pair = [r for r in rows if r["time_s"] == 0.7]
    assert len(pair) == 2 and pair[0]["valve_opening"] == pair[1]["valve_opening"]
    assert pair[0]["loop"]["stored_energy_j"] == pair[1]["loop"]["stored_energy_j"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["passport"]["model_id"] == s["engine"]
    checkpoint = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert ValvedPumpLoop.from_snapshot(checkpoint).snapshot() == checkpoint
    r = run_pump_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0


def test_valve_parameters_detached():
    m = model_for()
    v = m.valve
    v["initial_opening"] = 0
    assert m.valve["initial_opening"] == 1
    payload = deepcopy(m.snapshot())
    payload["schema_version"] = True
    with pytest.raises(NetworkError):
        ValvedPumpLoop.from_snapshot(payload)
