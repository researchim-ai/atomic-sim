"""Hybrid seating/release, constrained Jacobians, event energy and rollback."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.hydraulics.check_pump_storage import (
    FLOW_ROOT_KG_S,
    PROJECTION_J,
    RELEASE_PA,
    CheckPumpStorage,
)
from sdk.check_storage_experiment import prepare, run_check_storage_experiment


def example():
    return json.loads(Path("scenarios/ai05_check_valve_restart.json").read_text())


def open_model():
    s = example()
    p = s["model"]["plant_configuration"]
    p.update(initial_omega_rad_s=20, motor_torque_nm=0.2)
    v = p["hydraulic_configuration"]["storage_configuration"]["vessels"]
    v[0]["initial_gas_pressure_pa"] = 200000
    v[1]["initial_gas_pressure_pa"] = 100000
    return prepare(s)[1]


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_seat_is_localized_at_zero_flow_without_finite_kinetic_loss(method):
    m = open_model()
    assert m.mode == "open"
    o = m.advance(10, method=method, rtol=1e-10, max_step_s=0.02, atol_scale=0.01)
    assert m.mode == "closed" and o["mass_flow_kg_s"] == 0
    seat = next(e for e in m.events if e["kind"] == "check_closed")
    assert (
        abs(seat["flow_residual_kg_s"]) <= FLOW_ROOT_KG_S
        and seat["kinetic_projection_j"] <= PROJECTION_J
    )
    assert abs(o["energy_residual_j"]) < 1e-5 and o["check_reaction_pa"] >= 0


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_first_release_matches_independent_closed_shaft_analytic_solution(method):
    s = example()
    m = prepare(s)[1]
    p = s["model"]["plant_configuration"]["pump"]
    D = p["displacement_m3_rad"]
    C = p["leakage_m3_s_pa"]
    B = p["drag_nm_s_rad"]
    J = p["inertia_kg_m2"]
    decay = (D * D / C + B) / J
    equilibrium = 1 / (D * D / C + B)
    threshold = (1100 + RELEASE_PA) * C / D
    exact = -np.log(1 - threshold / equilibrium) / decay
    assert m.mode == "closed"
    m.advance(2, method=method, rtol=1e-11, atol_scale=0.01)
    event = next(e for e in m.events if e["kind"] == "check_opened")
    assert event["t"] == pytest.approx(exact, abs=1e-6) and event["drive_pa"] > 0


@pytest.mark.parametrize("phase", ["open", "closed"])
def test_phase_jacobian_and_nonlinear_energy_identity(phase):
    m = open_model() if phase == "open" else prepare(example())[1]
    y = m._state.copy()
    if phase == "open":
        y[8] = 1
    else:
        y[9] = 5
    before = m.snapshot()
    j = m.analytic_jacobian(0, y, phase).toarray()
    numeric = np.zeros_like(j)
    for col in range(14):
        h = 1e-5 * max(1, abs(y[col]))
        a = y.copy()
        b = y.copy()
        a[col] += h
        b[col] -= h
        numeric[:, col] = (m.derivatives(0, a, phase) - m.derivatives(0, b, phase)) / (2 * h)
    assert np.max(abs(j - numeric)) < 1e-3
    if phase == "closed":
        assert np.array_equal(j[:, 8], np.zeros(14)) and np.array_equal(j[8], np.zeros(14))
    inertance = m._plant._hydraulic._inertia
    J = m.configuration["plant_configuration"]["pump"]["inertia_kg_m2"]
    gradient = np.r_[0, np.ones(4), -1, 0, 0, inertance * y[8] / m.rho, J * y[9], -1, 1, 1, 1]
    extra = np.zeros(14)
    rate = m.derivatives(0, y, phase)
    extra[8] = inertance * rate[8] / m.rho
    extra[9] = J * rate[9]
    assert np.max(abs(gradient @ j + extra)) < 1e-7
    assert m.snapshot() == before


def test_exact_zero_drive_equilibrium_has_no_chatter_or_false_shaft_events():
    s = example()
    s["model"]["cracking_pressure_pa"] = 0
    s["model"]["plant_configuration"]["motor_torque_nm"] = 0
    for v in s["model"]["plant_configuration"]["hydraulic_configuration"]["storage_configuration"][
        "vessels"
    ]:
        v["initial_gas_pressure_pa"] = 100000
    m = prepare(s)[1]
    assert m.mode == "closed"
    assert np.array_equal(m.derivatives(0, m._state), np.zeros(14))
    m.advance(100)
    assert np.array_equal(m._state, np.zeros(14)) and m.events == []


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_blocked_shaft_decay_and_leak_heat_have_exact_solution(method):
    s = example()
    p = s["model"]["plant_configuration"]
    p.update(initial_omega_rad_s=20, motor_torque_nm=0)
    v = p["hydraulic_configuration"]["storage_configuration"]["vessels"]
    v[0]["initial_gas_pressure_pa"] = 100000
    v[1]["initial_gas_pressure_pa"] = 200000
    m = prepare(s)[1]
    o = m.advance(10, method=method, rtol=1e-11, atol_scale=0.01)
    a = 0.3
    factor = 20**2 * (1 - np.exp(-2 * a * 10)) / (2 * a)
    assert o["omega_rad_s"] == pytest.approx(20 * np.exp(-a * 10), abs=1e-8)
    assert o["leakage_heat_j"] == pytest.approx(0.01 * factor, abs=1e-6)
    assert o["bearing_heat_j"] == pytest.approx(0.02 * factor, abs=1e-6)
    assert o["head_work_j"] == 0 and o["transfer_mass_kg"] == 0 and o["mass_flow_kg_s"] == 0


def test_close_restart_reopen_and_control_history():
    s = example()
    m = prepare(s)[1]
    m.advance(4)
    m.set_controls(s["timeline"][0]["controls"])
    m.advance(6)
    assert m.mode == "closed"
    clone = CheckPumpStorage.from_snapshot(json.loads(json.dumps(m.snapshot())))
    assert clone.snapshot() == m.snapshot()
    for model in (m, clone):
        model.set_controls(s["timeline"][1]["controls"])
    a, b = m.advance(10), clone.advance(10)
    assert a["mass_flow_kg_s"] == pytest.approx(b["mass_flow_kg_s"], abs=1e-8)
    assert [e["kind"] for e in m.events].count("check_opened") >= 2


def test_observer_failure_after_seat_rolls_back_whole_interval(monkeypatch):
    m = open_model()
    before = m.snapshot()
    original = m.observation

    def fail(y=None, time_s=None, mode=None):
        if mode == "closed" and time_s > 0:
            raise NetworkError("synthetic failure after seating")
        return original(y, time_s, mode)

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.advance(10)
    assert m.snapshot() == before and m.mode == "open"


@pytest.mark.parametrize("change", ["mode", "config", "seat_energy", "history", "state", "bool"])
def test_checkpoint_tampering_rejected(change):
    m = open_model()
    m.advance(10)
    m.set_controls(dict(resistance_pa_s2_kg2=10000, motor_torque_nm=0))
    p = m.snapshot()
    if change == "mode":
        p["mode"] = "open"
    if change == "config":
        p["configuration"]["cracking_pressure_pa"] += 1
    if change == "seat_energy":
        next(e for e in p["events"] if e["kind"] == "check_closed")["kinetic_projection_j"] = 1
    if change == "history":
        p["active_controls"]["motor_torque_nm"] = 1
    if change == "state":
        p["state"][8] = 1e-12
    if change == "bool":
        p["schema_version"] = True
    with pytest.raises(NetworkError):
        CheckPumpStorage.from_snapshot(p)


def test_negative_initial_flow_and_invalid_threshold_rejected_before_output(tmp_path):
    for change in ("flow", "threshold"):
        s = example()
        if change == "flow":
            s["model"]["plant_configuration"]["hydraulic_configuration"][
                "initial_mass_flow_kg_s"
            ] = -0.1
        else:
            s["model"]["cracking_pressure_pa"] = -1
        with pytest.raises(NetworkError):
            run_check_storage_experiment(s, tmp_path / change)
        assert not (tmp_path / change).exists()


def test_output_grid_does_not_move_valve_events():
    a, b = prepare(example())[1], prepare(example())[1]
    a.advance(4, rtol=1e-10, atol_scale=0.01)
    for _ in range(10):
        b.advance(0.4, rtol=1e-10, atol_scale=0.01)
    assert [e["kind"] for e in a.events] == [e["kind"] for e in b.events]
    assert [e["t"] for e in a.events] == pytest.approx([e["t"] for e in b.events], abs=1e-6)


def test_sdk_prefix_atomic_controls_and_deadline(tmp_path):
    s = example()
    s.update(
        duration_s=2,
        step_s=0.3,
        timeline=[dict(t=0.7, controls=dict(resistance_pa_s2_kg2=10000, motor_torque_nm=1))],
    )
    r = run_check_storage_experiment(s, tmp_path / "run")
    assert r["numerical_acceptance"] and r["accepted_commands"] == 1
    p = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert CheckPumpStorage.from_snapshot(p).snapshot() == p
    rows = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    assert min(x["mass_flow_kg_s"] for x in rows) >= 0
    r = run_check_storage_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0
    CheckPumpStorage.from_snapshot(
        json.loads((tmp_path / "deadline" / "checkpoint.json").read_text())
    )


@pytest.mark.parametrize("case", ["release", "seat"])
def test_event_at_initial_time_advances_phase_then_finishes_interval(case):
    s = example()
    plant = s["model"]["plant_configuration"]
    if case == "release":
        s["model"]["cracking_pressure_pa"] = 0
        plant.update(initial_omega_rad_s=1e-8, motor_torque_nm=1)
        for vessel in plant["hydraulic_configuration"]["storage_configuration"]["vessels"]:
            vessel["initial_gas_pressure_pa"] = 100000
    else:
        plant["hydraulic_configuration"]["initial_mass_flow_kg_s"] = 1e-15
        plant["motor_torque_nm"] = 0
    model = prepare(s)[1]
    initial_energy = model.observation()["energy_residual_j"]
    result = model.advance(0.01)
    event = next(e for e in model.events if e["kind"] in ("check_opened", "check_closed"))
    assert event["t"] == pytest.approx(0, abs=1e-12)
    assert model.time_s == 0.01 and result["mass_flow_kg_s"] >= 0
    assert abs(result["energy_residual_j"] - initial_energy) < 1e-5
    assert event["kind"] == ("check_opened" if case == "release" else "check_closed")
