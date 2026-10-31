"""Storage mass/energy/boundary work, check threshold, rollback and restart."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from core.network import NetworkError
from physics.hydraulics.gas_storage_pair import GasStoragePair
from scripts.storage_references import adiabatic_at
from sdk.storage_experiment import prepare, run_storage_experiment


def example():
    return json.loads(Path("scenarios/ai05_storage_isolation.json").read_text())


def model_for(reverse=False, check=False, thermal=False):
    s = example()
    s["timeline"] = []
    if reverse:
        (
            s["model"]["vessels"][0]["initial_gas_pressure_pa"],
            s["model"]["vessels"][1]["initial_gas_pressure_pa"],
        ) = 100000, 200000
    if check:
        s["model"]["connection"].update(kind="check", cracking_pressure_pa=100)
    if thermal:
        for v in s["model"]["vessels"]:
            v.update(gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2, liquid_heater_w=20)
    return s, prepare(s)[1]


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize(
    "reverse,check", [(False, False), (True, False), (False, True), (True, True)]
)
def test_independent_adiabatic_quadrature_and_boundary_work(method, reverse, check):
    spec, m = model_for(reverse, check)
    for t in (0.1, 1, 3):
        o = m.advance(t - m.time_s, method=method, rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
        ref = adiabatic_at(spec, t)
        assert o["transfer_mass_kg"] == pytest.approx(ref["transfer_mass_kg"], abs=1e-9)
        assert list(o["pressure_pa"].values()) == pytest.approx(ref["pressure"], abs=1e-4)
        assert list(o["gas_temperature_k"].values()) == pytest.approx(
            ref["gas_temperature"], abs=1e-8
        )
        assert list(o["liquid_temperature_k"].values()) == pytest.approx(
            ref["liquid_temperature"], abs=1e-8
        )
        assert list(o["gas_boundary_work_j"].values()) == pytest.approx(
            ref["gas_boundary_work"], abs=1e-6
        )
        assert sum(o["liquid_mass_kg"].values()) == 20
        assert abs(o["energy_residual_j"]) < 1e-6 and o["dissipation_w"] >= 0


@pytest.mark.parametrize(
    "reverse,check", [(False, False), (True, False), (False, True), (True, True)]
)
def test_pure_analytic_jacobian_and_linear_energy_identity(reverse, check):
    _, m = model_for(reverse, check, True)
    y = m._state.copy()
    y[0] = 0.1
    y[1:5] = [20, -10, 10, -5]
    y[5] = sum(y[1:5])
    saved = m.snapshot()
    j = m.analytic_jacobian(0, y).toarray()
    numeric = np.zeros_like(j)
    for col in range(8):
        h = 1e-5 * max(1, abs(y[col]))
        a = y.copy()
        b = y.copy()
        a[col] += h
        b[col] -= h
        numeric[:, col] = (m.derivatives(0, a) - m.derivatives(0, b)) / (2 * h)
    assert np.max(abs(j - numeric)) < 1e-3
    assert np.max(abs(np.r_[0, np.ones(4), -1, 0, 0] @ j)) < 1e-8
    m.observation(y)
    assert m.snapshot() == saved


def test_pressure_equal_steady_and_exact_isolation():
    s = example()
    s["timeline"] = []
    for v in s["model"]["vessels"]:
        v["initial_gas_pressure_pa"] = 100000
    m = prepare(s)[1]
    before = m.snapshot()
    o = m.advance(100)
    assert np.array_equal(m._state, np.zeros(8)) and o["mass_flow_kg_s"] == 0
    assert before["configuration"] == m.snapshot()["configuration"]
    _, m = model_for()
    m.advance(0.5)
    state = m._state.copy()
    energy = m.observation()["energy_residual_j"]
    m.set_conductance(0)
    assert m.observation()["mass_flow_kg_s"] == 0
    m.advance(5)
    assert np.array_equal(m._state, state) and m.observation()["energy_residual_j"] == energy
    m.set_conductance(1e-8)
    assert m.advance(0.1)["transfer_mass_kg"] > state[0]


@pytest.mark.parametrize("rtol", [1e-10, 1e-11])
def test_heating_opens_check_valve_at_independent_analytic_time(rtol):
    s = example()
    s["timeline"] = []
    s["model"]["connection"].update(kind="check", cracking_pressure_pa=0, conductance_m3_s_pa=1e-8)
    for i, v in enumerate(s["model"]["vessels"]):
        v.update(
            initial_gas_pressure_pa=100000 + 5 * i,
            gas_liquid_ua_w_k=100 if i == 0 else 0,
            liquid_heater_w=100 if i == 0 else 0,
        )
    m = prepare(s)[1]
    cl = 10 * 4200
    cg = 100000 * 0.01 / (0.4 * 300)
    tau = cl * cg / (100 * (cl + cg))
    exact = brentq(
        lambda t: 100000 * (100 / (cl + cg) * (t - tau * (1 - np.exp(-t / tau)))) / 300 - 5, 0, 20
    )
    assert m.observation()["connection_mode"] == "blocked"
    m.enable_continuation(20)
    for _ in range(40):
        m.advance(0.5, rtol=rtol, max_step_s=0.025 if rtol == 1e-11 else 0.05, atol_scale=0.01)
    assert (
        len(m.events) == 1
        and m.events[0]["kind"] == "check_threshold"
        and m.events[0]["direction"] == 1
    )
    assert m.events[0]["t"] == pytest.approx(exact, abs=1e-5)
    assert m.observation()["mass_flow_kg_s"] > 0


def test_pressure_ports_and_single_enthalpy_flux():
    _, m = model_for()
    o = m.observation()
    p = o["ports"]
    for name, end in zip(m.names, ("left", "right")):
        a, b = p[name + ".transfer"], p["connection." + end]
        assert (
            a["pressure_pa"] == b["pressure_pa"]
            and a["mass_flow_kg_s"] == -b["mass_flow_kg_s"]
            and a["energy_flow_w"] == -b["energy_flow_w"]
        )
    assert sum(p["connection." + end]["energy_flow_w"] for end in ("left", "right")) == 0
    assert sum(o["gas_boundary_power_w"].values()) == -o["dissipation_w"]


def test_checkpoint_restart_command_and_rollback(monkeypatch):
    _, m = model_for()
    m.advance(0.7)
    m.set_conductance(0)
    m.advance(0.3)
    m.set_conductance(1e-8)
    payload = m.snapshot()
    clone = GasStoragePair.from_snapshot(json.loads(json.dumps(payload)))
    assert clone.snapshot() == payload
    assert clone.advance(1)["transfer_mass_kg"] == pytest.approx(
        m.advance(1)["transfer_mass_kg"], abs=1e-9
    )
    before = m.snapshot()
    cache = m._continuation

    def fail(*a, **k):
        raise NetworkError("synthetic failure")

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.set_conductance(0)
    assert m.snapshot() == before and m._continuation is cache
    with pytest.raises(NetworkError):
        m.advance(0.1)
    assert m.snapshot() == before and m._continuation is None


@pytest.mark.parametrize("change", ["hash", "energy", "event", "mode", "bool", "shape"])
def test_bad_checkpoints_rejected(change):
    _, m = model_for()
    m.advance(0.1)
    m.set_conductance(0)
    p = m.snapshot()
    if change == "hash":
        p["configuration"]["vessels"][0]["gas_gamma"] = 1.5
    if change == "energy":
        p["state"][1] += 1
    if change == "event":
        p["events"][0]["previous"] = 0
    if change == "mode":
        p["model_version"] = "2"
    if change == "bool":
        p["state"][0] = True
    if change == "shape":
        p["state"].append(0)
    with pytest.raises(NetworkError):
        GasStoragePair.from_snapshot(p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("initial_liquid_mass_kg", 0),
        ("total_volume_m3", 0.005),
        ("gas_gamma", 1),
        ("gas_liquid_ua_w_k", -1),
        ("initial_gas_pressure_pa", 0),
    ],
)
def test_bad_initial_model_rejected_before_output(key, value, tmp_path):
    s = example()
    s["model"]["vessels"][0][key] = value
    with pytest.raises(NetworkError):
        run_storage_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_invalid_trial_state_and_parameter_copies():
    _, m = model_for()
    p = m.configuration
    p["vessels"][0]["gas_gamma"] = 1.8
    assert m.configuration["vessels"][0]["gas_gamma"] == 1.4
    for x in (11, -11):
        y = m._state.copy()
        y[0] = x
        with pytest.raises(NetworkError):
            m.derivatives(0, y)
    assert m.snapshot()["state"] == [0.0] * 8


def test_sdk_prefix_and_deadline(tmp_path):
    s = example()
    s.update(
        duration_s=3,
        step_s=0.3,
        timeline=[dict(t=0.7, conductance_m3_s_pa=0), dict(t=1.1, conductance_m3_s_pa=1e-8)],
    )
    report = run_storage_experiment(s, tmp_path / "run")
    assert report["numerical_acceptance"] and report["accepted_commands"] == 2
    rows = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    pair = [r for r in rows if r["time_s"] == 0.7]
    assert len(pair) == 2 and pair[0]["liquid_mass_kg"] == pair[1]["liquid_mass_kg"]
    assert (
        pair[1]["mass_flow_kg_s"] == 0
        and pair[0]["gas_energy_change_j"] == pair[1]["gas_energy_change_j"]
    )
    r = run_storage_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0
    GasStoragePair.from_snapshot(
        json.loads((tmp_path / "deadline" / "checkpoint.json").read_text())
    )
