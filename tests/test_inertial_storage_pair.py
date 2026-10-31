"""Inertial coupling: kinetic inventory, nonlinear period, ports and rollback."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.hydraulics.inertial_storage_pair import InertialStoragePair
from scripts.inertial_storage_references import lossless_half_period
from sdk.storage_experiment import prepare, run_storage_experiment


def example():
    return json.loads(Path("scenarios/ai05_inertial_storage_damping.json").read_text())


def make(reverse=False, resistance=1000, flow=0):
    s = example()
    s["timeline"] = []
    s["model"]["line"]["resistance_pa_s2_kg2"] = resistance
    s["model"]["initial_mass_flow_kg_s"] = flow
    if reverse:
        v = s["model"]["storage_configuration"]["vessels"]
        v[0]["initial_gas_pressure_pa"], v[1]["initial_gas_pressure_pa"] = 100000, 200000
    return s, prepare(s)[1]


@pytest.mark.parametrize("flow", [-1.0, 1.0])
def test_analytic_jacobian_and_nonlinear_energy_identity(flow):
    _, m = make(flow=flow)
    y = m._state.copy()
    y[0] = 0.1
    y[1:5] = [10, -20, 20, -10]
    j = m.analytic_jacobian(0, y).toarray()
    fd = np.zeros_like(j)
    for col in range(9):
        h = 1e-5 * max(1, abs(y[col]))
        a = y.copy()
        b = y.copy()
        a[col] += h
        b[col] -= h
        fd[:, col] = (m.derivatives(0, a) - m.derivatives(0, b)) / (2 * h)
    assert np.max(abs(j - fd)) < 1e-3
    gradient = np.r_[0, np.ones(4), -1, 0, 0, m._inertia * flow / m.rho]
    correction = np.zeros(9)
    correction[8] = m._inertia / m.rho * m.derivatives(0, y)[8]
    assert np.max(abs(gradient @ j + correction)) < 1e-8


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("resistance", [0, 1000])
def test_adiabatic_gas_invariant_and_total_kinetic_energy(method, reverse, resistance):
    _, m = make(reverse, resistance)
    s = m._storage
    initial = m.snapshot()
    m.enable_continuation(10)
    for target in (1, 3, 10):
        o = m.advance(
            target - m.time_s, method=method, rtol=1e-11, max_step_s=0.02, atol_scale=0.01
        )
        for i, name in enumerate(m.names):
            expected = s._ug0[i] * (s._gas_volume0[i] / o["gas_volume_m3"][name]) ** (
                s._gamma[i] - 1
            )
            assert s._ug0[i] + m._state[3 + i] == pytest.approx(expected, abs=1e-6)
        assert sum(o["liquid_mass_kg"].values()) == 20
        assert abs(o["energy_residual_j"]) < 1e-5
        if resistance == 0:
            assert sum(o["gas_energy_change_j"].values()) + o[
                "kinetic_energy_change_j"
            ] == pytest.approx(0, abs=1e-6)
    assert initial["configuration"] == m.configuration


@pytest.mark.parametrize("reverse", [False, True])
def test_nonlinear_lossless_period_from_energy_quadrature(reverse):
    spec, m = make(reverse, 0)
    half = lossless_half_period(spec)
    m.enable_continuation(20)
    m.advance(20, rtol=1e-11, max_step_s=0.02, atol_scale=0.01)
    assert len(m.events) >= 3
    for index, e in enumerate(m.events):
        assert e["t"] == pytest.approx((index + 1) * half, abs=1e-6)


def test_small_signal_limit_matches_linear_gas_spring_frequency():
    s, m = make(resistance=0)
    vessels = s["model"]["storage_configuration"]["vessels"]
    for v, p in zip(vessels, (100000.5, 99999.5)):
        v["initial_gas_pressure_pa"] = p
    m = prepare(s)[1]
    m.enable_continuation(10)
    m.advance(10, rtol=1e-11, max_step_s=0.02, atol_scale=0.001)
    omega = np.sqrt(
        sum(
            m._storage._gamma
            * np.array([v["initial_gas_pressure_pa"] for v in vessels])
            / m._storage._gas_volume0
        )
        / (m.rho * m._inertia)
    )
    assert m.events[0]["t"] == pytest.approx(np.pi / omega, abs=1e-6)


@pytest.mark.parametrize("flow", [-1, 1])
def test_port_flux_difference_charges_kinetic_inventory(flow):
    _, m = make(flow=flow)
    o = m.observation()
    ports = o["ports"]
    for name, end in zip(m.names, ("left", "right")):
        a, b = ports[name + ".transfer"], ports["connection." + end]
        assert (
            a["pressure_pa"] == b["pressure_pa"]
            and a["mass_flow_kg_s"] == -b["mass_flow_kg_s"]
            and a["energy_flow_w"] == -b["energy_flow_w"]
        )
    kin_rate = m._inertia * flow * o["flow_acceleration_kg_s2"] / m.rho
    assert sum(
        ports["connection." + end]["energy_flow_w"] for end in ("left", "right")
    ) == pytest.approx(kin_rate, abs=1e-8)
    assert sum(o["gas_boundary_power_w"].values()) + o["dissipation_w"] + kin_rate == pytest.approx(
        0, abs=1e-8
    )


def test_resistance_switch_is_energy_and_flow_continuous():
    _, m = make()
    m.advance(0.7)
    before = m.observation()
    state = m._state.copy()
    m.set_resistance(10000)
    assert (
        np.array_equal(m._state, state)
        and m.observation()["mass_flow_kg_s"] == before["mass_flow_kg_s"]
    )
    assert m.observation()["kinetic_energy_j"] == before["kinetic_energy_j"]
    clone = InertialStoragePair.from_snapshot(json.loads(json.dumps(m.snapshot())))
    assert clone.snapshot() == m.snapshot()
    assert clone.advance(1)["transfer_mass_kg"] == pytest.approx(
        m.advance(1)["transfer_mass_kg"], abs=1e-9
    )


def test_pure_trials_and_failed_operations_restore_state(monkeypatch):
    _, m = make()
    m.advance(0.1)
    before = m.snapshot()
    cache = m._continuation
    m.derivatives(0, m._state)
    m.analytic_jacobian(0, m._state)
    m.observation()
    assert m.snapshot() == before

    def fail(*a, **kw):
        raise NetworkError("synthetic observation failure")

    monkeypatch.setattr(m, "observation", fail)
    with pytest.raises(NetworkError):
        m.set_resistance(0)
    assert m.snapshot() == before and m._continuation is cache
    with pytest.raises(NetworkError):
        m.advance(0.1)
    assert m.snapshot() == before and m._continuation is None


@pytest.mark.parametrize("change", ["hash", "energy", "history", "model", "direction", "shape"])
def test_bad_checkpoint_rejected(change):
    _, m = make()
    m.advance(4)
    m.set_resistance(10000)
    p = m.snapshot()
    if change == "hash":
        p["configuration"]["line"]["inertance_m_inv"] *= 2
    if change == "energy":
        p["state"][8] += 1
    if change == "history":
        p["active_resistance_pa_s2_kg2"] += 1
    if change == "model":
        p["model_version"] = "2"
    if change == "direction":
        next(e for e in p["events"] if e["kind"] == "flow_zero")["direction"] = True
    if change == "shape":
        p["state"].append(0)
    with pytest.raises(NetworkError):
        InertialStoragePair.from_snapshot(p)


@pytest.mark.parametrize(
    "key,value",
    [("inertance_m_inv", 0), ("resistance_pa_s2_kg2", -1), ("resistance_pa_s2_kg2", float("inf"))],
)
def test_invalid_line_before_output(key, value, tmp_path):
    s = example()
    s["model"]["line"][key] = value
    with pytest.raises(NetworkError):
        run_storage_experiment(s, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_parallel_algebraic_transfer_rejected():
    s = example()
    s["model"]["storage_configuration"]["connection"]["conductance_m3_s_pa"] = 1e-8
    with pytest.raises(NetworkError, match="disabled"):
        prepare(s)


def test_sdk_resistance_commands_and_deadline(tmp_path):
    s = example()
    s.update(duration_s=2, step_s=0.3, timeline=[dict(t=0.7, resistance_pa_s2_kg2=10000)])
    r = run_storage_experiment(s, tmp_path / "run")
    assert r["numerical_acceptance"] and r["accepted_commands"] == 1
    rows = json.loads((tmp_path / "run" / "trajectory.json").read_text())
    pair = [r for r in rows if r["time_s"] == 0.7]
    assert (
        len(pair) == 2
        and pair[0]["mass_flow_kg_s"] == pair[1]["mass_flow_kg_s"]
        and pair[0]["kinetic_energy_j"] == pair[1]["kinetic_energy_j"]
    )
    assert (
        json.loads((tmp_path / "run" / "manifest.json").read_text())["passport"]["model_id"]
        == s["engine"]
    )
    r = run_storage_experiment(s, tmp_path / "deadline", wall_limit_s=1e-12)
    assert r["status"] == "failed" and r["completed_time_s"] == 0
    InertialStoragePair.from_snapshot(
        json.loads((tmp_path / "deadline" / "checkpoint.json").read_text())
    )


def test_exact_equilibrium_has_no_roundtrip_heat_or_false_flow_events():
    spec, _ = make()
    for vessel in spec["model"]["storage_configuration"]["vessels"]:
        vessel.update(initial_gas_pressure_pa=100000, gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2)
    model = prepare(spec)[1]
    assert np.array_equal(model.derivatives(0, model._state), np.zeros(9))
    model.enable_continuation(100)
    model.advance(100)
    assert np.array_equal(model._state, np.zeros(9)) and model.events == []
