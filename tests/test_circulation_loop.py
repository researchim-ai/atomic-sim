"""Nonlinear circulation, physical port energy, initialization and atomic controls."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.hydraulics.circulation_loop import CirculationLoop, steady_circulation
from scripts.circulation_references import flow_and_work, modal_temperatures, switched_reference
from sdk.circulation_experiment import prepare, run_circulation_experiment


def example():
    return json.loads(Path("scenarios/ai05_circulation_reversal.json").read_text())


def model_for(head=100, flow=-0.5, walls=False):
    spec = example()
    for node in spec["model"]["topology"]["nodes"]:
        node["parameters"]["drive_pressure_pa"] = head if node["id"] == "s0" else 0
        if walls:
            node["parameters"].update(fluid_wall_ua_w_k=100, ambient_ua_w_k=20)
    spec["initialization"]["mass_flow_kg_s"] = flow
    return prepare(spec)[1]


@pytest.mark.parametrize(
    "head,flow", [(0, 1), (0, -1), (100, 0), (-100, 0), (100, -0.5), (-100, 0.5)]
)
@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_flow_work_and_dissipative_heat_match_independent_solution(head, flow, method):
    model = model_for(head, flow)
    model.enable_continuation(10)
    total_capacity = sum(model.capacity)
    initial_k = model.inertia_total * flow**2 / (2 * model.rho)
    for target in (0.5, 2.0, 10.0):
        obs = model.advance(
            target - model.time_s, method=method, max_step_s=0.05, rtol=1e-9, atol_scale=0.1
        )
        exact, work, zero = flow_and_work(
            target, flow, head, model.inertia_total, model.resistance_total, model.rho
        )
        kinetic = model.inertia_total * exact**2 / (2 * model.rho)
        temperature = 300 + (work - kinetic + initial_k) / total_capacity
        assert obs["mass_flow_kg_s"] == pytest.approx(exact, abs=1e-8)
        assert obs["external_head_work_j"] == pytest.approx(work, abs=1e-7)
        assert list(obs["fluid_temperature_k"].values()) == pytest.approx(
            [temperature] * model.n, abs=1e-9
        )
        assert abs(obs["energy_residual_j"]) < 1e-5
        assert max(abs(v) for v in obs["local_energy_residual_w"].values()) < 1e-6
    if zero is not None:
        found = [e for e in model.events if e["kind"] == "flow_zero"]
        assert len(found) == 1 and found[0]["t"] == pytest.approx(zero, abs=1e-6)


@pytest.mark.parametrize("flow", [-0.6, 0.8])
def test_analytic_jacobian_and_nonlinear_energy_identity(flow):
    model = model_for(flow=flow, walls=True)
    y = model._state.copy()
    y[1:4] = [0.5, -0.2, 0.1]
    y[4:7] = [2, 1, 3]
    direction = np.random.default_rng(12).normal(size=len(y))
    eps = 1e-6
    fd = (model.derivatives(0, y + eps * direction) - model.derivatives(0, y - eps * direction)) / (
        2 * eps
    )
    jac = model.analytic_jacobian(0, y)
    assert jac @ direction == pytest.approx(fd, rel=1e-6, abs=1e-7)
    gradient = np.r_[
        model.inertia_total * flow / model.rho, model.capacity, model.wall_capacity, 0, 0
    ]
    identity = (
        np.asarray(jac.T @ gradient).ravel()
        - jac[-2, :].toarray().ravel()
        - jac[-1, :].toarray().ravel()
    )
    identity[0] += model.inertia_total * model.derivatives(0, y)[0] / model.rho
    assert identity == pytest.approx(np.zeros(len(y)), abs=1e-7)


def test_shared_ports_are_antisymmetric_and_pressure_gauge_is_inert():
    model = model_for()
    spec = example()
    spec["model"]["reference"]["pressure_pa"] = 1e6
    other = prepare(spec)[1]
    a = model.advance(3)
    b = other.advance(3)
    assert a["mass_flow_kg_s"] == b["mass_flow_kg_s"]
    assert a["fluid_temperature_k"] == b["fluid_temperature_k"]
    for i, name in enumerate(model.names):
        right = a["ports"][name + ".right"]
        left = a["ports"][model.names[(i + 1) % model.n] + ".left"]
        assert right["pressure_pa"] == left["pressure_pa"]
        assert right["mass_flow_kg_s"] + left["mass_flow_kg_s"] == 0
        assert right["energy_flow_w"] + left["energy_flow_w"] == 0


def test_steady_initialization_matches_uniform_closed_form_and_holds():
    spec = example()["model"]
    for node in spec["topology"]["nodes"]:
        node["parameters"].update(fluid_wall_ua_w_k=100, ambient_ua_w_k=20, wall_power_w=10)
    model = steady_circulation(**spec)
    flow = np.sqrt(100 / model.resistance_total)
    dissipation = model.resistance[0] * flow**3 / model.rho
    wall = 300 + (10 + dissipation) / 20
    fluid = wall + dissipation / 100
    initial = model.observation()
    assert list(initial["fluid_temperature_k"].values()) == pytest.approx([fluid] * 3, abs=1e-9)
    assert list(initial["wall_temperature_k"].values()) == pytest.approx([wall] * 3, abs=1e-9)
    assert initial["flow_acceleration_kg_s2"] == pytest.approx(0, abs=1e-12)
    after = model.advance(100, max_step_s=0.5)
    assert after["fluid_temperature_k"] == pytest.approx(initial["fluid_temperature_k"], abs=1e-9)
    with pytest.raises(NetworkError, match="anchor"):
        steady_circulation(**example()["model"])


def test_wall_pulse_matches_independent_fourier_modes():
    model = model_for(flow=np.sqrt(100 / 150), walls=True)
    spec = model.snapshot()
    spec["initial_wall_temperature_k"]["s0"] = 302
    model = CirculationLoop(
        spec["topology"],
        spec["fluid"],
        spec["reference"],
        300,
        spec["initial_mass_flow_kg_s"],
        spec["initial_temperature_k"],
        spec["initial_wall_temperature_k"],
    )
    for end in (1.0, 10.0, 30.0):
        obs = model.advance(end - model.time_s, max_step_s=0.1, atol_scale=0.1)
        fluid, wall = modal_temperatures(
            end,
            [300] * 3,
            [302, 300, 300],
            np.sqrt(100 / 150),
            1000,
            4200,
            0.01,
            1,
            10,
            1000,
            100,
            20,
            300,
        )
        assert list(obs["fluid_temperature_k"].values()) == pytest.approx(fluid, abs=1e-8)
        assert list(obs["wall_temperature_k"].values()) == pytest.approx(wall, abs=1e-8)


def test_head_switch_reinitializes_pressure_without_energy_jump_and_restores():
    model = model_for(flow=0.8)
    model.enable_continuation(10)
    before = model.advance(3)
    after = model.set_heads({"s0": -100, "s1": 0, "s2": 0})
    assert after["mass_flow_kg_s"] == before["mass_flow_kg_s"]
    assert after["stored_energy_change_j"] == before["stored_energy_change_j"]
    assert after["ports"] != before["ports"]
    checkpoint = model.snapshot()
    restored = CirculationLoop.from_snapshot(checkpoint)
    assert restored.snapshot() == checkpoint
    for m in (model, restored):
        obs = m.advance(7)
        flow, work, zeros = switched_reference(
            10, 0.8, 100, m.inertia_total, m.resistance_total, m.rho, [(3, -100)]
        )
        assert obs["mass_flow_kg_s"] == pytest.approx(flow, abs=1e-8)
        assert obs["external_head_work_j"] == pytest.approx(work, abs=1e-7)
        assert [e["t"] for e in m.events if e["kind"] == "flow_zero"] == pytest.approx(
            zeros, abs=1e-6
        )
    assert model.snapshot() == restored.snapshot()


def test_failed_head_change_and_failed_observer_are_atomic(monkeypatch):
    model = model_for()
    model.advance(0.5)
    before = model.snapshot()
    with pytest.raises(NetworkError):
        model.set_heads({"s0": -1e9, "s1": 0, "s2": 0})
    assert model.snapshot() == before

    def fail(*args, **kwargs):
        raise NetworkError("observer failed")

    monkeypatch.setattr(model, "observation", fail)
    with pytest.raises(NetworkError):
        model.advance(0.5)
    assert model.snapshot() == before


@pytest.mark.parametrize(
    "fault",
    ["version", "branch", "direction", "self", "reference", "parameter", "zero_loss", "medium"],
)
def test_bad_hydraulic_configuration_rejected_before_output(tmp_path, fault):
    spec = example()
    model = spec["model"]
    top = model["topology"]
    if fault == "version":
        top["nodes"][0]["model_version"] = "2"
    elif fault == "branch":
        top["connections"][1]["from"] = "s0.right"
    elif fault == "direction":
        top["connections"][0]["from"] = "s0.left"
    elif fault == "self":
        top["connections"][0]["to"] = "s0.left"
    elif fault == "reference":
        model["reference"]["segment"] = "missing"
    elif fault == "parameter":
        top["nodes"][0]["parameters"]["length_m"] = -1
    elif fault == "zero_loss":
        for node in top["nodes"]:
            node["parameters"]["loss_coefficient"] = 0
    else:
        model["fluid"]["material"] = "undeclared"
    with pytest.raises(NetworkError):
        run_circulation_experiment(spec, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("fault", ["fluid", "topology", "state", "events", "head", "time", "model"])
def test_checkpoint_binds_configuration_and_history(fault):
    model = model_for()
    model.advance(1)
    model.set_heads({"s0": -100, "s1": 0, "s2": 0})
    payload = model.snapshot()
    if fault == "fluid":
        payload["fluid"]["density_kg_m3"] = 900
    elif fault == "topology":
        payload["topology"]["nodes"][0]["parameters"]["loss_coefficient"] = 12
    elif fault == "state":
        payload["state"][1] = -1000
    elif fault == "events":
        payload["events"][0]["previous_pa"][0] = 999
    elif fault == "head":
        payload["active_heads_pa"]["s0"] = 1
    elif fault == "time":
        payload["time_s"] = -1
    else:
        payload["model_id"] = "conservative-euler-liquid-channel"
    with pytest.raises(NetworkError):
        CirculationLoop.from_snapshot(payload)


def test_snapshots_and_parameters_are_detached():
    model = model_for()
    before = model.snapshot()
    copied = model.snapshot()
    copied["fluid"]["density_kg_m3"] = 17
    copied["topology"]["nodes"][0]["parameters"]["drive_pressure_pa"] = 17
    model.fluid["density_kg_m3"] = 19
    model.reference["pressure_pa"] = 1
    assert model.snapshot() == before
    with pytest.raises(ValueError):
        model.capacity[0] = 17


def test_real_sdk_timeline_failure_prefix_and_exclusive_output(tmp_path):
    spec = example()
    spec["timeline"] = [dict(t=2, heads_pa=dict(s0=-100, s1=0, s2=0))]
    report = run_circulation_experiment(spec, tmp_path / "run")
    assert report["numerical_acceptance"] and report["accepted_commands"] == 1
    payload = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert CirculationLoop.from_snapshot(payload).snapshot() == payload
    with pytest.raises(FileExistsError):
        run_circulation_experiment(spec, tmp_path / "run")
    spec["timeline"][0]["heads_pa"]["s0"] = -1e9
    failed = run_circulation_experiment(spec, tmp_path / "failed")
    assert (
        failed["status"] == "failed"
        and failed["completed_time_s"] == 2
        and failed["accepted_commands"] == 0
    )


def test_nonfinite_enthalpy_flux_is_rejected_even_for_zero_flow():
    spec = example()
    spec["model"]["fluid"].update(density_kg_m3=1e-300, heat_capacity_j_kg_k=1e308, area_m2=1)
    spec["initialization"]["mass_flow_kg_s"] = 0
    for node in spec["model"]["topology"]["nodes"]:
        node["parameters"]["drive_pressure_pa"] = 0
    with pytest.raises(NetworkError, match="Nonfinite total-enthalpy"):
        prepare(spec)


def test_cycle_canonicalization_is_independent_of_input_order():
    spec = example()
    _, a = prepare(spec)
    spec["model"]["topology"]["nodes"].reverse()
    spec["model"]["topology"]["connections"].reverse()
    _, b = prepare(spec)
    assert a.snapshot() == b.snapshot()
    assert a.advance(1) == b.advance(1)


def test_disconnected_cycles_are_rejected():
    from physics.hydraulics.circulation_loop import topology_order

    spec = example()["model"]["topology"]
    extra = deepcopy(spec["nodes"][0])
    extra["id"] = "s3"
    spec["nodes"].append(extra)
    spec["connections"] = [
        dict(id="a", **{"from": "s0.right", "to": "s1.left"}),
        dict(id="b", **{"from": "s1.right", "to": "s0.left"}),
        dict(id="c", **{"from": "s2.right", "to": "s3.left"}),
        dict(id="d", **{"from": "s3.right", "to": "s2.left"}),
    ]
    with pytest.raises(NetworkError, match="islands"):
        topology_order(spec, "s0")


def test_unrepresentable_time_step_keeps_state():
    model = model_for()
    model.advance(1)
    before = model.snapshot()
    with pytest.raises(NetworkError):
        model.advance(1e-30)
    assert model.snapshot() == before


def test_wall_deadline_produces_a_restorable_failed_prefix(tmp_path):
    report = run_circulation_experiment(example(), tmp_path / "deadline", wall_limit_s=1e-9)
    assert report["status"] == "failed" and not report["numerical_acceptance"]
    payload = json.loads((tmp_path / "deadline/checkpoint.json").read_text())
    assert CirculationLoop.from_snapshot(payload).snapshot() == payload


def test_zero_flow_generalized_jacobian_is_finite_and_conservative():
    model = model_for(flow=0, walls=True)
    y = model._state.copy()
    y[1:4] = [0, 2, -1]
    jac = model.analytic_jacobian(0, y)
    assert np.isfinite(jac.data).all()
    gradient = np.r_[0, model.capacity, model.wall_capacity, 0, 0]
    residual = (
        np.asarray(jac.T @ gradient).ravel()
        - jac[-2, :].toarray().ravel()
        - jac[-1, :].toarray().ravel()
    )
    residual[0] += model.inertia_total * model.derivatives(0, y)[0] / model.rho
    assert residual == pytest.approx(np.zeros(len(y)), abs=1e-7)


def test_nonuniform_segments_match_transported_mass_reference():
    from scripts.circulation_references import adiabatic_temperatures

    spec = example()
    spec["initialization"]["mass_flow_kg_s"] = -3
    spec["initialization"]["fluid_temperature_k"] = dict(s0=303, s1=300, s2=297)
    for i, node in enumerate(spec["model"]["topology"]["nodes"]):
        node["parameters"].update(
            length_m=[1, 0.7, 1.3][i],
            loss_coefficient=[4, 17, 9][i],
            drive_pressure_pa=10000 if i == 0 else 0,
        )
    model = prepare(spec)[1]
    for end in (0.2, 1.0, 10.0):
        obs = model.advance(end - model.time_s, atol_scale=0.1)
        expected = adiabatic_temperatures(
            end, [303, 300, 297], -3, 10000, 1000, 4200, 0.01, [1, 0.7, 1.3], [4, 17, 9]
        )
        assert list(obs["fluid_temperature_k"].values()) == pytest.approx(expected, abs=1e-8)
        assert abs(obs["energy_residual_j"]) < 1e-5
