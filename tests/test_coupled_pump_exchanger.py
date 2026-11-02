"""Coupled hydraulic/thermal energy, pure residual, restart and atomic failure."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.thermal.coupled_pump_exchanger import CoupledPumpExchanger
from sdk.coupled_exchanger_experiment import prepare, restore, run_coupled_exchanger_experiment


@pytest.fixture
def spec():
    return json.loads(Path("scenarios/ai05_coupled_pump_exchanger.json").read_text())


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_simultaneous_heat_transfer_and_energy(spec, method):
    plant = prepare(spec)[1]
    row = plant.advance(2, method=method, rtol=1e-10)
    assert max(row["heat_into_loops_w"]["a"]) < 0
    assert min(row["heat_into_loops_w"]["b"]) > 0
    assert abs(row["total_energy_residual_j"]) < 1e-5
    assert abs(row["shared_energy_residual_j"]) < 1e-5
    assert abs(row["reciprocal_heat_residual_w"]) < 1e-8
    assert row["loops"]["a"]["loop"]["mass_flow_kg_s"] != 0.5
    # Both hydraulic powers, casing leakage and finite wall energy remain visible.
    assert row["loops"]["a"]["leakage_heat_j"] > 0
    assert row["shared_wall_energy_change_j"] != 0


@pytest.mark.parametrize("reverse", [False, True])
def test_jacobian_matches_independent_finite_difference_and_rhs_is_pure(spec, reverse):
    plant = prepare(spec)[1]
    state = plant._state.copy()
    if reverse:
        for model, offset in zip(plant._models, plant._offsets):
            state[offset] *= -1
            state[offset + model._size] *= -1
    before = plant.snapshot()
    analytic = plant.analytic_jacobian(0.0, state).toarray()
    numeric = np.zeros_like(analytic)
    for index in range(len(state)):
        step = 1e-4 * max(1.0, abs(state[index]))
        plus, minus = state.copy(), state.copy()
        plus[index] += step
        minus[index] -= step
        numeric[:, index] = (plant.derivatives(0.0, plus) - plant.derivatives(0.0, minus)) / (
            2 * step
        )
    assert np.max(abs(analytic - numeric) / (1 + abs(numeric))) < 1e-6
    assert plant.snapshot() == before


def test_shared_heat_is_internal_not_external_input(spec):
    for raw in spec["loops"].values():
        raw.update(initial_omega_rad_s=0, motor_torque_nm=0)
        raw["loop"]["initialization"]["mass_flow_kg_s"] = 0
        for node in raw["loop"]["model"]["topology"]["nodes"]:
            node["parameters"].update(ambient_ua_w_k=0, wall_power_w=0)
    plant = prepare(spec)[1]
    row = plant.advance(5)
    assert row["coupling_heat_into_loops_j"]["a"] < 0
    assert row["coupling_heat_into_loops_j"]["b"] > 0
    assert abs(row["external_heat_j"]) < 1e-8
    assert abs(row["stored_energy_change_j"]) < 1e-6


def test_steady_coupled_initialization_without_individual_hot_loop_sink(spec):
    spec["initialization"] = {"mode": "steady"}
    plant = prepare(spec)[1]
    report = plant.initialization_report()
    for values in report["loops"].values():
        assert abs(values["momentum_residual_pa"]) < 1e-6
        assert abs(values["shaft_residual_nm"]) < 1e-8
        assert (
            max(
                abs(v)
                for v in values["fluid_energy_residual_w"] + values["casing_energy_residual_w"]
            )
            < 1e-6
        )
    before = plant.observation()
    after = plant.advance(10)
    assert after["wall_temperature_k"] == pytest.approx(before["wall_temperature_k"], abs=1e-9)
    for side in ("a", "b"):
        assert list(after["loops"][side]["loop"]["fluid_temperature_k"].values()) == pytest.approx(
            list(before["loops"][side]["loop"]["fluid_temperature_k"].values()), abs=1e-9
        )


def test_unanchored_steady_is_rejected(spec):
    spec["initialization"] = {"mode": "steady"}
    for raw in spec["loops"].values():
        for node in raw["loop"]["model"]["topology"]["nodes"]:
            node["parameters"]["ambient_ua_w_k"] = 0
    with pytest.raises(NetworkError, match="unanchored"):
        prepare(spec)


def test_shared_wall_heater_and_ambient(spec):
    for cell in spec["exchanger"]["cells"]:
        cell.update(ambient_ua_w_k=10, wall_power_w=100)
    row = prepare(spec)[1].advance(1)
    assert row["wall_external_heat_j"] < 0
    assert abs(row["total_energy_residual_j"]) < 1e-5


def test_checkpoint_and_controls_roundtrip(spec):
    plant = prepare(spec)[1]
    plant.advance(0.5)
    plant.set_controls(dict(a_motor_torque_nm=-0.2, b_motor_torque_nm=0))
    snap = plant.snapshot()
    other = CoupledPumpExchanger.from_snapshot(snap)
    assert other.snapshot() == snap
    left, right = plant.advance(1), other.advance(1)
    assert left == right


def test_initialization_report_keeps_initial_controls(spec):
    plant = prepare(spec)[1]
    before = plant.initialization_report()
    plant.set_controls(dict(a_motor_torque_nm=0, b_motor_torque_nm=0))
    assert plant.initialization_report() == before


def test_failed_step_rolls_back(spec, monkeypatch):
    plant = prepare(spec)[1]
    before = plant.snapshot()

    def fail(*args):
        raise NetworkError("synthetic acceptance failure")

    monkeypatch.setattr(plant, "observation", fail)
    with pytest.raises(NetworkError):
        plant.advance(0.1)
    assert plant.snapshot() == before


def test_controls_rollback_including_both_motors(spec, monkeypatch):
    plant = prepare(spec)[1]
    before = plant.snapshot()
    torques = [m._torque for m in plant._models]

    def fail(*args):
        raise NetworkError("synthetic controls rejection")

    monkeypatch.setattr(plant, "observation", fail)
    with pytest.raises(NetworkError):
        plant.set_controls(dict(a_motor_torque_nm=-1, b_motor_torque_nm=-2))
    assert plant.snapshot() == before
    assert [m._torque for m in plant._models] == torques


@pytest.mark.parametrize(
    "defect", ["wall_energy", "loop_energy", "history", "shape", "hash", "schema", "time", "bool"]
)
def test_corrupt_snapshots_rejected(spec, defect):
    plant = prepare(spec)[1]
    plant.advance(0.1)
    snap = plant.snapshot()
    if defect == "wall_energy":
        snap["state"][plant._wall_start] += 0.1
    elif defect == "loop_energy":
        snap["state"][1] += 0.1
    elif defect == "history":
        snap["controls"]["a_motor_torque_nm"] = 0
    elif defect == "shape":
        snap["state"].pop()
    elif defect == "hash":
        snap["configuration"]["energy_tolerance_j"] += 1
    elif defect == "schema":
        snap["schema_version"] = True
    elif defect == "time":
        snap["time_s"] = -1
    else:
        snap["state"][0] = True
    with pytest.raises(NetworkError):
        CoupledPumpExchanger.from_snapshot(snap)


def test_mapping_and_atomic_controls_validation(spec):
    spec["exchanger"]["cells"][1]["a_segment"] = "s0"
    with pytest.raises(NetworkError, match="at most one"):
        prepare(spec)


@pytest.mark.parametrize("initial", ["given", "steady"])
def test_sdk_restart_exact_commands_and_prefix(spec, tmp_path, initial):
    spec["initialization"] = {"mode": initial}
    spec.update(duration_s=2, step_s=0.3)
    spec["timeline"] = [
        dict(t=0, controls=dict(a_motor_torque_nm=0, b_motor_torque_nm=0.2)),
        dict(t=1.37, controls=dict(a_motor_torque_nm=-0.2, b_motor_torque_nm=0.2)),
    ]
    full = run_coupled_exchanger_experiment(spec, tmp_path / "full")
    assert full["numerical_acceptance"]
    prefix = deepcopy(spec)
    prefix["duration_s"] = 1.5
    run_coupled_exchanger_experiment(prefix, tmp_path / "prefix")
    payload = json.loads((tmp_path / "prefix/checkpoint.json").read_text())
    resumed = run_coupled_exchanger_experiment(spec, tmp_path / "resumed", checkpoint=payload)
    assert resumed["numerical_acceptance"]
    a = json.loads((tmp_path / "full/checkpoint.json").read_text())["state"]
    b = json.loads((tmp_path / "resumed/checkpoint.json").read_text())["state"]
    assert a == pytest.approx(b, abs=1e-6)
    tampered = deepcopy(spec)
    tampered["timeline"][0]["controls"]["a_motor_torque_nm"] = 1
    with pytest.raises(NetworkError, match="prefix"):
        restore(tampered, payload)


def test_sdk_rolls_back_interval_after_internal_command(spec, tmp_path, monkeypatch):
    spec.update(duration_s=1, step_s=1)
    spec["timeline"] = [dict(t=0.3, controls=dict(a_motor_torque_nm=0, b_motor_torque_nm=0.2))]
    original = CoupledPumpExchanger.advance

    def fail_after_command(self, dt, **kwargs):
        if self.events:
            raise NetworkError("synthetic interior failure")
        return original(self, dt, **kwargs)

    monkeypatch.setattr(CoupledPumpExchanger, "advance", fail_after_command)
    report = run_coupled_exchanger_experiment(spec, tmp_path / "failed")
    assert report["status"] == "failed"
    assert report["completed_time_s"] == 0
    assert report["events"] == []
    snap = json.loads((tmp_path / "failed/checkpoint.json").read_text())
    assert snap == prepare(spec)[1].snapshot()


def test_deadline_keeps_consistent_checkpoint(spec, tmp_path):
    report = run_coupled_exchanger_experiment(spec, tmp_path / "deadline", wall_limit_s=1e-15)
    assert report["status"] == "failed"
    snap = json.loads((tmp_path / "deadline/checkpoint.json").read_text())
    assert CoupledPumpExchanger.from_snapshot(snap).time_s == 0
