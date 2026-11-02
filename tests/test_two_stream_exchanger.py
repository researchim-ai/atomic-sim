"""Physical identities, directionality, restart and transactional failures."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from physics.thermal.two_stream_exchanger import TwoStreamExchanger
from sdk.exchanger_experiment import prepare, restore, run_exchanger_experiment


@pytest.fixture
def spec():
    return json.loads(Path("scenarios/ai05_exchanger_counterflow.json").read_text())


def model(spec):
    return prepare(spec)[1]


def one_cell(spec, temperatures=(330, 300, 315)):
    spec = deepcopy(spec)
    spec["timeline"] = []
    spec["model"]["cells"] = 1
    for key, value in spec["model"].items():
        if isinstance(value, list):
            spec["model"][key] = value[:1]
    for key in ("a_cp_j_kg_k", "b_cp_j_kg_k"):
        spec["model"][key] = 1000
    for key in ("a_cell_mass_kg", "b_cell_mass_kg"):
        spec["model"][key] = [1]
    spec["model"]["wall_capacity_j_k"] = [1000]
    for key in ("a_wall_ua_w_k", "b_wall_ua_w_k"):
        spec["model"][key] = [100]
    spec["controls"].update(a_mass_flow_kg_s=0, b_mass_flow_kg_s=0)
    spec["initialization"]["temperature_k"] = dict(
        zip(("a", "b", "wall"), ([v] for v in temperatures))
    )
    return spec


@pytest.mark.parametrize(
    "temperatures,decay,amplitudes,mean",
    [
        ((330, 300, 315), 0.1, (15, -15, 0), 315),
        ((330, 330, 300), 0.3, (10, 10, -20), 320),
    ],
)
def test_closed_analytic_eigenmodes(spec, temperatures, decay, amplitudes, mean):
    plant = model(one_cell(spec, temperatures))
    row = plant.advance(7)
    expected = mean + np.array(amplitudes) * np.exp(-decay * 7)
    assert np.array(list(row["temperature_k"].values())).ravel() == pytest.approx(
        expected, abs=1e-10
    )
    assert abs(row["stored_energy_change_j"]) < 1e-8


@pytest.mark.parametrize("flow", [-0.4, 0.4])
def test_single_stirred_tank_analytic(spec, flow):
    spec = one_cell(spec)
    spec["model"]["a_wall_ua_w_k"] = [0]
    spec["controls"]["a_mass_flow_kg_s"] = flow
    plant = model(spec)
    row = plant.advance(3)
    exact = 370 + (330 - 370) * np.exp(-abs(flow) * 3)
    assert row["temperature_k"]["a"][0] == pytest.approx(exact, abs=1e-10)
    assert row["ports"]["a"]["outlet_end"] == ("right" if flow > 0 else "left")


def test_reversing_coordinate_and_flows_is_symmetric(spec):
    spec["initialization"]["temperature_k"]["a"] = [320, 330, 340]
    spec["initialization"]["temperature_k"]["b"] = [310, 300, 290]
    reverse = deepcopy(spec)
    for key, value in reverse["model"].items():
        if isinstance(value, list):
            value.reverse()
    for values in reverse["initialization"]["temperature_k"].values():
        values.reverse()
    for side in ("a", "b"):
        reverse["controls"][f"{side}_mass_flow_kg_s"] *= -1
    left, right = model(spec).advance(11), model(reverse).advance(11)
    for side in ("a", "b", "wall"):
        assert left["temperature_k"][side] == pytest.approx(
            right["temperature_k"][side][::-1], abs=1e-9
        )


def test_steady_initialization_and_residual(spec):
    spec["initialization"] = {"mode": "steady"}
    plant = model(spec)
    before = plant.observation()
    after = plant.advance(100)
    assert (
        max(abs(v) for row in plant.initialization_report()["energy_rate_residual_w"] for v in row)
        < 1e-6
    )
    for key in ("a", "b", "wall"):
        assert after["temperature_k"][key] == pytest.approx(before["temperature_k"][key], abs=1e-9)


def test_unanchored_steady_is_rejected(spec):
    spec = one_cell(spec)
    spec["initialization"] = {"mode": "steady"}
    with pytest.raises(NetworkError, match="unanchored"):
        model(spec)


def test_ambient_heater_and_wall_storage(spec):
    spec["model"]["ambient_ua_w_k"] = [1, 2, 3]
    spec["model"]["wall_power_w"] = [100, 200, 300]
    row = model(spec).advance(12)
    assert abs(row["energy_residual_j"]) < 1e-5
    assert abs(row["port_power_residual_w"]) < 1e-6
    assert row["ambient_heat_out_w"] > 0
    assert any(abs(x) > 0 for x in row["integrated_net_cell_energy_j"][2])


def test_checkpoint_roundtrip_and_continuation(spec):
    plant = model(spec)
    plant.advance(5)
    boundary = plant.boundary_controls
    boundary.update(a_mass_flow_kg_s=-0.2)
    plant.set_controls(boundary)
    snapshot = plant.snapshot()
    restarted = TwoStreamExchanger.from_snapshot(snapshot)
    assert restarted.snapshot() == snapshot
    assert restarted.advance(7) == plant.advance(7)


@pytest.mark.parametrize("defect", ["hash", "time", "energy", "history", "shape", "bool", "schema"])
def test_corrupt_checkpoints_rejected(spec, defect):
    plant = model(spec)
    plant.advance(2)
    snap = plant.snapshot()
    if defect == "hash":
        snap["configuration"]["cells"] += 1
    elif defect == "time":
        snap["time_s"] = -1
    elif defect == "energy":
        snap["cell_energy_integrals_j"][0] += 1
    elif defect == "history":
        snap["controls"]["a_mass_flow_kg_s"] = 0
    elif defect == "shape":
        snap["temperature_k"].pop()
    elif defect == "bool":
        snap["temperature_k"][0] = True
    else:
        snap["schema_version"] = True
    with pytest.raises(NetworkError):
        TwoStreamExchanger.from_snapshot(snap)


def test_failed_advance_does_not_accept_state(spec, monkeypatch):
    plant = model(spec)
    before = plant.snapshot()

    def reject(*args):
        raise NetworkError("synthetic observation failure")

    monkeypatch.setattr(plant, "_observe", reject)
    with pytest.raises(NetworkError):
        plant.advance(1)
    assert plant.snapshot() == before


def test_controls_are_atomic_and_defensively_copied(spec):
    plant = model(spec)
    before = plant.snapshot()
    invalid = plant.boundary_controls
    invalid.update(a_mass_flow_kg_s=-1, b_inlet_temperature_k=-2)
    with pytest.raises(NetworkError):
        plant.set_controls(invalid)
    assert plant.snapshot() == before
    config = plant.configuration
    config["wall_capacity_j_k"][0] = -1
    assert plant.configuration == before["configuration"]


@pytest.mark.parametrize("step", [0.1, 0.3, 5])
def test_sdk_event_grid_restart_and_manifests(spec, tmp_path, step):
    spec.update(duration_s=3, step_s=step)
    boundary = deepcopy(spec["controls"])
    boundary["a_mass_flow_kg_s"] = -0.4
    spec["timeline"] = [dict(t=0, controls=boundary), dict(t=1.37, controls=spec["controls"])]
    report = run_exchanger_experiment(spec, tmp_path / "full")
    assert report["numerical_acceptance"]
    assert [e["t"] for e in report["events"]] == [0, 1.37]
    assert report["completed_time_s"] == 3
    first = deepcopy(spec)
    first["duration_s"] = 1.5
    run_exchanger_experiment(first, tmp_path / "prefix")
    checkpoint = json.loads((tmp_path / "prefix/checkpoint.json").read_text())
    resumed = run_exchanger_experiment(spec, tmp_path / "resumed", checkpoint=checkpoint)
    assert resumed["numerical_acceptance"]
    a = json.loads((tmp_path / "full/checkpoint.json").read_text())["temperature_k"]
    b = json.loads((tmp_path / "resumed/checkpoint.json").read_text())["temperature_k"]
    assert a == pytest.approx(b, abs=1e-9)


def test_restart_rejects_changed_scenario(spec):
    plant = model(spec)
    plant.advance(3)
    spec["model"]["a_cp_j_kg_k"] += 1
    with pytest.raises(NetworkError, match="definition"):
        restore(spec, plant.snapshot())


def test_sdk_rolls_back_whole_output_interval_after_control(spec, tmp_path, monkeypatch):
    spec.update(duration_s=5, step_s=5)
    spec["timeline"] = [dict(t=1, controls=dict(spec["controls"], a_mass_flow_kg_s=-0.4))]
    original = TwoStreamExchanger.advance

    def fail_after_command(self, dt):
        if self.events:
            raise NetworkError("synthetic failure after interior command")
        return original(self, dt)

    monkeypatch.setattr(TwoStreamExchanger, "advance", fail_after_command)
    result = run_exchanger_experiment(spec, tmp_path / "failed")
    assert result["status"] == "failed"
    assert result["completed_time_s"] == 0
    assert result["events"] == []
    assert len(json.loads((tmp_path / "failed/trajectory.json").read_text())) == 1
    assert json.loads((tmp_path / "failed/checkpoint.json").read_text()) == model(spec).snapshot()


def test_deadline_exports_consistent_prefix(spec, tmp_path):
    result = run_exchanger_experiment(spec, tmp_path / "deadline", wall_limit_s=1e-15)
    assert result["status"] == "failed"
    snapshot = json.loads((tmp_path / "deadline/checkpoint.json").read_text())
    assert TwoStreamExchanger.from_snapshot(snapshot).time_s == 0


@pytest.mark.parametrize("defect", ["cells", "mass", "temperature", "flow", "timeline", "resource"])
def test_invalid_inputs_fail_before_output(spec, tmp_path, defect):
    if defect == "cells":
        spec["model"]["cells"] = True
    elif defect == "mass":
        spec["model"]["a_cell_mass_kg"][0] = 0
    elif defect == "temperature":
        spec["initialization"]["temperature_k"]["a"][0] = -1
    elif defect == "flow":
        spec["controls"]["a_mass_flow_kg_s"] = float("nan")
    elif defect == "timeline":
        spec["timeline"].reverse()
    else:
        spec["duration_s"] = 100000
    with pytest.raises(NetworkError):
        run_exchanger_experiment(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
