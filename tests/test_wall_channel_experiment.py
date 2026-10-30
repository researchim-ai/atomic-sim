import csv
import json

import pytest

from physics.hydraulics.wall_channel import WallChannel
from physics.materials.iapws import saturation_properties
from sdk.channel_experiment import run_channel_experiment, validate_channel_experiment


def spec():
    liquid, vapor = saturation_properties(7e6)
    h0 = float(liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg))
    return dict(
        schema_version=1,
        name="wall-switch",
        engine="isobaric-wall-channel",
        duration_s=2.0,
        step_s=0.5,
        channel=dict(
            cells=1,
            total_volume_m3=0.01,
            pressure_pa=7e6,
            inlet_enthalpy_j_kg=h0,
            inlet_mass_flow_kg_s=0.0,
            heat_w=5000.0,
        ),
        wall=dict(
            total_heat_capacity_j_k=1000.0,
            total_ua_w_k=100.0,
            initial_temperature_k=[float(liquid.T_k) + 50.0],
            power_off_time_s=0.7,
        ),
        solver=dict(rtol=1e-9, method="Radau", max_step_s=0.1),
    )


def test_switch_is_an_output_boundary_and_checkpoint_restores(tmp_path):
    folder = tmp_path / "run"
    report = run_channel_experiment(spec(), folder)
    assert report["status"] == "completed"
    assert report["numerical_acceptance"]
    assert report["events"] == [{"kind": "heater_off", "t": 0.7}]
    with (folder / "trajectory.csv").open() as stream:
        row = next(row for row in csv.DictReader(stream) if float(row["time_s"]) == 0.7)
    assert float(row["heat_w"]) == 0.0
    assert float(row["wall_to_fluid_w"]) > 0
    payload = json.loads((folder / "checkpoint.json").read_text())
    assert WallChannel.from_snapshot(payload).snapshot() == payload
    normalized = json.loads((folder / "input.json").read_text())
    assert validate_channel_experiment(normalized) == normalized


def test_wall_cannot_be_ignored_by_bare_engine():
    configured = spec()
    configured["engine"] = "isobaric-channel"
    with pytest.raises(ValueError, match="engine"):
        validate_channel_experiment(configured)


def test_wall_engine_requires_complete_wall_config():
    configured = spec()
    configured.pop("wall")
    with pytest.raises(ValueError, match="стенки"):
        validate_channel_experiment(configured)


def test_future_switch_is_not_reported_as_already_applied(tmp_path):
    configured = spec()
    configured["wall"]["power_off_time_s"] = 3.0
    report = run_channel_experiment(configured, tmp_path / "run")
    assert report["events"] == []
    assert report["status"] == "completed"
