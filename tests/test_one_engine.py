"""Один шаг сессии для SDK, сценария, датасета и пульта."""

import json
from pathlib import Path

import pytest

from core.status import Applicability, DomainError
from generate_dataset import generate_vver_trajectory
from scenarios.session import run_scenario
from sdk.api import create_session, restore_snapshot, snapshot_json, step_session
from sdk.console import VVERConsole


def test_commands_and_snapshot_roundtrip():
    plant = create_session("vver")
    plant.set_boron_ppm(1500.0)
    assert plant.reactivity() < 0.0
    step_session(plant, 0.05, {"rod_command": 0.05, "feed_bias_kg_s": -100.0})
    assert plant.rod_command == pytest.approx(0.05)
    assert plant.feed_bias_kg_s == pytest.approx(-100.0)
    payload = json.loads(snapshot_json(plant))
    other = create_session("vver")
    restore_snapshot(other, payload)
    assert other.rod_command == pytest.approx(0.05)
    assert other.feed_bias_kg_s == pytest.approx(-100.0)
    assert other.network.volumes["core"].boron_fraction * 1.0e6 == pytest.approx(1500.0, rel=1e-6)
    with pytest.raises(DomainError) as rejected:
        create_session("rbmk")
    assert rejected.value.applicability == Applicability.UNSUPPORTED


def test_legacy_scenario_is_unsupported_and_vver_file_is_hashed():
    with pytest.raises(DomainError) as rejected:
        run_scenario(Path("scenarios/basic_ramp.yaml"))
    assert rejected.value.applicability == Applicability.UNSUPPORTED
    report = run_scenario(Path("scenarios/vver_load_rejection.yaml"))
    assert report["model_id"] == "vver-lumped-loop"
    assert len(report["sha256"]) == 64
    assert report["rows"][1]["steam_flow_kg_s"] < report["rows"][0]["steam_flow_kg_s"]
    again = run_scenario(Path("scenarios/vver_load_rejection.yaml"))
    assert again["sha256"] == report["sha256"]


def test_dataset_and_console_use_vver_session():
    rows = generate_vver_trajectory(3, 0.2, 0.2, "rod")
    assert rows[0]["meta"]["model_id"] == "vver-lumped-loop"
    assert "T_clad_C" not in rows[0]["obs"]
    assert rows[0]["obs"]["power_MW"] > 1000.0
    console = VVERConsole()
    state = console.step(dt=0.1)
    direct = create_session("vver")
    observed = step_session(direct, 0.1, {"rod_command": 0.0, "throttle": 1.0, "feed_bias_kg_s": 0.0})
    assert state["thermal_power_mw"] == pytest.approx(observed["thermal_power_w"] / 1.0e6, rel=1e-4)
    assert "flux" not in state
