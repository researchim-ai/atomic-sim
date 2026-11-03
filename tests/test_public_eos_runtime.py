"""Installed EOS semantics, resource anchoring, agent route and experiment resume."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from atomic_agent_tools import AgentToolServer, ToolError
from atomic_model_runtime import create_eos_assembly, create_eos_episode
from atomic_model_runtime.episode import EOSEpisode, model_card
from atomic_model_runtime.experiment import run_experiment
from atomic_model_runtime.resources import ROOT, resolve, resource


def guarded(v):
    return dict(
        session_id=v["session_id"],
        expected_revision=v["revision"],
        observation_id=v["observation_id"],
    )


def scenario():
    env = EOSEpisode(dict(episode_horizon_s=0.03))
    powers = env._model.snapshot()["active_heat_power_w"].copy()
    powers["wall.w0"] = 500.0
    return dict(
        schema_version=1,
        model_id=model_card()["model_id"],
        config=env.config.as_dict(),
        timeline=[dict(decision=2, heat_power_w=powers)],
    )


@pytest.mark.parametrize("preset", ["mixed_ports", "uniform", "gravity"])
def test_public_episode_and_full_spec_api_match_original_assembly(preset):
    e = EOSEpisode(dict(preset=preset, episode_horizon_s=0.03))
    direct = create_eos_assembly(e._spec)
    powers = direct.snapshot()["active_heat_power_w"].copy()
    powers["wall.w0"] = 500.0
    for i in range(3):
        obs, reward, terminated, truncated, info = e.step(
            np.array([powers[k] for k in e.config.action_channels])
        )
        state = direct.step(
            powers,
            min((i + 1) * 0.01, 0.03) - direct.snapshot()["time_s"],
            expected_revision=direct.revision,
            command_id=f"agent.step-{i + 1}",
        )
        assert obs.tolist() == list(state["port_temperatures_k"].values())
        assert reward == 0 and not terminated and truncated == (i == 2)
    assert e.export_episode()["model_checkpoint"] == direct.checkpoint()


def test_resource_resolution_ignores_working_directory_and_shadow_budgets(tmp_path, monkeypatch):
    (tmp_path / "validation").mkdir()
    (tmp_path / "validation/ai05_controlled_assembly_budget_v1.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    e = create_eos_episode()
    e.step([0.0] * len(e.config.action_channels))
    assert e.export_episode()["steps"] == 1
    r = resource("validation/ai05_controlled_assembly_budget_v1.json")
    assert str(r) == "validation/ai05_controlled_assembly_budget_v1.json"
    assert Path(r).parent == ROOT / "validation"
    assert resolve("artifacts/user.json") == Path("artifacts/user.json")
    assert resolve("validation/user-output.json") == Path("validation/user-output.json")


@pytest.mark.parametrize("bad", ["../data", "/tmp/input", "unowned/file"])
def test_resource_names_reject_escape(bad):
    with pytest.raises(ValueError):
        resource(bad)


def test_exact_json_restart_and_non_accumulating_decision_clock():
    e = EOSEpisode()
    values = [0.0] * len(e.config.action_channels)
    for _ in range(5):
        e.step(values)
    cp = json.loads(json.dumps(e.export_episode()))
    other = EOSEpisode.from_episode(cp)
    for _ in range(5):
        e.step(values)
        other.step(values)
    assert e.export_episode() == other.export_episode()
    assert e.export_episode()["steps"] == 10 and e.observation_metadata()["time_s"] == 0.1


@pytest.mark.parametrize(
    "defect", ["steps", "observation", "history", "receipt", "config", "done", "reward"]
)
def test_corrupt_eos_episode_rejected(defect):
    e = EOSEpisode()
    e.step([0.0] * len(e.config.action_channels))
    p = deepcopy(e.export_episode())
    if defect == "steps":
        p["steps"] = True
    elif defect == "observation":
        p["observation"][0] += 1
    elif defect == "history":
        p["action_log"][0]["commands"]["wall.w0"] = 1
    elif defect == "receipt":
        p["action_log"][0]["dt_s"] += 0.001
    elif defect == "config":
        p["config"]["unknown"] = 1
    elif defect == "done":
        p["done"] = True
    elif defect == "reward":
        p["reward_total"] = 1.0
    with pytest.raises(ValueError):
        EOSEpisode.from_episode(p)


@pytest.mark.parametrize(
    "action", [[], ["0"] * 14, [True] * 14, [float("nan")] * 14, [10001.0] * 14]
)
def test_bad_episode_action_is_atomic(action):
    e = EOSEpisode()
    before = e.export_episode()
    with pytest.raises(ValueError):
        e.step(action)
    assert e.export_episode() == before


def test_agent_tools_eos_route_replay_and_native_pwr_coexist():
    server = AgentToolServer()
    eos = server.invoke(
        "create_eos_episode", dict(session_id="eos", config=dict(episode_horizon_s=0.02))
    )
    pwr = server.invoke("create_episode", dict(session_id="pwr"))
    assert (
        eos["model"]["model_id"] == "eos-thermal-assembly-v1"
        and pwr["model"]["model_id"] != "eos-thermal-assembly-v1"
    )
    assert all(v["unit"] == "W" for v in eos["action_channels"].values())
    saved = server.invoke("save_checkpoint", guarded(eos))
    action = {name: 0.0 for name in eos["action_channels"]}
    action["wall.w0"] = 500.0
    first = server.invoke("step_episode", dict(**guarded(eos), action=action))
    restored = server.invoke(
        "restore_checkpoint", dict(**guarded(first), checkpoint_id=saved["checkpoint_id"])
    )
    again = server.invoke("step_episode", dict(**guarded(restored), action=action))
    assert (
        again["observation"] == first["observation"]
        and again["last_transition"] == first["last_transition"]
    )
    other = AgentToolServer()
    view = other.import_session("eos", server.export_session("eos"))
    assert view["observation"] == again["observation"]


def test_agent_rejected_eos_action_does_not_commit_model_or_lifecycle():
    server = AgentToolServer()
    v = server.invoke("create_eos_episode", dict(session_id="eos"))
    before = server.export_session("eos")
    action = {name: 0.0 for name in v["action_channels"]}
    action["wall.w0"] = 10001.0
    with pytest.raises(ToolError):
        server.invoke("step_episode", dict(**guarded(v), action=action))
    assert server.export_session("eos") == before


def test_experiment_csv_units_hashes_and_exact_pause_resume(tmp_path):
    spec = scenario()
    full = run_experiment(spec, tmp_path / "full")
    paused = run_experiment(spec, tmp_path / "paused", max_decisions=1)
    assert paused["status"] == "paused"
    cp = json.loads((tmp_path / "paused/checkpoint.json").read_text())
    resumed = run_experiment(spec, tmp_path / "resumed", checkpoint=cp)
    assert resumed["status"] == "completed" and resumed["start_time_s"] == 0.01
    assert json.loads((tmp_path / "full/checkpoint.json").read_text()) == json.loads(
        (tmp_path / "resumed/checkpoint.json").read_text()
    )
    import hashlib

    for name, expected in full["artifacts"].items():
        assert hashlib.sha256((tmp_path / "full" / name).read_bytes()).hexdigest() == expected
    assert full["units"]["energy_residual_j"] == "J"
    assert len((tmp_path / "full/trajectory.csv").read_text().splitlines()) == 5


@pytest.mark.parametrize("defect", ["unknown", "duplicate", "clock", "mask", "future_power"])
def test_entire_timeline_rejected_before_output_creation(tmp_path, defect):
    spec = scenario()
    if defect == "unknown":
        spec["unknown"] = 0
    elif defect == "duplicate":
        spec["timeline"].append(deepcopy(spec["timeline"][0]))
    elif defect == "clock":
        spec["timeline"][0]["decision"] = 100
    elif defect == "mask":
        spec["timeline"][0]["heat_power_w"].pop("wall.w0")
    elif defect == "future_power":
        spec["timeline"][0]["heat_power_w"]["wall.w0"] = float("nan")
    with pytest.raises(ValueError):
        run_experiment(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_checkpoint_does_not_allow_different_future_scenario(tmp_path):
    spec = scenario()
    run_experiment(spec, tmp_path / "paused", max_decisions=1)
    cp = json.loads((tmp_path / "paused/checkpoint.json").read_text())
    spec["timeline"][0]["heat_power_w"]["wall.w0"] = 100.0
    with pytest.raises(ValueError):
        run_experiment(spec, tmp_path / "different", checkpoint=cp)
    assert not (tmp_path / "different").exists()


def test_jsonl_eos_worker_uses_resources_outside_project_cwd(tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    with subprocess.Popen(
        [sys.executable, "-m", "atomic_agent_tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=env,
        text=True,
    ) as proc:

        def call(tool, args):
            proc.stdin.write(json.dumps(dict(request_id="r", tool=tool, arguments=args)) + "\n")
            proc.stdin.flush()
            response = json.loads(proc.stdout.readline())
            assert response["ok"], response
            return response["result"]

        v = call("create_eos_episode", dict(session_id="eos", config=dict(episode_horizon_s=0.02)))
        action = {key: 0.0 for key in v["action_channels"]}
        action["wall.w0"] = 500.0
        v = call("step_episode", dict(**guarded(v), action=action))
        assert v["time_s"] == 0.01 and v["model"]["model_id"] == model_card()["model_id"]
        v = call("step_episode", dict(**guarded(v), action=action))
        assert v["status"] == "completed"
        proc.stdin.close()
        proc.wait(timeout=60)
        assert proc.returncode == 0
