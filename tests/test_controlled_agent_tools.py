"""Real controlled-core tool calls, hostile requests and JSONL checkpoint resume."""

import json
import subprocess
import sys
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from atomic_agent_tools import AgentToolServer, ToolError
from atomic_agent_tools.protocol import LIMITS, SCHEMAS, canonical, validate
from atomic_agent_tools.server import digest
from atomic_pwr_agent import Episode


def create(server, **config):
    return server.invoke("create_controlled_episode", dict(session_id="pwr", config=config))


def guard(view):
    return dict(
        session_id=view["session_id"],
        expected_revision=view["revision"],
        observation_id=view["observation_id"],
    )


def step(server, view, action=None):
    return server.invoke("step_controlled_episode", dict(**guard(view), action=action or {}))


def test_full_state_matches_python_and_inspection_is_read_only():
    server = AgentToolServer()
    view = create(server, seed=17, max_steps=2)
    reference = Episode(seed=17, max_steps=2)
    assert view["model"]["model_id"] == "pwr-reduced-flow-controlled"
    assert view["observation_kind"] == "full_model_state"
    for commands in ({"throttle": 0.95}, {}):
        assert server.invoke("inspect_episode", {"session_id": "pwr"}) == view
        view = step(server, view, commands)
        expected = reference.step(commands, observation_id=reference.observe()["observation_id"])
        assert [v["value"] for v in view["observation"]] == expected["observation"]["state"]
        assert view["core_revision"] == reference.observe()["revision"]
    assert view["status"] == "completed" and view["last_transition"]["truncated"]
    assert view["last_transition"]["reward"] is None
    with pytest.raises(ToolError, match="not active"):
        step(server, view)


def test_restore_rewinds_core_but_not_tool_revision_and_rejects_stale_action():
    server = AgentToolServer()
    initial = create(server)
    saved = server.invoke("save_checkpoint", guard(initial))
    first = step(server, initial, {"rod_command": 0.001})
    restored = server.invoke(
        "restore_checkpoint", dict(**guard(first), checkpoint_id=saved["checkpoint_id"])
    )
    assert restored["core_revision"] == 0 and restored["revision"] == 2
    assert restored["observation"] == initial["observation"]
    with pytest.raises(ToolError, match="fresh observation"):
        step(server, initial)
    second = step(server, restored, {"rod_command": 0.001})
    assert second["observation"] == first["observation"]
    assert second["revision"] == 3 and second["core_revision"] == 1


@pytest.mark.parametrize(
    "action",
    [
        dict(scram=1),
        dict(throttle=True),
        dict(throttle=float("nan")),
        dict(motor_on=True, motor_off=True),
        dict(unknown=1),
    ],
)
def test_invalid_command_is_noncommitting(action):
    server = AgentToolServer()
    view = create(server)
    saved = server.export_session("pwr")
    result = server.handle_request(
        dict(
            request_id="bad",
            tool="step_controlled_episode",
            arguments=dict(**guard(view), action=action),
        )
    )
    assert not result["ok"] and result["error"]["code"] == "invalid_argument"
    assert server.export_session("pwr") == saved


def test_scram_cancel_and_tool_model_mismatch():
    server = AgentToolServer()
    view = create(server)
    with pytest.raises(ToolError, match="step_controlled_episode"):
        server.invoke("step_episode", dict(**guard(view), action={"rod_command": 0}))
    view = step(server, view, {"scram": True})
    assert view["last_transition"]["terminated"] and view["status"] == "completed"
    saved = server.export_session("pwr")
    other = AgentToolServer()
    assert other.import_session("restored", saved)["status"] == "completed"
    server.invoke("close_episode", guard(view))
    view = create(server)
    view = server.invoke("cancel_episode", guard(view))
    assert view["status"] == "cancelled"
    saved = server.export_session("pwr")
    assert other.import_session("cancelled", saved)["status"] == "cancelled"
    with pytest.raises(ToolError):
        step(server, view)


@pytest.mark.parametrize("defect", ["flags", "phase", "config", "checksum", "reward"])
def test_host_checkpoint_tamper_rejected(defect):
    server = AgentToolServer()
    create(server)
    saved = deepcopy(server.export_session("pwr"))
    p = saved["payload"]
    if defect == "flags":
        p["transition"]["terminated"] = 0
    elif defect == "phase":
        p["phase"] = "completed"
    elif defect == "config":
        p["episode"]["checkpoint"]["payload"]["config"]["max_steps"] = 0
    elif defect == "reward":
        p["transition"]["reward"] = 0
    else:
        saved["sha256"] = "x"
    if defect != "checksum":
        saved["sha256"] = digest(p)
    with pytest.raises(ToolError):
        server.import_session("broken", saved)
    with pytest.raises(ToolError, match="Unknown session"):
        server.invoke("inspect_episode", {"session_id": "broken"})


def test_failed_solver_candidate_does_not_commit(monkeypatch):
    server = AgentToolServer()
    view = create(server)
    saved = server.export_session("pwr")
    original = Episode.step

    def reject(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("injected failure after candidate changed")

    monkeypatch.setattr(Episode, "step", reject)
    with pytest.raises(ToolError, match="no state committed"):
        step(server, view)
    assert server.export_session("pwr") == saved


def test_schemas_agree_on_boolean_flags_and_duration_domain():
    schema = SCHEMAS["step_controlled_episode"]
    request = dict(
        session_id="pwr", expected_revision=0, observation_id="a" * 64, action={"scram": True}
    )
    validate(schema, request)
    Draft202012Validator(schema).validate(request)
    request["action"]["scram"] = 1
    with pytest.raises(ToolError):
        validate(schema, request)
    assert list(Draft202012Validator(schema).iter_errors(request))
    for duration in (0, -1, True, 6):
        with pytest.raises(ToolError):
            validate(
                SCHEMAS["create_controlled_episode"],
                dict(session_id="pwr", config=dict(duration_s=duration)),
            )


def test_resource_limit_and_cross_model_restore(monkeypatch):
    server = AgentToolServer()
    view = create(server)
    saved = server.invoke("save_checkpoint", guard(view))
    legacy = server.invoke("create_episode", dict(session_id="legacy"))
    with pytest.raises(ToolError, match="Cannot change controlled model"):
        server.invoke(
            "restore_checkpoint", dict(**guard(legacy), checkpoint_id=saved["checkpoint_id"])
        )
    monkeypatch.setitem(LIMITS, "max_model_transactions", server._model_transactions)
    before = server.export_session("pwr")
    with pytest.raises(ToolError, match="budget"):
        step(server, view)
    assert server.export_session("pwr") == before


def test_jsonl_cli_host_resume(tmp_path):
    path = tmp_path / "controlled.json"
    request = dict(
        request_id="create",
        tool="create_controlled_episode",
        arguments=dict(session_id="pwr", config=dict(seed=17)),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "atomic_agent_tools",
            "--session-id",
            "pwr",
            "--checkpoint-out",
            str(path),
        ],
        input=canonical(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    original = json.loads(proc.stdout)["result"]
    request = dict(request_id="inspect", tool="inspect_episode", arguments=dict(session_id="pwr"))
    proc = subprocess.run(
        [sys.executable, "-m", "atomic_agent_tools", "--session-id", "pwr", "--resume", str(path)],
        input=canonical(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)["result"]
    assert result["observation"] == original["observation"]
    assert result["observation_id"] != original["observation_id"]


def test_exported_case_replays_in_training_runtime():
    from atomic_model_training.episode import run_episode

    server = AgentToolServer()
    view = create(server, max_steps=2)
    args = dict(case_id="llm_load", family="load_change")
    with pytest.raises(ToolError, match="No accepted steps"):
        server.invoke("export_controlled_case", dict(**guard(view), **args))
    view = step(server, view, {"throttle": 0.95})
    view = step(server, view)
    before = server.export_session("pwr")
    exported = server.invoke("export_controlled_case", dict(**guard(view), **args))
    result = run_episode(exported["case"])
    assert result["checkpoint"] == before["payload"]["episode"]["checkpoint"]["payload"]["core"]
    assert server.export_session("pwr") == before
    assert not exported["model"]["validated"]
