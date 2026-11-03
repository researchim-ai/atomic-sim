"""Public tool semantics, stale observations, exact native replay and boundaries."""

import json
from copy import deepcopy

import pytest

from agent_envs import EnvConfig, PWRPlantEnv
from agent_envs.replay import continue_episode
from atomic_agent_tools import AgentToolServer, ToolError
from atomic_agent_tools.protocol import LIMITS, SCHEMAS, canonical, catalog
from atomic_agent_tools.server import digest
from core.status import DomainError


def create(server, name="episode", **config):
    return server.invoke(
        "create_episode", dict(session_id=name, config=dict(episode_horizon_s=0.03, **config))
    )


def guard(view):
    return dict(
        session_id=view["session_id"],
        expected_revision=view["revision"],
        observation_id=view["observation_id"],
    )


def step(server, view, action=None):
    return server.invoke(
        "step_episode",
        dict(**guard(view), action={"rod_command": 0.001} if action is None else action),
    )


def test_matches_native_environment_including_sensor_rng_and_done():
    server = AgentToolServer()
    v = create(server, sensor_noise=0.01, seed=123)
    native = PWRPlantEnv(
        config=EnvConfig(seed=123, episode_horizon_s=0.03, action_dt_s=0.01, sensor_noise=0.01)
    )
    native.reset(seed=123)
    checkpoint = native.export_episode()
    for _ in range(3):
        assert server.invoke("inspect_episode", {"session_id": "episode"}) == v
        v = step(server, v)
        replay = continue_episode(checkpoint, [0.001])
        checkpoint = replay["checkpoint"]
        obs, reward, terminated, truncated, info = (
            replay["observation"],
            replay["reward"],
            replay["terminated"],
            replay["truncated"],
            replay["info"],
        )
        assert [o["value"] for o in v["observation"]] == obs
        assert v["last_transition"] == dict(
            reward=reward, terminated=terminated, truncated=truncated, reason=info["reason"]
        )
        assert server.export_session("episode")["payload"]["episode"] == checkpoint
    assert v["status"] == "completed"
    with pytest.raises(ToolError, match="not active"):
        step(server, v)


def test_public_observation_and_checkpoint_handle_do_not_expose_internal_state():
    server = AgentToolServer()
    v = create(server)
    saved = server.invoke("save_checkpoint", guard(v))
    assert set(saved) == {"checkpoint_id", "session"}
    assert "plant" not in v and "rng" not in canonical(saved) and "sensors" not in v
    assert set(v["observation"][0]) == {
        "name",
        "unit",
        "value",
        "valid",
        "sample_time_s",
        "saturated",
    }
    with pytest.raises(ToolError):
        server.invoke("export_session", {"session_id": "episode"})


def test_restore_rewinds_physics_but_revision_and_observation_tokens_advance():
    server = AgentToolServer()
    v = create(server, sensor_noise=0.01, seed=42)
    saved = server.invoke("save_checkpoint", guard(v))
    first = step(server, v)
    restored = server.invoke(
        "restore_checkpoint", dict(**guard(first), checkpoint_id=saved["checkpoint_id"])
    )
    assert restored["time_s"] == 0.0 and restored["revision"] == 2
    with pytest.raises(ToolError):
        step(server, v)
    second = step(server, restored)
    assert (
        second["observation"] == first["observation"]
        and second["last_transition"] == first["last_transition"]
    )
    assert second["observation_id"] != first["observation_id"]


def test_host_export_import_is_exact_for_native_episode_but_new_generation():
    server = AgentToolServer()
    v = step(server, create(server, sensor_noise=0.01))
    checkpoint = json.loads(json.dumps(server.export_session("episode"), allow_nan=False))
    other = AgentToolServer()
    fork = other.import_session("episode", checkpoint)
    assert fork["observation_id"] != v["observation_id"]
    assert other.export_session("episode") == checkpoint
    a = step(server, v)
    b = step(other, fork)
    assert a["observation"] == b["observation"] and a["last_transition"] == b["last_transition"]
    assert other.export_session("episode") == server.export_session("episode")


@pytest.mark.parametrize(
    "action",
    [
        {},
        {"throttle": 0.5},
        {"rod_command": True},
        {"rod_command": "0"},
        {"rod_command": float("nan")},
        {"rod_command": float("inf")},
        {"rod_command": 1.01},
        {"rod_command": 0, "unknown": 1},
    ],
)
def test_invalid_actions_leave_all_physical_and_sensor_state_unchanged(action):
    server = AgentToolServer()
    v = create(server, sensor_noise=0.01)
    before = server.export_session("episode")
    with pytest.raises(ToolError):
        step(server, v, action)
    assert server.export_session("episode") == before
    assert server.invoke("inspect_episode", {"session_id": "episode"}) == v


@pytest.mark.parametrize(
    "change",
    [dict(expected_revision=True), dict(expected_revision=99), dict(observation_id="0" * 64)],
)
def test_stale_and_malformed_guards_are_atomic(change):
    server = AgentToolServer()
    v = create(server)
    args = guard(v)
    args.update(change)
    with pytest.raises(ToolError):
        server.invoke("step_episode", dict(**args, action={"rod_command": 0.0}))
    assert server.invoke("inspect_episode", {"session_id": "episode"}) == v


def test_cancel_save_restore_and_close_recreate_do_not_accept_old_tokens():
    server = AgentToolServer()
    v = create(server)
    saved = server.invoke("save_checkpoint", guard(v))
    cancelled = server.invoke("cancel_episode", guard(v))
    assert cancelled["status"] == "cancelled" and cancelled["time_s"] == v["time_s"]
    with pytest.raises(ToolError):
        step(server, cancelled)
    restored = server.invoke(
        "restore_checkpoint", dict(**guard(cancelled), checkpoint_id=saved["checkpoint_id"])
    )
    server.invoke("close_episode", guard(restored))
    new = create(server)
    assert new["revision"] == 0 and new["observation_id"] != v["observation_id"]
    with pytest.raises(ToolError):
        step(server, v)
    server.invoke("forget_checkpoint", {"checkpoint_id": saved["checkpoint_id"]})
    with pytest.raises(ToolError):
        server.invoke(
            "restore_checkpoint", dict(**guard(new), checkpoint_id=saved["checkpoint_id"])
        )


def test_checkpoint_capacity_is_explicit_and_duplicate_handles_do_not_cost_slots(monkeypatch):
    monkeypatch.setitem(LIMITS, "max_checkpoints", 1)
    server = AgentToolServer()
    v = create(server)
    saved = server.invoke("save_checkpoint", guard(v))
    assert server.invoke("save_checkpoint", guard(v)) == saved
    v = step(server, v)
    with pytest.raises(ToolError, match="capacity"):
        server.invoke("save_checkpoint", guard(v))
    server.invoke("forget_checkpoint", {"checkpoint_id": saved["checkpoint_id"]})
    assert server.invoke("save_checkpoint", guard(v))["checkpoint_id"] != saved["checkpoint_id"]


def test_resource_counters_are_not_rewound_by_restore(monkeypatch):
    monkeypatch.setitem(LIMITS, "max_model_transactions", 3)
    server = AgentToolServer()
    v = create(server)
    saved = server.invoke("save_checkpoint", guard(v))
    v = step(server, v)
    v = server.invoke("restore_checkpoint", dict(**guard(v), checkpoint_id=saved["checkpoint_id"]))
    with pytest.raises(ToolError, match="budget"):
        step(server, v)
    assert v == server.invoke("inspect_episode", {"session_id": "episode"})


@pytest.mark.parametrize(
    "config",
    [
        {"seed": True},
        {"episode_horizon_s": 11},
        {"action_dt_s": 0.1},
        {"episode_horizon_s": 10, "action_dt_s": 0.001},
        {"action_channels": ["rod_command", "rod_command"]},
        {"unknown": 1},
        {"sensor_faults": [""]},
        {"sensor_noise": float("nan")},
    ],
)
def test_invalid_configuration_does_not_create_a_session(config):
    server = AgentToolServer()
    with pytest.raises(ToolError):
        server.invoke("create_episode", {"session_id": "episode", "config": config})
    with pytest.raises(ToolError):
        server.invoke("inspect_episode", {"session_id": "episode"})
    assert create(server)["revision"] == 0


def test_names_channels_units_and_public_schema_validation():
    server = AgentToolServer()
    with pytest.raises(ToolError):
        server.invoke("create_episode", {"session_id": "../file"})
    v = create(server, action_channels=["throttle", "rod_command"])
    v = step(server, v, {"rod_command": 0.001, "throttle": 0.8})
    assert (
        v["action_channels"]["rod_command"]["unit"] == "1/s"
        and v["action_channels"]["throttle"]["unit"] == "1"
    )
    schema_library = pytest.importorskip("jsonschema")
    for tool in catalog()["tools"]:
        schema_library.Draft202012Validator.check_schema(tool["input_schema"])
    schema_library.validate(
        {
            "session_id": "episode",
            "action": {"rod_command": 0.0},
            "expected_revision": 0,
            "observation_id": "a" * 64,
        },
        SCHEMAS["step_episode"],
    )


def test_returned_catalog_cannot_mutate_runtime_schemas():
    before = canonical(SCHEMAS)
    result = catalog()
    result["tools"][1]["input_schema"]["properties"]["session_id"]["pattern"] = ".*"
    assert canonical(SCHEMAS) == before


def test_program_error_rolls_back_and_is_not_a_physical_failure(monkeypatch):
    server = AgentToolServer()
    v = create(server)
    before = server.export_session("episode")

    def broken(self, action):
        raise ValueError("Program bug")

    monkeypatch.setattr(PWRPlantEnv, "step", broken)
    reply = server.handle_request(
        dict(
            request_id="r1",
            tool="step_episode",
            arguments=dict(**guard(v), action={"rod_command": 0.0}),
        )
    )
    assert not reply["ok"] and reply["error"]["code"] == "internal_error"
    assert server.export_session("episode") == before


def test_physical_domain_stop_is_preserved_as_truncation(monkeypatch):
    from plants.vver import VVERPlant

    server = AgentToolServer()
    v = create(server)

    def rejected(self, *args, **kwargs):
        raise DomainError("Injected domain boundary")

    monkeypatch.setattr(VVERPlant, "advance", rejected)
    after = step(server, v)
    assert after["status"] == "failed" and after["last_transition"]["truncated"]
    assert (
        not after["last_transition"]["terminated"]
        and after["last_transition"]["reason"] == "domain_error"
    )
    assert after["time_s"] == v["time_s"]
    server.import_session("failed_copy", server.export_session("episode"))


@pytest.mark.parametrize(
    "defect", ["checksum", "provenance", "lifecycle", "reward", "flags", "configuration"]
)
def test_host_corrupt_checkpoint_rejected_without_replacing_existing_state(defect):
    server = AgentToolServer()
    v = create(server)
    bad = deepcopy(server.export_session("episode"))
    p = bad["payload"]
    if defect == "provenance":
        p["provenance"]["physics_source_sha256"] = "wrong"
    elif defect == "lifecycle":
        p["phase"] = "completed"
    elif defect == "reward":
        p["transition"]["reward"] = float("nan")
    elif defect == "flags":
        p["transition"]["terminated"] = True
    elif defect == "configuration":
        p["episode"]["config"]["rtol"] = 1e-12
    if defect != "checksum" and defect != "reward":
        bad["sha256"] = digest(p)
    elif defect == "checksum":
        bad["sha256"] = "wrong"
    with pytest.raises((ToolError, ValueError)):
        server.import_session("copy", bad)
    assert server.invoke("inspect_episode", {"session_id": "episode"}) == v


def test_concurrent_actions_commit_only_once():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    server = AgentToolServer()
    v = create(server)
    barrier = Barrier(2)

    def act(_):
        barrier.wait(timeout=10)
        try:
            step(server, v)
            return True
        except ToolError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(act, range(2)))
    assert (
        sum(results) == 1
        and server.invoke("inspect_episode", {"session_id": "episode"})["time_s"] == 0.01
    )


def test_request_budget_and_error_correlation(monkeypatch):
    monkeypatch.setitem(LIMITS, "max_requests", 1)
    server = AgentToolServer()
    reply = server.handle_request(dict(request_id="bad", tool="no_such_tool", arguments={}))
    assert reply["request_id"] == "bad" and not reply["ok"]
    assert (
        server.handle_request(dict(request_id="next", tool="describe_simulator", arguments={}))[
            "error"
        ]["code"]
        == "resource_limit"
    )
