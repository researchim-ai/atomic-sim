"""Split leakage, independent evaluation, IPC failures and report rebuilding."""

import sys
import time
from copy import deepcopy

import numpy as np
import pytest

from agent_envs.benchmark import campaign, policy_command, rebuild, run_episode, tune
from agent_envs.benchmark_metrics import metrics, summarize
from agent_envs.benchmark_pack import digest, generate_pack, validate_pack
from agent_envs.benchmark_policy import BaselinePolicy
from agent_envs.policy_process import PolicyError, PolicyProcess
from sdk.experiment import _source_fingerprint


def short_pack():
    pack = generate_pack()
    for split, cases in pack["splits"].items():
        if split != "test":
            pack["splits"][split] = cases[:1]
        for case in pack["splits"][split]:
            cfg = case["config"]
            cfg.update(episode_horizon_s=0.02, action_dt_s=0.01, timeline=[])
            group = dict(cfg)
            group.pop("seed")
            case["group_id"] = digest(group)
    return validate_pack(pack)


def tuning_fixture(pack):
    parameters = dict(kp=0.001, ki=0.0001, kd=0, output_limit=0.0005)
    return dict(
        test_used=False,
        training_digest=digest(pack["splits"]["train"]),
        development_digest=digest(pack["splits"]["dev"]),
        selected_parameters=parameters,
        candidates=[
            dict(
                parameters=parameters,
                train=dict(
                    episodes=len(pack["splits"]["train"]), completed=1, mean_iae_completed_s=0
                ),
            )
        ],
        source_sha256=_source_fingerprint(),
    )


def test_pack_counts_seed_and_configuration_disjointness():
    pack = generate_pack()
    assert len(pack["splits"]["train"]) == 32 and len(pack["splits"]["dev"]) == 16
    assert len(pack["splits"]["test"]) == 80
    assert pack == generate_pack()
    assert pack["blind_test"] is False
    bad = deepcopy(pack)
    bad["splits"]["test"][0]["config"] = deepcopy(bad["splits"]["train"][0]["config"])
    bad["splits"]["test"][0]["config"]["seed"] = 90000
    group = dict(bad["splits"]["test"][0]["config"])
    group.pop("seed")
    bad["splits"]["test"][0]["group_id"] = digest(group)
    with pytest.raises(ValueError, match="leakage"):
        validate_pack(bad)
    bad = deepcopy(pack)
    bad["blind_test"] = True
    with pytest.raises(ValueError, match="public"):
        validate_pack(bad)


@pytest.mark.parametrize("kind", ["hold", "pid", "rule"])
def test_real_policy_process_receives_only_public_protocol_and_resets(kind):
    params = dict(kp=0.001, ki=0.0001, kd=0, output_limit=0.0005) if kind == "pid" else None
    case = short_pack()["splits"]["train"][0]
    with PolicyProcess(policy_command(kind, params)) as policy:
        first = run_episode(case, policy)
        second = run_episode(case, policy)
    assert metrics(first)["completed"]
    assert first["steps"] == second["steps"]
    assert first["checkpoint"]["rng"] == second["checkpoint"]["rng"]


@pytest.mark.parametrize(
    "script",
    [
        "import sys,time; sys.stdin.readline(); sys.stdout.write('{'); sys.stdout.flush(); time.sleep(30)",
        "import sys,time; sys.stdin.readline(); time.sleep(30)",
    ],
)
def test_partial_line_and_silent_worker_have_enforced_deadlines(script):
    start = time.monotonic()
    with PolicyProcess([sys.executable, "-u", "-c", script], timeout_s=0.2) as worker:
        with pytest.raises(PolicyError, match="deadline"):
            worker.reset(1, 0.1)
    assert time.monotonic() - start < 5


@pytest.mark.parametrize(
    "answer",
    [
        {"action": [float("nan")], "diagnosis": "healthy"},
        {"action": [2.0], "diagnosis": "healthy"},
        {"action": [True], "diagnosis": "healthy"},
        {"action": [0.0], "diagnosis": "healthy", "truth": 1},
    ],
)
def test_bad_policy_cannot_get_success_by_invalid_action(answer):
    class Worker:
        def reset(self, *args):
            pass

        def act(self, *args):
            if set(answer) != {"action", "diagnosis"} or not np.isfinite(answer["action"]).all():
                raise PolicyError("invalid response")
            return answer

    raw = run_episode(short_pack()["splits"]["train"][0], Worker())
    m = metrics(raw)
    assert raw["reason"] == "policy_error" and not m["completed"]
    assert raw["checkpoint"]["plant"]["time_s"] == 0


def test_independent_metrics_ignore_env_reward_and_catch_early_finish():
    case = short_pack()["splits"]["train"][0]
    with PolicyProcess(policy_command("hold")) as worker:
        raw = run_episode(case, worker)
    original = metrics(raw)
    modified = deepcopy(raw)
    for row in modified["steps"]:
        row["reward"] = 1e10
    other = metrics(modified)
    assert original["iae_s"] == other["iae_s"]
    assert original["action_cost"] == other["action_cost"]
    assert original["cumulative_reward"] != other["cumulative_reward"]
    modified["steps"] = modified["steps"][:1]
    assert metrics(modified)["completed"] is False


def test_tuning_never_evaluates_test_and_validates_selected_candidate(monkeypatch, tmp_path):
    import agent_envs.benchmark as module

    pack = short_pack()
    seen = []

    def fake(cases, command, folder, name):
        seen.extend(c["id"] for c in cases)
        return [
            dict(
                policy="pid",
                case_id=c["id"],
                family=c["family"],
                metrics=dict(completed=True, iae_s=0.0, diagnosis_correct=True, wall_time_s=0.0),
            )
            for c in cases
        ]

    monkeypatch.setattr(module, "run_cases", fake)
    result = tune(pack, tmp_path / "tune")
    assert all(not k.startswith("test-") for k in seen)
    assert result["test_used"] is False
    result["test_used"] = True
    with pytest.raises(ValueError):
        module.verify_tuning(pack, result)


def test_actual_campaign_rebuild_and_checksum_failure(tmp_path):
    pack = short_pack()
    path = tmp_path / "campaign"
    report = campaign(pack, tuning_fixture(pack), path, split="dev")
    assert report["recorded_episodes"] == 3 and report["all_episodes_completed"]
    assert rebuild(path) == report
    episode = next((path / "hold").glob("*.json"))
    episode.write_text(episode.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        rebuild(path)


def test_metrics_and_bootstrap_are_invariant_to_record_order():
    rows = []
    for policy in ("hold", "pid"):
        for i in range(4):
            rows.append(
                dict(
                    policy=policy,
                    case_id=str(i),
                    family="hold" if i < 2 else "tracking",
                    metrics=dict(
                        completed=True,
                        iae_s=float(i) + (policy == "pid") * 0.1,
                        diagnosis_correct=True,
                        wall_time_s=0.1,
                    ),
                )
            )
    a = summarize(rows)
    assert a == summarize(list(reversed(rows)))
    assert a["paired"][0]["mean_delta_iae_s"] == pytest.approx(-0.1)
    assert a["paired"][0]["ci95"] == pytest.approx([-0.1, -0.1])


def test_pid_antiwindup_and_invalid_measurement_safe_hold():
    pid = BaselinePolicy("pid", dict(kp=2, ki=1, kd=0, output_limit=0.01))
    pid.reset(1, 0.1)
    meta = dict(valid=[True] * 4, saturated=[False] * 4)
    for _ in range(20):
        assert pid.act([1.1, 500, 15, 0.5], meta)["action"][0] == 0.01
    assert pid.pid.integral == 0
    meta["valid"][0] = False
    result = pid.act([1.1, 500, 15, 0.5], meta)
    assert result == dict(action=[0.0], diagnosis="sensor_fault")


@pytest.mark.parametrize(
    "response",
    [
        '{"action":[NaN],"diagnosis":"healthy"}',
        '{"action":[0],"diagnosis":"healthy","request":"hidden_truth"}',
        '{"action":[2],"diagnosis":"healthy"}',
    ],
)
def test_actual_ipc_rejects_nan_privileged_requests_and_illegal_actions(response):
    script = (
        "import sys;sys.stdin.readline();print('{\"ready\":true}',flush=True);sys.stdin.readline();print("
        + repr(response)
        + ",flush=True)"
    )
    with PolicyProcess([sys.executable, "-u", "-c", script]) as worker:
        raw = run_episode(short_pack()["splits"]["train"][0], worker)
    assert raw["reason"] == "policy_error" and metrics(raw)["completed"] is False


def test_policy_wire_contains_no_seed_case_manifest_or_truth():
    script = """import json,sys
request=json.loads(sys.stdin.readline())
assert set(request)=={'schema_version','op','target','dt_s'}
print('{"ready":true}',flush=True)
while True:
 line=sys.stdin.readline()
 if not line: break
 request=json.loads(line)
 assert set(request)=={'schema_version','op','observation','measurement'}
 assert not {'truth','config','seed','family','diagnosis_label','case_id'} & set(request)
 print('{"action":[0.0],"diagnosis":"healthy"}',flush=True)
"""
    with PolicyProcess([sys.executable, "-u", "-c", script]) as worker:
        raw = run_episode(short_pack()["splits"]["train"][0], worker)
    assert metrics(raw)["completed"]


@pytest.mark.parametrize(
    "identifier", ["../outside", "test-hold-00/../../outside", "/tmp/outside", True]
)
def test_case_ids_cannot_escape_campaign_directory(identifier):
    pack = generate_pack()
    pack["splits"]["test"][0]["id"] = identifier
    with pytest.raises(ValueError, match="Case ID"):
        validate_pack(pack)
