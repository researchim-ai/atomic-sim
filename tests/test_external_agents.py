"""Real per-decision processes, permission/CAS guards and native transaction replay."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from atomic_model_agents import jobs, transport
from atomic_model_runtime.episode import EOSEpisode
from atomic_model_workspace.common import atomic_json, digest, read
from examples.eos_external_agent_workflow import descriptor, fixtures

COMMAND = [sys.executable, "-m", "atomic_model_agents.baseline"]


@pytest.fixture
def job(tmp_path):
    scenario, hold, _ = fixtures()
    folder = tmp_path / "job"
    jobs.create(scenario, descriptor(hold), folder, COMMAND)
    return folder


def custom(tmp_path, body, *, timeout=1, rss=128 * 1024**2):
    path = tmp_path / "agent.py"
    path.write_text(body)
    command = [sys.executable, str(path)]
    scenario, hold, _ = fixtures()
    desc = descriptor(hold)
    desc.update(decision_timeout_s=timeout, rss_limit_bytes=rss)
    folder = tmp_path / "custom"
    jobs.create(scenario, desc, folder, command)
    return folder, command


GOOD = """import json,sys
r=json.loads(sys.stdin.readline())
s=r['state'].copy();s['decisions']=s.get('decisions',0)+1
reply=dict(schema_version=1,request_id=r['request_id'],decision=r['decision'],observation_sha256=r['observation_sha256'],heat_power_w={x['name']:0 for x in r['allowed_actions']},state=s)
"""


def test_permissions_native_equivalence_and_explicit_resume(job, tmp_path):
    before = jobs.load(job)["payload"]
    partial = jobs.run(job, COMMAND, max_decisions=1)
    row = partial["payload"]["rows"][1]
    assert [c["name"] for c in row["request"]["observation"]["channels"]] == ["wall.w0"]
    assert row["request"]["allowed_actions"] == [dict(name="wall.w0", unit="W")]
    assert all(v == 0 for k, v in row["model"]["heat_power_w"].items() if k != "wall.w0")
    assert (
        partial["payload"]["agent_state"]["decisions"] == 1
        and before["agent_state"].get("decisions") is None
    )
    with pytest.raises(ValueError, match="Stale"):
        jobs.resume(job, expected_sha256="0" * 64, command=COMMAND)
    jobs.resume(job, expected_sha256=partial["sha256"], command=COMMAND)
    final = jobs.run(job, COMMAND)
    assert (
        final["payload"]["status"] == "completed"
        and final["payload"]["agent_state"]["decisions"] == 3
    )
    assert final["payload"]["rows"][:2] == partial["payload"]["rows"]
    native = EOSEpisode(before["scenario"]["config"])
    while not native._done:
        native.step([500 if n == "wall.w0" else 0 for n in native.config.action_channels])
    assert native.export_episode() == final["payload"]["episode"]
    jobs.export(job, tmp_path / "export")
    assert jobs.replay(tmp_path / "export")["agent_executed"] is False


@pytest.mark.parametrize(
    "defect",
    ["unknown_sensor", "unknown_action", "duplicate", "empty", "schema", "timeout", "rss", "state"],
)
def test_descriptor_admission_before_output(tmp_path, defect):
    scenario, hold, _ = fixtures()
    d = descriptor(hold)
    if defect == "unknown_sensor":
        d["observations"] = ["hidden"]
    elif defect == "unknown_action":
        d["actions"] = ["hidden"]
    elif defect == "duplicate":
        d["actions"] *= 2
    elif defect == "empty":
        d["observations"] = []
    elif defect == "schema":
        d["schema_version"] = True
    elif defect == "timeout":
        d["decision_timeout_s"] = float("nan")
    elif defect == "rss":
        d["rss_limit_bytes"] = True
    else:
        d["initial_state"] = "x" * 40000
    with pytest.raises(ValueError):
        jobs.create(scenario, d, tmp_path / "bad", COMMAND)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "reply['request_id']='stale'",
        "reply['decision']=True",
        "reply['observation_sha256']='0'*64",
        "reply['heat_power_w']['primary.channel.c0']=1",
        "reply['heat_power_w']={}",
        "reply['heat_power_w']['wall.w0']=True",
        "reply['heat_power_w']['wall.w0']=1e99",
        "reply['state']='x'*40000",
        "reply['extra']='bad'",
    ],
)
def test_invalid_reply_preserves_native_and_agent_state(tmp_path, mutation):
    folder, command = custom(tmp_path, GOOD + "\n" + mutation + "\nprint(json.dumps(reply))")
    before = jobs.load(folder)["payload"]
    result = jobs.run(folder, command)["payload"]
    assert result["status"] == "failed" and result["reason"] == "agent_response_contract"
    assert (
        result["episode"] == before["episode"]
        and result["agent_state"] == before["agent_state"]
        and result["rows"] == before["rows"]
    )
    assert result["attempts"][0]["outcome"] == "agent_response_contract"
    jobs.load(folder)


@pytest.mark.parametrize(
    "body,reason",
    [
        ("print('not JSON')", "agent_jsonl"),
        ("print('{}\\n{}')", "agent_jsonl"),
        ('print(\'{"x":1,"x":2}\')', "agent_jsonl"),
        ("print('NaN')", "agent_jsonl"),
        ("import sys;sys.exit(7)", "agent_exit"),
        ("print('x'*100000)", "agent_stdout_limit"),
        ("import sys;sys.stderr.write('x'*20000)", "agent_stderr_limit"),
        ("import time;time.sleep(5)", "agent_timeout"),
    ],
)
def test_process_errors_are_bounded_and_transactional(tmp_path, body, reason):
    folder, command = custom(tmp_path, body, timeout=0.5)
    before = jobs.load(folder)["payload"]
    result = jobs.run(folder, command)["payload"]
    assert result["status"] == "failed" and result["reason"] == reason, result
    assert result["episode"] == before["episode"] and len(result["rows"]) == 1
    assert result["attempts"][0]["telemetry"]["wall_s"] < 2
    jobs.load(folder)


@pytest.mark.parametrize("close_pipes", [False, True])
def test_rss_limit_with_open_or_closed_pipes(tmp_path, close_pipes):
    body = (
        "import os\nos.close(0);os.close(1);os.close(2)\n" if close_pipes else ""
    ) + "import time\nmemory=bytearray(64*1024**2)\ntime.sleep(3)"
    folder, command = custom(tmp_path, body, rss=16 * 1024**2)
    value = jobs.run(folder, command)["payload"]
    assert value["reason"] == "agent_rss_limit" and len(value["rows"]) == 1, value


def test_cancel_inflight_and_resume_same_identity(tmp_path):
    flag = tmp_path / "once"
    body = (
        GOOD
        + f"\nfrom pathlib import Path\nimport time\nf=Path({str(flag)!r})\nif not f.exists():\n f.write_text('once');time.sleep(5)\nprint(json.dumps(reply))\n"
    )
    folder, command = custom(tmp_path, body, timeout=2)
    results = []
    thread = threading.Thread(target=lambda: results.append(jobs.run(folder, command)))
    thread.start()
    deadline = time.monotonic() + 5
    while not flag.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    jobs.cancel(folder)
    thread.join(timeout=5)
    assert not thread.is_alive()
    value = results[0]
    assert (
        value["payload"]["status"] == "cancelled"
        and value["payload"]["agent_state"].get("decisions") is None
    )
    jobs.resume(folder, expected_sha256=value["sha256"], command=command)
    assert jobs.run(folder, command)["payload"]["agent_state"]["decisions"] == 3


def test_command_and_entry_change_cannot_resume(job, tmp_path):
    value = jobs.run(job, COMMAND, max_decisions=1)
    with pytest.raises(ValueError, match="identity"):
        jobs.resume(job, expected_sha256=value["sha256"], command=[*COMMAND, "different"])
    folder, command = custom(tmp_path, GOOD + "\nprint(json.dumps(reply))")
    value = jobs.run(folder, command, max_decisions=1)
    Path(command[1]).write_text(GOOD + "\nprint(json.dumps(reply))\n# changed")
    with pytest.raises(ValueError, match="identity"):
        jobs.resume(folder, expected_sha256=value["sha256"], command=command)


def test_replay_detects_native_and_state_tampering(job, tmp_path):
    jobs.run(job, COMMAND)
    jobs.export(job, tmp_path / "export")
    path = tmp_path / "export"
    head = read(path / "HEAD.json")
    checkpoint = read(path / head["checkpoint"])
    checkpoint["payload"]["rows"][1]["model"]["observation"]["channels"][0]["value"] += 1
    checkpoint["sha256"] = digest(checkpoint["payload"])
    name = "revisions/" + checkpoint["sha256"] + ".json"
    (path / head["checkpoint"]).unlink()
    atomic_json(path / name, checkpoint)
    head.update(checkpoint=name, sha256=checkpoint["sha256"])
    atomic_json(path / "HEAD.json", head)
    import hashlib

    manifest = read(path / "manifest.json")
    manifest["artifacts"] = {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="native row"):
        jobs.replay(path)


def test_cli_and_explicit_no_implicit_resume(job):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atomic_model_agents",
            "run",
            str(job),
            "--max-decisions",
            "1",
            "--agent-command",
            *COMMAND,
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["accepted_decisions"] == 1
    assert json.loads(result.stdout)["status"] == "paused"


def test_fresh_process_uses_recorded_state_only(tmp_path):
    body = (
        GOOD
        + "\nimport os\nassert s.get('last_pid')!=os.getpid()\ns['last_pid']=os.getpid()\nprint(json.dumps(reply))"
    )
    folder, command = custom(tmp_path, body)
    value = jobs.run(folder, command)["payload"]
    assert value["status"] == "completed"
    assert len({row["response"]["state"]["last_pid"] for row in value["rows"][1:]}) == 3


def test_entry_modified_during_reply_is_not_applied(tmp_path):
    folder, command = custom(
        tmp_path,
        GOOD
        + "\nfrom pathlib import Path\np=Path(__file__);p.write_text(p.read_text()+'\\n# changed')\nprint(json.dumps(reply))",
    )
    before = jobs.load(folder)["payload"]
    value = jobs.run(folder, command)["payload"]
    assert value["reason"] == "agent_identity_changed"
    assert value["episode"] == before["episode"] and value["agent_state"] == before["agent_state"]


def test_replay_never_starts_external_agent(tmp_path, monkeypatch):
    folder, command = custom(tmp_path, GOOD + "\nprint(json.dumps(reply))")
    jobs.run(folder, command)
    jobs.export(folder, tmp_path / "export")
    Path(command[1]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("agent process executed during replay")

    monkeypatch.setattr(transport, "exchange", forbidden)
    assert jobs.replay(tmp_path / "export")["replay_exact"]


def test_total_deadline_pauses_without_accepting_late_response(tmp_path):
    folder, command = custom(tmp_path, "import time;time.sleep(5)", timeout=2)
    value = jobs.run(folder, command, deadline_s=0.1)["payload"]
    assert value["status"] == "paused" and len(value["rows"]) == 1


def test_descendant_retaining_pipe_is_terminated_on_timeout(tmp_path):
    marker = tmp_path / "pid"
    body = f"import subprocess,sys\nfrom pathlib import Path\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\nPath({str(marker)!r}).write_text(str(p.pid))\n"
    folder, command = custom(tmp_path, body, timeout=0.5)
    value = jobs.run(folder, command)["payload"]
    assert value["reason"] == "agent_timeout"
    pid = int(marker.read_text())
    proc = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 3
    while proc.exists():
        if proc.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("defect", ["negative_rss", "bool_bytes", "missing", "unknown_outcome"])
def test_typed_attempt_telemetry_cannot_be_relabelled(job, defect):
    value = jobs.run(job, COMMAND, max_decisions=1)
    attempt = value["payload"]["attempts"][0]
    if defect == "negative_rss":
        attempt["telemetry"]["peak_rss_bytes"] = -1
    elif defect == "bool_bytes":
        attempt["telemetry"]["stdout_bytes"] = True
    elif defect == "missing":
        attempt["telemetry"] = {}
    else:
        attempt["outcome"] = "claimed-winner"
    value["sha256"] = digest(value["payload"])
    name = "revisions/" + value["sha256"] + ".json"
    atomic_json(job / name, value)
    atomic_json(job / "HEAD.json", dict(schema_version=1, checkpoint=name, sha256=value["sha256"]))
    with pytest.raises(ValueError, match="telemetry|outcome"):
        jobs.load(job)


def test_native_refusal_retains_checkpoint_and_typed_criterion(job, monkeypatch):
    from atomic_model_runtime.episode import EpisodeRejected

    before = jobs.load(job)["payload"]

    def refuse(*args, **kwargs):
        raise EpisodeRejected("energy", "retained native energy criterion")

    monkeypatch.setattr(EOSEpisode, "step", refuse)
    value = jobs.run(job, COMMAND)["payload"]
    assert value["status"] == "rejected" and value["reason"] == "numerical:energy"
    assert value["episode"] == before["episode"] and value["agent_state"] == before["agent_state"]
    assert jobs.load(job)["payload"] == value
