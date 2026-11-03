"""Real JSONL subprocess transport and trusted host checkpoint persistence."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atomic_agent_tools import AgentToolServer
from atomic_agent_tools.protocol import LIMITS, canonical

ROOT = Path(__file__).resolve().parents[1]


def message(tool, args, request_id="r1"):
    return dict(request_id=request_id, tool=tool, arguments=args)


def run(lines, *args):
    return subprocess.run(
        [sys.executable, "-m", "atomic_agent_tools", *map(str, args)],
        input=lines,
        capture_output=True,
        cwd=ROOT,
        timeout=60,
    )


@pytest.mark.parametrize(
    "line",
    [
        b"not-json\n",
        b'{"request_id":"a","request_id":"b"}\n',
        b'{"x":NaN}\n',
        b"\xff\n",
        b"[" * 2000 + b"]" * 2000 + b"\n",
    ],
)
def test_bad_input_is_one_error_and_worker_continues(line):
    result = run(line + (canonical(message("describe_simulator", {})) + "\n").encode())
    assert result.returncode == 0
    replies = [json.loads(s) for s in result.stdout.splitlines()]
    assert (
        len(replies) == 2
        and replies[0]["error"]["code"] in {"invalid_json", "invalid_argument"}
        and replies[1]["ok"]
    )


def test_oversized_line_drains_once_and_recovers():
    result = run(
        b"x" * (LIMITS["max_line_bytes"] + 20)
        + b"\n"
        + (canonical(message("describe_simulator", {})) + "\n").encode()
    )
    replies = [json.loads(s) for s in result.stdout.splitlines()]
    assert (
        len(replies) == 2
        and replies[0]["error"]["code"] == "request_too_large"
        and replies[1]["ok"]
    )


def test_host_checkpoint_file_can_resume_in_a_fresh_process(tmp_path):
    checkpoint = tmp_path / "episode.json"
    create = message(
        "create_episode",
        dict(session_id="episode", config=dict(seed=7, sensor_noise=0.01, episode_horizon_s=0.03)),
    )
    result = run((canonical(create) + "\n").encode(), "--checkpoint-out", checkpoint)
    assert result.returncode == 0, result.stderr.decode()
    saved = json.loads(checkpoint.read_text())
    assert saved["payload"]["episode"]["plant"]["time_s"] == 0.0
    resumed = run(
        (canonical(message("inspect_episode", {"session_id": "episode"})) + "\n").encode(),
        "--resume",
        checkpoint,
    )
    assert resumed.returncode == 0, resumed.stderr.decode()
    view = json.loads(resumed.stdout)["result"]
    original = json.loads(result.stdout)["result"]
    assert original["observation"] == view["observation"]
    assert original["observation_id"] != view["observation_id"]
    host = AgentToolServer()
    loaded = host.import_session("episode", saved)
    args = dict(
        session_id="episode",
        action={"rod_command": 0.001},
        expected_revision=loaded["revision"],
        observation_id=loaded["observation_id"],
    )
    expected = host.invoke("step_episode", args)
    # An interactive client reads the new guard, then sends its first action.
    with subprocess.Popen(
        [sys.executable, "-m", "atomic_agent_tools", "--resume", str(checkpoint)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=ROOT,
        text=True,
    ) as proc:
        proc.stdin.write(canonical(message("inspect_episode", {"session_id": "episode"})) + "\n")
        proc.stdin.flush()
        current = json.loads(proc.stdout.readline())["result"]
        args.update(expected_revision=current["revision"], observation_id=current["observation_id"])
        proc.stdin.write(canonical(message("step_episode", args)) + "\n")
        proc.stdin.flush()
        actual = json.loads(proc.stdout.readline())
        proc.stdin.close()
        proc.wait(timeout=60)
        assert proc.returncode == 0
    assert actual["result"]["observation"] == expected["observation"]
    assert actual["result"]["last_transition"] == expected["last_transition"]


def test_existing_host_output_is_not_overwritten(tmp_path):
    target = tmp_path / "existing.json"
    target.write_text("keep")
    result = run(b"", "--checkpoint-out", target)
    assert result.returncode != 0 and target.read_text() == "keep"


def test_host_resume_rejects_bad_checksum(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"payload":{},"sha256":"wrong"}')
    result = run(b"", "--resume", path)
    assert result.returncode != 0 and result.stdout == b""


def test_tool_request_cannot_supply_host_file_paths(tmp_path):
    result = run(
        (
            canonical(
                message(
                    "create_episode",
                    dict(session_id="episode", checkpoint_out=str(tmp_path / "file")),
                )
            )
            + "\n"
        ).encode()
    )
    reply = json.loads(result.stdout)
    assert not reply["ok"] and not (tmp_path / "file").exists()
