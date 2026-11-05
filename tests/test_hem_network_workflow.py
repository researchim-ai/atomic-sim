"""On-disk graph program integrity and command-line entry point."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from atomic_model_hem_network import workflow
from atomic_model_hem_network.cases import baseline
from atomic_model_workspace.common import atomic_json


def spec():
    return dict(
        schema_version=1, model=baseline("loop"), program=[dict(duration_s=0.0002, command=None)]
    )


def test_saved_program_replays_and_does_not_overwrite(tmp_path):
    folder = tmp_path / "run"
    saved = workflow.run(spec(), folder)
    assert workflow.replay(folder) == saved
    with pytest.raises(FileExistsError):
        workflow.run(spec(), folder)


@pytest.mark.parametrize("damage", ["result", "inventory", "symlink", "source"])
def test_saved_program_tampering_rejected(tmp_path, damage):
    folder = tmp_path / "run"
    workflow.run(spec(), folder)
    if damage == "inventory":
        (folder / "extra").write_text("extra")
    elif damage == "symlink":
        (folder / "input.json").rename(tmp_path / "input.json")
        (folder / "input.json").symlink_to(tmp_path / "input.json")
    else:
        p = folder / "result.json"
        v = json.loads(p.read_text())
        if damage == "source":
            v["source"] = {}
        else:
            v["observations"][-1]["cells"][0]["mass_kg"] += 0.01
        atomic_json(p, v)
    with pytest.raises(ValueError):
        workflow.replay(folder)


def test_failed_program_does_not_publish_partial_artifact(tmp_path):
    request = spec()
    request["program"].append(dict(duration_s=-1, command=None))
    folder = tmp_path / "run"
    with pytest.raises(ValueError):
        workflow.run(request, folder)
    assert not folder.exists()


def test_cli_runs_from_arbitrary_cwd(tmp_path):
    source = tmp_path / "request.json"
    source.write_text(json.dumps(spec()))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    target = tmp_path / "run"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "atomic_model_hem_network",
            "run",
            str(source),
            "--output",
            str(target),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    p = subprocess.run(
        [sys.executable, "-m", "atomic_model_hem_network", "replay", str(target)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_audit_rejects_other_source_before_recomputing(tmp_path):
    file = tmp_path / "campaign.json"
    atomic_json(file, dict(source={}))
    with pytest.raises(ValueError, match="source"):
        workflow.audit(file)
