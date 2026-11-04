"""Real subprocess crashes and bounded fault injection for the durable scheduler."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from atomic_model_rollouts import controller, reports, store
from atomic_model_workspace.common import read

PROJECT = Path(__file__).resolve().parents[1] / "examples/eos_workspace/project.json"


def campaign(tmp_path):
    folder = tmp_path / "campaign"
    store.create(PROJECT, folder)
    return folder


@pytest.mark.parametrize("phase", ["before_go", "after_go"])
def test_controller_process_loss_before_or_after_go_is_recoverable(tmp_path, phase):
    folder = campaign(tmp_path)
    script = """
import os, sys
from atomic_model_rollouts.controller import Controller
import atomic_model_rollouts.controller as module
original_write = module.atomic_json
def write(path, value):
    if sys.argv[2] == "before_go" and path.name == "GO.json":
        os._exit(37)
    return original_write(path, value)
module.atomic_json = write
original = Controller._launch
def crash(self, *args):
    result = original(self, *args)
    os._exit(37)
Controller._launch = crash
Controller(sys.argv[1]).run()
"""
    process = subprocess.run([sys.executable, "-c", script, str(folder), phase], timeout=30)
    assert process.returncode == 37
    value = store.load(folder)
    assert value["payload"]["status"] == "running"
    running = [e for e in value["payload"]["entries"].values() if e["status"] == "running"]
    assert len(running) == 1
    pid = running[0]["attempts"][-1]["pid"]
    deadline = time.monotonic() + 10
    while (info := store.process_info(pid)) and info["state"] != "Z":
        assert time.monotonic() < deadline, "parent-death worker did not exit"
        time.sleep(0.01)
    with pytest.raises(ValueError):
        controller.recover(folder, expected_sha256="0" * 64)
    recovered = controller.recover(folder, expected_sha256=value["sha256"])
    assert recovered["payload"]["status"] == "paused"
    attempt = next(
        e["attempts"][-1] for e in recovered["payload"]["entries"].values() if e["attempts"]
    )
    assert attempt["outcome"] == "recovered_controller_loss"
    assert attempt["wall_s"] is None  # Lost timing must never be invented.
    final = controller.Controller(folder).run(resume=True, expected_sha256=recovered["sha256"])
    assert final["payload"]["status"] == "completed"
    assert reports.report(folder)["counts"]["completed"] == 6


def test_killed_worker_preserves_other_jobs_and_can_resume(tmp_path, monkeypatch):
    folder = campaign(tmp_path)
    original = controller.Controller._launch
    victim = []

    def launch(self, id, cpu):
        item = original(self, id, cpu)
        if not victim:
            victim.append(id)
            os.kill(item["process"].pid, signal.SIGKILL)
        return item

    monkeypatch.setattr(controller.Controller, "_launch", launch)
    final = controller.Controller(folder).run()
    entries = final["payload"]["entries"]
    assert entries[victim[0]]["status"] == "interrupted"
    assert sum(e["status"] == "completed" for e in entries.values()) == 5
    before = store.native(folder, victim[0])["episode"]["steps"]
    resumed = controller.Controller(folder).run(resume=True, expected_sha256=final["sha256"])
    assert resumed["payload"]["status"] == "completed"
    assert store.native(folder, victim[0])["episode"]["steps"] >= before


def test_observed_worker_rss_limit_isolated(tmp_path, monkeypatch):
    folder = campaign(tmp_path)
    original_launch = controller.Controller._launch
    original_rss = controller._rss
    victim = []

    def launch(self, id, cpu):
        item = original_launch(self, id, cpu)
        if not victim:
            victim.append((id, item["process"].pid))
        return item

    def rss(pid):
        return 8 * 1024**3 if victim and pid == victim[0][1] else original_rss(pid)

    monkeypatch.setattr(controller.Controller, "_launch", launch)
    monkeypatch.setattr(controller, "_rss", rss)
    result = controller.Controller(folder).run()
    entries = result["payload"]["entries"]
    assert entries[victim[0][0]]["reason"] == "worker_rss_limit"
    assert sum(e["status"] == "completed" for e in entries.values()) == 5
    assert read(folder / "HEAD.json")["sha256"] == result["sha256"]


@pytest.mark.parametrize("fault", ["output_limit", "invalid_checkpoint"])
def test_faulted_job_does_not_destroy_campaign(tmp_path, monkeypatch, fault):
    folder = campaign(tmp_path)
    original = controller.Controller._launch
    victim = []

    def launch(self, id, cpu):
        if not victim and fault == "invalid_checkpoint":
            victim.append(id)
            (folder / "jobs" / id / "HEAD.json").write_text("{}")
        item = original(self, id, cpu)
        if not victim and fault == "output_limit":
            victim.append(id)
            with (item["work"] / "worker.log").open("ab") as stream:
                stream.write(b"x" * (store.default_options()["log_limit_bytes"] + 1))
        return item

    monkeypatch.setattr(controller.Controller, "_launch", launch)
    result = controller.Controller(folder).run()
    entry = result["payload"]["entries"][victim[0]]
    assert entry["status"] == ("failed" if fault == "invalid_checkpoint" else "interrupted")
    assert sum(e["status"] == "completed" for e in result["payload"]["entries"].values()) == 5
    if fault == "output_limit":
        assert entry["reason"] == "worker_output_limit"
        assert entry["attempts"][-1]["log_truncated"]
    else:
        summary = reports.report(folder)
        assert summary["counts"]["native_unavailable"] == 1
        assert summary["counts"]["completed"] == 5
