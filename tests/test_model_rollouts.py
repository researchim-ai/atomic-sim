"""Durable native rollout admission, lifecycle and transport regression checks."""

import copy
import shutil
from pathlib import Path

import pytest

from atomic_model_rollouts import reports, store
from atomic_model_rollouts.controller import Controller, _clear_lock, recover
from atomic_model_workspace import jobs
from atomic_model_workspace.common import atomic_json

PROJECT = Path(__file__).resolve().parents[1] / "examples/eos_workspace/project.json"


def create(tmp_path, **changes):
    cfg = store.default_options()
    cfg.update(changes)
    folder = tmp_path / "campaign"
    store.create(PROJECT, folder, cfg)
    return folder


@pytest.mark.parametrize(
    "change",
    [
        {"workers": 0},
        {"workers": True},
        {"seeds": [1, 1]},
        {"seeds": [True]},
        {"seeds": [-1]},
        {"seeds": []},
        {"decision_slice": 0},
        {"max_attempts": 9},
        {"job_wall_s": float("nan")},
        {"log_limit_bytes": 1},
        {"worker_rss_bytes": 0},
        {"total_rss_bytes": 1},
    ],
)
def test_invalid_admission_is_side_effect_free(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        create(tmp_path, **change)
    assert not (tmp_path / "campaign").exists()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = create(tmp_path_factory.mktemp("rollouts"), decision_slice=2)
    value = Controller(folder).run()
    assert value["payload"]["status"] == "completed"
    return folder


def test_completed_native_control_and_sliced_attempts(completed, tmp_path):
    value = store.load(completed)
    p = value["payload"]
    for id, item in store.expected(p["project"], p["config"]).items():
        target = tmp_path / id
        jobs.create_job(item["spec"], target)
        control = jobs.run_job(target)
        assert store.native(completed, id)["episode"] == control["episode"]
        attempts = p["entries"][id]["attempts"]
        assert len(attempts) >= 2
        assert all(a["outcome"] == "exited" for a in attempts)
        assert attempts[-1]["after_decisions"] == control["episode"]["steps"]
    assert not (completed / "CONTROLLER.json").exists()


def test_deadline_resume_requires_cas(tmp_path):
    folder = create(tmp_path)
    manager = Controller(folder)
    paused = manager.run(wall_s=0.05)
    assert paused["payload"]["status"] == "paused"
    with pytest.raises((ValueError, RuntimeError)):
        manager.run(resume=True, expected_sha256="0" * 64)
    final = manager.run(resume=True, expected_sha256=paused["sha256"])
    assert final["payload"]["status"] == "completed"


@pytest.mark.parametrize("kind", ["pause", "cancel"])
def test_async_control_and_resume(tmp_path, kind):
    folder = create(tmp_path)
    manager = Controller(folder)
    manager.start()
    getattr(manager, kind)()
    stopped = manager.wait()
    assert stopped["payload"]["status"] == ("paused" if kind == "pause" else "cancelled")
    assert not (folder / "CONTROLLER.json").exists()
    result = manager.run(resume=True, expected_sha256=stopped["sha256"])
    assert result["payload"]["status"] == "completed"


def test_live_recovery_refused(tmp_path):
    folder = create(tmp_path)
    manager = Controller(folder)
    current = manager.start()
    try:
        with pytest.raises(ValueError):
            recover(folder, expected_sha256=current["sha256"])
        with pytest.raises(ValueError):
            reports.export(folder, tmp_path / "export")
    finally:
        manager.close()


def test_unknown_lock_never_removed(tmp_path):
    lock = tmp_path / ".lock"
    lock.write_text("unknown")
    with pytest.raises(ValueError):
        _clear_lock(tmp_path, 2147483647)
    assert lock.read_text() == "unknown"


def test_completed_report_export_recompute_and_corruption(completed, tmp_path):
    output = tmp_path / "export"
    report = reports.export(completed, output)
    assert reports.verify(output) == reports.report(completed)
    assert reports.verify(output)["counts"]["completed"] == 6
    assert report is not None
    with (output / "report.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError):
        reports.verify(output)


def test_extra_export_file_refused(completed, tmp_path):
    output = tmp_path / "export"
    reports.export(completed, output)
    (output / "unexpected").write_text("x")
    with pytest.raises(ValueError):
        reports.verify(output)


def test_metadata_tamper_refused(completed, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(completed, folder)
    value = store.load(folder)
    modified = copy.deepcopy(value["payload"])
    modified["source"]["rollouts_sha256"] = "0" * 64
    store.publish(folder, modified)
    with pytest.raises(ValueError):
        store.load(folder)


def test_memory_admission_reduces_parallelism(tmp_path):
    cfg = store.default_options()
    folder = create(tmp_path, total_rss_bytes=cfg["worker_rss_bytes"])
    report = reports.report(folder)
    assert report["effective_workers"] == 1
    assert report["counts"]["completed"] == 0
    assert all(not e["complete"] for e in report["entries"].values())


def test_stale_control_token_ignored(tmp_path):
    folder = create(tmp_path)
    atomic_json(folder / "CANCEL.json", dict(schema_version=1, token="stale", kind="cancel"))
    assert Controller(folder).run()["payload"]["status"] == "completed"
