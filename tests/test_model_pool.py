"""Exact fresh-episode semantics, worker reuse and fault isolation."""

import hashlib
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from atomic_model_pool.identity import identity
from atomic_model_pool.runner import compare, cpu_budget, pooled_trial, prepare_pool
from atomic_model_workspace.common import read


@pytest.fixture
def spec():
    value = read(Path(__file__).resolve().parents[1] / "examples/eos_pool_comparison.json")
    p = value["performance"]
    p.update(cases=p["cases"][:3], workers=[1], repeats=1, cpu_count=1)
    # Revisit original conditions after an intervening different preset/action.
    p["cases"][2] = deepcopy(p["cases"][0])
    p["cases"][2]["id"] = "repeat-original"
    return value


@pytest.mark.parametrize(
    "defect",
    ["schema", "unknown", "cap_bool", "cap_zero", "cap_large", "base_unknown", "action", "seed"],
)
def test_pool_preflight_precedes_output(spec, tmp_path, defect):
    if defect == "schema":
        spec["schema_version"] = True
    elif defect == "unknown":
        spec["module"] = "user.code"
    elif defect == "cap_bool":
        spec["max_cases_per_worker"] = True
    elif defect == "cap_zero":
        spec["max_cases_per_worker"] = 0
    elif defect == "cap_large":
        spec["max_cases_per_worker"] = 65
    elif defect == "base_unknown":
        spec["performance"]["module"] = "user.code"
    elif defect == "action":
        spec["performance"]["cases"][1]["heat_power_w"]["wall.w0"] = float("nan")
    elif defect == "seed":
        spec["performance"]["cases"][1]["seed"] = -1
    with pytest.raises(ValueError):
        compare(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_reuse_is_fresh_native_episodes_and_archive_hashes(spec, tmp_path):
    result = compare(spec, tmp_path / "run")
    assert result["passed"] and result["numerical_identity_exact"]
    a, b = result["trials"]
    assert a["mode"] == "classic" and b["mode"] == "pool"
    assert b["worker_processes_spawned"] == 1
    assert [r["task_ordinal"] for r in b["rows"]] == [1, 2, 3]
    assert len(set(r["pid"] for r in b["rows"])) == 1
    hashes = {r["case_id"]: r["result"]["episode_sha256"] for r in b["rows"]}
    assert hashes["case-0"] == hashes["repeat-original"]
    assert b["rows"][0]["result"]["episode_sha256"] == a["rows"][0]["result"]["episode_sha256"]
    manifest = read(tmp_path / "run/manifest.json")
    for path, expected in manifest["artifacts"].items():
        assert hashlib.sha256((tmp_path / "run" / path).read_bytes()).hexdigest() == expected
    assert result["summaries"][0]["eligible"]
    assert not result["validated"] and not result["plant_referenced"]
    with pytest.raises(FileExistsError):
        compare(spec, tmp_path / "run")


def test_task_cap_recycles_worker_and_keeps_exact_physics(spec, tmp_path):
    spec["max_cases_per_worker"] = 1
    report = compare(spec, tmp_path / "run")
    assert report["passed"]
    trial = next(t for t in report["trials"] if t["mode"] == "pool")
    assert trial["worker_processes_spawned"] == 3 and all(
        r["task_ordinal"] == 1 for r in trial["rows"]
    )


def test_deadline_replaces_worker_continues_queue_and_disables_speedup(spec, tmp_path):
    spec["performance"]["worker_timeout_s"] = 0.01
    report = compare(spec, tmp_path / "run")
    assert not report["passed"] and not report["two_x_vs_classic_same_workers"]
    pooled = next(t for t in report["trials"] if t["mode"] == "pool")
    assert pooled["worker_processes_spawned"] == 3 and len(pooled["rows"]) == 3
    assert all(r["status"] == "deadline" for r in pooled["rows"])
    assert report["summaries"][0]["pool_speedup_at_same_worker_count"] is None


def test_crashed_worker_does_not_block_remaining_tasks(spec, tmp_path, monkeypatch):
    import atomic_model_pool.runner as module

    def crash(folder, index, env):
        log = (folder / f"crash-{index}.log").open("w")
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys;sys.stdin.buffer.readline();sys.exit(17)"],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return dict(process=process, log=log, tasks=0, busy=None, index=index)

    monkeypatch.setattr(module, "_start", crash)
    spec = prepare_pool(spec)
    with cpu_budget(1) as cpus:
        r = pooled_trial(spec, tmp_path / "pool", 1, cpus, identity(), 8)
    assert len(r["rows"]) == 3 and all(x["status"] == "worker_exit" for x in r["rows"])
    assert not r["all_completed"] and r["completed"] == 0


def test_rss_limit_replaces_worker_after_each_failed_task(spec, tmp_path, monkeypatch):
    import atomic_model_pool.runner as module

    monkeypatch.setattr(module, "rss_bytes", lambda pid: 2 * 1024**3)
    spec = prepare_pool(spec)
    with cpu_budget(1) as cpus:
        r = pooled_trial(spec, tmp_path / "pool", 1, cpus, identity(), 8)
    assert len(r["rows"]) == 3 and all(x["status"] == "rss_limit" for x in r["rows"])
    assert not r["all_completed"]


def test_cpu_affinity_restored_on_exception():
    previous = os.sched_getaffinity(0)
    with pytest.raises(RuntimeError):
        with cpu_budget(1):
            assert len(os.sched_getaffinity(0)) == 1
            raise RuntimeError("injected")
    assert os.sched_getaffinity(0) == previous


def test_mode_order_alternates_and_parallel_results_are_exact(spec, tmp_path):
    spec["performance"].update(workers=[1, 2], repeats=2)
    r = compare(spec, tmp_path / "run")
    assert r["passed"] and r["numerical_identity_exact"]
    assert [(t["repeat"], t["workers"], t["mode"]) for t in r["trials"]] == [
        (0, 1, "classic"),
        (0, 1, "pool"),
        (0, 2, "classic"),
        (0, 2, "pool"),
        (1, 1, "pool"),
        (1, 1, "classic"),
        (1, 2, "pool"),
        (1, 2, "classic"),
    ]


def test_corrupt_completion_discarded_and_worker_replaced(spec, tmp_path, monkeypatch):
    import atomic_model_pool.runner as module

    original = module.read

    def corrupt(path):
        value = original(path)
        if str(path).endswith(".result.json"):
            value["token"] = "wrong"
        return value

    monkeypatch.setattr(module, "read", corrupt)
    spec = prepare_pool(spec)
    with cpu_budget(1) as cpus:
        r = pooled_trial(spec, tmp_path / "pool", 1, cpus, identity(), 8)
    assert r["completed"] == 0 and r["worker_processes_spawned"] == 3
    assert all(x["status"] == "invalid_result" for x in r["rows"])
    assert all("identity mismatch" in x["error"] for x in r["rows"])
