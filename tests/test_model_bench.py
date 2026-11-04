"""Measured process isolation, fixed-work accounting and unchanged native results."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from atomic_model_bench.harness import benchmark, percentile
from atomic_model_bench.spec import prepare
from atomic_model_bench.worker import profile_group
from atomic_model_workspace.common import read


@pytest.fixture
def spec():
    value = read(Path(__file__).resolve().parents[1] / "examples/eos_performance.json")
    value.update(cases=value["cases"][:2], workers=[1, 2], repeats=1, cpu_count=1)
    return value


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "unknown",
        "workers",
        "duplicate_workers",
        "baseline",
        "repeats",
        "bool",
        "cpu",
        "deadline",
        "memory",
        "action",
        "config",
        "duplicate_case",
        "seed",
    ],
)
def test_bad_workload_admitted_before_output(spec, tmp_path, defect):
    if defect == "schema":
        spec["schema_version"] = True
    elif defect == "unknown":
        spec["secret"] = 1
    elif defect == "workers":
        spec["workers"] = [1, 3]
    elif defect == "duplicate_workers":
        spec["workers"] = [1, 1]
    elif defect == "baseline":
        spec["workers"] = [2]
    elif defect == "repeats":
        spec["repeats"] = 0
    elif defect == "bool":
        spec["workers"] = [True]
    elif defect == "cpu":
        spec["cpu_count"] = 100000
    elif defect == "deadline":
        spec["worker_timeout_s"] = float("nan")
    elif defect == "memory":
        spec["worker_rss_limit_mib"] = 1
    elif defect == "action":
        spec["cases"][1]["heat_power_w"]["wall.w0"] = 10001
    elif defect == "config":
        spec["cases"][1]["config"]["max_step_s"] = 0.5
    elif defect == "duplicate_case":
        spec["cases"][1]["id"] = spec["cases"][0]["id"]
    elif defect == "seed":
        spec["cases"][1]["seed"] = True
    with pytest.raises(ValueError):
        benchmark(spec, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_percentiles_are_interpolated_not_invented_tail_samples():
    assert percentile([], 0.95) is None
    assert percentile([3], 0.95) == 3
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)


def test_profile_groups_use_explicit_paths_and_exclusive_times():
    assert profile_group("scripts/compressible_storage_graph.py", "rho_T") == "synthetic_eos"
    assert profile_group("scipy/integrate/_ivp/common.py", "num_jac") == "scipy_jacobian"
    assert profile_group("scipy/integrate/_ivp/radau.py", "step") == "scipy_integrator_other"
    assert (
        profile_group("scripts/open_eos_channel.py", "evaluate") == "native_equations_and_admission"
    )
    assert profile_group("other.py", "rho_T") == "other"


def test_real_kernel_fixed_cpu_scaling_exact_replay_profile_and_hashes(spec, tmp_path):
    report = benchmark(spec, tmp_path / "bench")
    assert report["passed"] and report["numerical_identity_exact"]
    assert all(t["shared_cpu_set"] == report["fixed_cpu_set"] for t in report["trials"])
    assert len(report["fixed_cpu_set"]) == 1
    assert all(t["completed"] == 2 and t["all_completed"] for t in report["trials"])
    assert all(t["simulated_s"] == 0.04 for t in report["trials"])
    assert report["summaries"][0]["speedup_vs_one"] == 1
    assert (
        report["summaries"][1]["efficiency_speedup_per_worker"]
        == report["summaries"][1]["speedup_vs_one"] / 2
    )
    trial = report["profile"]
    row = trial["rows"][0]
    result = read(tmp_path / "bench/profile" / row["result"]["result_path"])
    assert result["profile"] and result["profiled"]
    assert sum(result["profile_exclusive_self_s_by_group"].values()) == pytest.approx(
        sum(r["self_s"] for r in result["profile"])
    )
    assert result["peak_rss_bytes"] > 0 and result["cpu_user_s_before_result_write"] > 0
    assert all(v == "1" for v in result["threads"].values())
    assert result["affinity_cpus"] == report["fixed_cpu_set"]
    manifest = read(tmp_path / "bench/manifest.json")
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((tmp_path / "bench" / name).read_bytes()).hexdigest() == expected
    assert not report["validated"] and not report["plant_referenced"]
    with pytest.raises(FileExistsError):
        benchmark(spec, tmp_path / "bench")


def test_real_deadlines_are_failures_not_fast_episodes(spec, tmp_path):
    spec.update(worker_timeout_s=0.01, workers=[1])
    report = benchmark(spec, tmp_path / "timeout", profile=False)
    assert not report["passed"] and not report["two_x_goal_met"]
    assert all(r["status"] == "deadline" for t in report["trials"] for r in t["rows"])
    assert report["summaries"][0]["speedup_vs_one"] is None
    assert report["trials"][0]["completed"] == 0 and report["trials"][0]["episodes_per_second"] == 0


def test_nonlinux_backend_rejected_before_output(spec, tmp_path, monkeypatch):
    import atomic_model_bench.spec as module

    monkeypatch.setattr(module.platform, "system", lambda: "Unsupported")
    with pytest.raises(ValueError, match="Linux"):
        benchmark(spec, tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


def test_insufficient_case_count_rejects_idle_worker_comparison(spec):
    spec = deepcopy(spec)
    spec["workers"] = [1, 4]
    with pytest.raises(ValueError, match="at least"):
        prepare(spec)


def test_affinity_restored_even_on_output_failure(spec, tmp_path):
    import os

    previous = os.sched_getaffinity(0)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        benchmark(spec, tmp_path / "exists")
    assert os.sched_getaffinity(0) == previous


def test_rss_supervisor_stops_workers_and_keeps_other_case_results(spec, tmp_path, monkeypatch):
    import atomic_model_bench.harness as module

    monkeypatch.setattr(module, "rss_bytes", lambda pid: 2 * 1024**3)
    spec["workers"] = [1]
    result = benchmark(spec, tmp_path / "rss", profile=False)
    assert not result["passed"] and not result["two_x_goal_met"]
    assert len(result["trials"][0]["rows"]) == 2
    assert all(r["status"] == "rss_limit" for r in result["trials"][0]["rows"])
    assert result["summaries"][0]["speedup_vs_one"] is None
