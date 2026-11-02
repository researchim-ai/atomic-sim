"""Actual integration meshes and diagnostic rollback, not just requested policies."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.network import NetworkError
from core.solver_trace import summarize_solver_trace
from sdk.steam_cycle_experiment import run_steam_cycle_experiment
from sdk.turbine_experiment import run_turbine_experiment


def trace():
    return dict(
        schema_version=1,
        backend="scipy.solve_ivp",
        method="Radau",
        start_time_s=0.0,
        end_time_s=0.2,
        max_step_s=0.1,
        rtol=1e-9,
        atol_scale=0.01,
        scipy_nfev=25,
        scipy_njev=1,
        scipy_nlu=4,
        mesh_times_s=[0.0, 0.1, 0.2],
    )


def test_mesh_counts_and_exact_temporal_coverage():
    result = summarize_solver_trace([trace()], start_time_s=0, end_time_s=0.2)
    assert result["scipy_segments"] == 1
    assert result["reported_mesh_steps"] == 2
    assert result["max_reported_step_s"] == 0.1
    assert result["scipy_nfev"] == 25


@pytest.mark.parametrize(
    "defect",
    ["gap", "cap", "unordered", "endpoint", "counter", "nonfinite", "schema", "empty", "tolerance"],
)
def test_corrupt_trace_rejected(defect):
    item = trace()
    if defect == "gap":
        item["start_time_s"] = 0.01
        item["mesh_times_s"][0] = 0.01
    elif defect == "cap":
        item["max_step_s"] = 0.05
    elif defect == "unordered":
        item["mesh_times_s"] = [0.0, 0.15, 0.1, 0.2]
    elif defect == "endpoint":
        item["mesh_times_s"][-1] = 0.19
    elif defect == "counter":
        item["scipy_nfev"] = True
    elif defect == "nonfinite":
        item["mesh_times_s"][1] = float("nan")
    elif defect == "schema":
        item["schema_version"] = True
    elif defect == "empty":
        item["mesh_times_s"] = []
    else:
        item["rtol"] = 1
    with pytest.raises(NetworkError):
        summarize_solver_trace([item], start_time_s=0, end_time_s=0.2)


def test_boundary_hold_reports_no_solver_work():
    item = dict(
        trace(),
        backend="boundary_hold",
        end_time_s=0,
        mesh_times_s=[0.0],
        scipy_nfev=0,
        scipy_njev=0,
        scipy_nlu=0,
    )
    result = summarize_solver_trace([item], start_time_s=0, end_time_s=0)
    assert result["scipy_segments"] == result["reported_mesh_steps"] == result["scipy_nfev"] == 0
    item["scipy_nfev"] = 1
    with pytest.raises(NetworkError):
        summarize_solver_trace([item], start_time_s=0, end_time_s=0)


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_sdk_records_actual_mesh_across_control_breaks(tmp_path, method):
    spec = json.loads(Path("scenarios/ai05_dynamic_steam_cycle.json").read_text())
    spec.update(
        duration_s=0.3,
        step_s=0.3,
        timeline=[dict(t=0.13, controls=dict(spec["controls"], heater_power_w=280000))],
    )
    spec["solver"].update(method=method, max_step_s=0.025)
    report = run_steam_cycle_experiment(spec, tmp_path / "run")
    assert report["numerical_acceptance"]
    records = json.loads((tmp_path / "run/solver_trace.json").read_text())
    assert len(records) == 2
    assert [r["end_time_s"] for r in records] == [0.13, 0.3]
    assert all(r["method"] == method for r in records)
    fresh = summarize_solver_trace(records, start_time_s=0, end_time_s=0.3)
    assert fresh == report["solver_statistics"]
    assert fresh["max_reported_step_s"] <= 0.025 + 1e-12
    assert fresh["reported_mesh_steps"] >= 12
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert "solver_trace.json" in manifest["artifacts"]


def test_exact_kernel_does_not_invent_scipy_statistics(tmp_path):
    spec = json.loads(Path("scenarios/ai05_turbine_load_rejection.json").read_text())
    spec.update(duration_s=0.2, timeline=[])
    report = run_turbine_experiment(spec, tmp_path / "exact")
    assert report["numerical_acceptance"]
    assert report["solver_statistics"]["available"] is False
    assert report["solver_statistics"]["scipy_nfev"] == 0


def test_summary_rejects_incomplete_or_overlapping_segments():
    with pytest.raises(NetworkError):
        summarize_solver_trace([trace()], start_time_s=0, end_time_s=1)
    second = deepcopy(trace())
    with pytest.raises(NetworkError):
        summarize_solver_trace([trace(), second], start_time_s=0, end_time_s=0.2)
