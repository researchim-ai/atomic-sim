"""Qualification-only old traversal, identical source digest and native state."""

from pathlib import Path

import pytest

from atomic_model_workspace.common import read
from scripts.qualify_source_scan import legacy_fingerprint, run
from sdk.experiment import _source_fingerprint


def test_qualification_reference_matches_current_source_bytes():
    assert legacy_fingerprint() == _source_fingerprint()


def test_same_tree_old_new_worker_comparison_is_exact(tmp_path):
    value = read(Path(__file__).resolve().parents[1] / "examples/eos_performance.json")
    value.update(cases=value["cases"][:2], workers=[1], repeats=1, cpu_count=1)
    report = run(value, tmp_path / "qualification")
    assert report["passed"] and report["numerical_identity_exact"]
    assert report["microbenchmark"]["speedup"] > 0  # No hardware-dependent speed gate.
    assert len(report["trials"]) == 2 and all(t["completed"] == 2 for t in report["trials"])
    assert [t["strategy"] for t in report["trials"]] == ["legacy", "optimized"]
    assert report["summaries"][0]["eligible"]
    assert not report["validated"] and not report["plant_referenced"]
    with pytest.raises(FileExistsError):
        run(value, tmp_path / "qualification")


def test_invalid_case_is_rejected_before_qualification_output(tmp_path):
    value = read(Path(__file__).resolve().parents[1] / "examples/eos_performance.json")
    value["cases"][-1]["heat_power_w"]["wall.w0"] = 10001
    with pytest.raises(ValueError):
        run(value, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
