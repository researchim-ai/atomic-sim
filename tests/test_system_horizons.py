"""Full horizons and comparative diagnostics never accept a successful prefix."""

import csv
import json

import pytest

from scripts.qualify_system_horizons import (
    FILES,
    SCALES,
    common_time_difference,
    inspect_case,
    requests,
)
from sdk.experiment import _source_fingerprint


def row(time, **updates):
    return dict(time_s=time, **{key: scale * 0.5 for key, scale in SCALES.items()}, **updates)


def test_common_time_comparison_requires_entire_horizon():
    left = [row(0), row(30), row(60)]
    right = [row(0), row(15), row(30), row(45), row(60)]
    result = common_time_difference(left, right)
    assert result["common_times"] == 3 and result["numerical_refinement_passed"]
    with pytest.raises(ValueError):
        common_time_difference(left, right[:-1])
    bad = [row(0), row(30), row(60)]
    bad[-1]["fuel_temperature_k"] += 10
    assert common_time_difference(left, bad)["numerical_refinement_passed"] is False


def make_case(folder, *, complete=True):
    spec = requests()["scram_coarse"]
    t = 60 if complete else 20
    report = dict(
        status="completed" if complete else "failed",
        requested_duration_s=60.0,
        completed_time_s=t,
        numerical_acceptance=complete,
        balance={key: 0.0 for key in spec["budgets"]},
        budget_checks={key: True for key in spec["budgets"]},
        error=None if complete else {"message": "model domain"},
        events=[],
    )
    for name in FILES:
        (folder / name).write_text("{}")
    (folder / "input.json").write_text(json.dumps(spec))
    (folder / "manifest.json").write_text(json.dumps(dict(source_sha256=_source_fingerprint())))
    (folder / "report.json").write_text(json.dumps(report))
    with (folder / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", *SCALES])
        writer.writeheader()
        writer.writerows([row(0), row(t)])
    return spec, report


def test_complete_case_and_failed_prefix_remain_distinct(tmp_path):
    spec, _ = make_case(tmp_path, complete=False)
    record, _ = inspect_case(tmp_path, spec)
    assert record["full_horizon_completed"] is False
    assert record["numerical_acceptance"] is False
    assert record["requested_duration_s"] == 60 and record["completed_time_s"] == 20


@pytest.mark.parametrize(
    "change",
    ["prefix_promotion", "budget_promotion", "wrong_input", "clock", "extra_file", "wrong_source"],
)
def test_reject_inconsistent_evidence(tmp_path, change):
    spec, report = make_case(tmp_path)
    if change == "prefix_promotion":
        report["status"] = "failed"
    elif change == "budget_promotion":
        key = next(iter(spec["budgets"]))
        report["balance"][key] = 2 * spec["budgets"][key]
    elif change == "wrong_input":
        inp = dict(spec, duration_s=120)
        (tmp_path / "input.json").write_text(json.dumps(inp))
    elif change == "clock":
        report["completed_time_s"] = 30
    elif change == "extra_file":
        (tmp_path / "unknown.txt").write_text("extra")
    else:
        (tmp_path / "manifest.json").write_text(json.dumps(dict(source_sha256="bad")))
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        inspect_case(tmp_path, spec)
