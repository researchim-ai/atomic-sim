"""Independent analytic metrics, data refusal, binding and split leakage guards."""

import hashlib
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atomic_model_reference.bundle import check_catalog, import_reference, load_bundle
from atomic_model_reference.compare import compare_job
from atomic_model_reference.metrics import compare_curve, first_crossing, max_gap
from atomic_model_workspace.common import atomic_json, read
from atomic_model_workspace.jobs import create_job, load_job, run_job


@pytest.fixture
def fixture(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/eos_reference"
    shutil.copytree(source, tmp_path / "reference")
    return tmp_path / "reference/reference.json"


def change(path, fn):
    value = read(path)
    fn(value)
    atomic_json(path, value)


def job_spec(path):
    spec = read(path)
    return dict(
        schema_version=1,
        scenario=dict(
            schema_version=1,
            config=spec["problem"]["config"],
            seed=17,
            split="dev",
            objective=dict(sensor="wall.w0", target_k=300, tolerance_k=1),
        ),
        policy=spec["problem"]["policy"],
        decision_timeout_s=1,
    )


def test_exact_linear_integrals_crossing_and_peaks():
    m = compare_curve([0, 2], [-1, 1], [0, 1, 2], [0, 0, 0], [0, 2])
    assert m["mae"] == 0.5 and m["integral_absolute_error"] == 1 and m["bias"] == 0
    assert m["rmse"] == pytest.approx(math.sqrt(1 / 3)) and m["integral_error"] == 0
    assert (
        m["peak_absolute_error"] == 1
        and m["model_peak_time_s"] == 2
        and m["reference_peak_time_s"] == 0
    )
    assert m["fraction_time_within_declared_uncertainty"] is None
    assert first_crossing([0, 2], [-1, 1], 0, 2, 0, "up") == 1
    assert first_crossing([0, 2], [1, -1], 0, 2, 0, "down") == 1


def test_unequal_sampling_time_weighting_and_window():
    m = compare_curve([0, 0.1, 2], [0, 0.1, 2], [0, 1, 2], [0, 0, 0], [0.5, 1.5])
    assert m["mae"] == pytest.approx(1) and m["bias"] == pytest.approx(1)
    assert m["rmse"] == pytest.approx(math.sqrt(13 / 12)) and m["model_integral"] == pytest.approx(
        1
    )
    assert m["peak_absolute_error"] == pytest.approx(1.5)
    assert max_gap([0, 1, 10], 0, 0.5) == 1
    with pytest.raises(ValueError):
        compare_curve([0, 1], [0, 1], [0, 2], [0, 0], [0, 2])


def test_uncertainty_interval_fraction_is_not_sample_fraction():
    m = compare_curve([0, 2], [-1, 1], [0, 2], [0, 0], [0, 2], [0.5, 0.5])
    assert m["fraction_time_within_declared_uncertainty"] == pytest.approx(0.5)
    zero = compare_curve([0, 2], [0, 0], [0, 2], [0, 0], [0, 2], [0, 0])
    assert zero["fraction_time_within_declared_uncertainty"] == 1


def test_directional_plateau_and_initial_equality_are_explicit():
    assert first_crossing([0, 1, 2, 3], [-1, 0, 0, 1], 0, 3, 0, "up") == 1
    assert first_crossing([0, 1], [0, 1], 0, 1, 0, "up") is None
    assert first_crossing([0, 1], [1, 0], 0, 1, 0, "up") is None


def test_import_units_raw_bytes_normalization_and_real_job(fixture, tmp_path):
    data = import_reference(fixture, tmp_path / "bundle")
    assert data["times_s"] == [0, 0.01, 0.02, 0.03]
    assert data["values"]["wall_temperature"] == [300] * 4
    assert (
        data["uncertainty"]["wall_temperature"] == [0.01] * 4
    )  # Celsius offset does not apply to sigma.
    assert data["kind"] == "numerical_reference" and not data["validated"]
    assert load_bundle(tmp_path / "bundle")[0] == data
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    report = compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    assert not report["passed"] and report["channels"]["wall_temperature"]["metrics"]["rmse"] > 0.01
    rows = load_job(tmp_path / "job")["rows"]

    def temp(row):
        return next(c["value"] for c in row["observation"]["channels"] if c["name"] == "wall.w0")

    expected = (
        sum(
            (b["time_s"] - a["time_s"]) * ((temp(a) - 300) + (temp(b) - 300)) / 2
            for a, b in zip(rows, rows[1:])
        )
        / 0.03
    )
    assert report["channels"]["wall_temperature"]["metrics"]["mae"] == pytest.approx(expected)
    assert report["evidence_kind"] == "numerical_reference" and report["promotion"].startswith(
        "none"
    )
    for name, sha in read(tmp_path / "report/manifest.json")["artifacts"].items():
        assert hashlib.sha256((tmp_path / "report" / name).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "unknown",
        "role",
        "kind",
        "license",
        "sha",
        "path",
        "unit",
        "signal",
        "column",
        "criteria",
        "event",
        "gap",
        "window",
        "mask",
    ],
)
def test_invalid_reference_rejected_before_bundle_creation(fixture, tmp_path, defect):
    if defect == "schema":
        change(fixture, lambda p: p.update(schema_version=True))
    elif defect == "unknown":
        change(fixture, lambda p: p.update(extra=1))
    elif defect == "role":
        change(fixture, lambda p: p.update(role="unknown"))
    elif defect == "kind":
        change(fixture, lambda p: p.update(kind="validated"))
    elif defect == "license":
        change(fixture, lambda p: p["provenance"].update(license=""))
    elif defect == "sha":
        change(fixture, lambda p: p["csv"].update(sha256="0" * 64))
    elif defect == "path":
        change(fixture, lambda p: p["csv"].update(path="../constant.csv"))
    elif defect == "unit":
        change(fixture, lambda p: p["channels"]["wall_temperature"].update(unit="kg"))
    elif defect == "signal":
        change(
            fixture, lambda p: p["channels"]["wall_temperature"].update(signal="temperature:hidden")
        )
    elif defect == "column":
        change(fixture, lambda p: p["channels"]["wall_temperature"].update(column=None))
    elif defect == "criteria":
        change(fixture, lambda p: p["criteria"]["channels"].update(extra={}))
    elif defect == "event":
        change(
            fixture,
            lambda p: p["criteria"].update(
                events=[
                    dict(
                        id="e",
                        channel="missing",
                        threshold=300,
                        direction="up",
                        max_time_error_s=0.1,
                    )
                ]
            ),
        )
    elif defect == "gap":
        change(fixture, lambda p: p["criteria"].update(max_reference_gap_s=0.001))
    elif defect == "window":
        change(fixture, lambda p: p["criteria"].update(window_s=[0, 1]))
    elif defect == "mask":
        change(fixture, lambda p: p["problem"]["policy"]["heat_power_w"].pop("wall.w0"))
    with pytest.raises(ValueError):
        import_reference(fixture, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "csv",
    [
        "time_ms,wall_c,sigma_c\n0,26.85,.1\n0,26.85,.1\n",
        "time_ms,wall_c,sigma_c\n10,26.85,.1\n0,26.85,.1\n",
        "time_ms,wall_c,sigma_c\n0,nan,.1\n30,26.85,.1\n",
        "time_ms,wall_c,sigma_c\n0,26.85,-.1\n30,26.85,.1\n",
        "time_ms,wall_c,wall_c\n0,26.85,.1\n30,26.85,.1\n",
        "time_ms,wall_c,sigma_c\n0,26.85\n30,26.85,.1\n",
        "time_ms,wall_c,sigma_c\n0,-274,.1\n30,26.85,.1\n",
    ],
)
def test_malformed_numeric_csv_rejected_even_with_correct_hash(fixture, tmp_path, csv):
    data = fixture.parent / "constant.csv"
    data.write_text(csv)
    change(fixture, lambda p: p["csv"].update(sha256=hashlib.sha256(data.read_bytes()).hexdigest()))
    with pytest.raises(ValueError):
        import_reference(fixture, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_symlink_raw_reference_and_rehashed_bundle_tampering_rejected(fixture, tmp_path):
    import_reference(fixture, tmp_path / "bundle")
    path = tmp_path / "bundle/reference.json"
    value = read(path)
    value["values"]["wall_temperature"][0] += 1
    atomic_json(path, value)
    manifest = read(tmp_path / "bundle/manifest.json")
    manifest["artifacts"]["reference.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(tmp_path / "bundle/manifest.json", manifest)
    with pytest.raises(ValueError, match="disagrees"):
        load_bundle(tmp_path / "bundle")
    csv = fixture.parent / "constant.csv"
    csv.rename(fixture.parent / "original.csv")
    csv.symlink_to("original.csv")
    with pytest.raises(ValueError):
        import_reference(fixture, tmp_path / "linked")


def test_physical_case_splits_cannot_be_changed_by_crop_step_or_new_label(fixture, tmp_path):
    import_reference(fixture, tmp_path / "validation")
    change(
        fixture,
        lambda p: p.update(
            reference_id="relabeled-calibration", physical_case_id="invented-id", role="calibration"
        ),
    )
    change(
        fixture, lambda p: p["problem"]["config"].update(max_step_s=0.001, episode_horizon_s=0.04)
    )
    import_reference(fixture, tmp_path / "calibration")
    with pytest.raises(ValueError, match="leak"):
        check_catalog([tmp_path / "validation", tmp_path / "calibration"])
    change(fixture, lambda p: p.update(role="validation"))
    import_reference(fixture, tmp_path / "same-role")
    assert check_catalog([tmp_path / "validation", tmp_path / "same-role"])["accepted"]


def test_wrong_problem_rejected_and_partial_job_not_given_metrics(fixture, tmp_path):
    import_reference(fixture, tmp_path / "bundle")
    spec = job_spec(fixture)
    create_job(spec, tmp_path / "paused")
    run_job(tmp_path / "paused", max_decisions=1)
    report = compare_job(tmp_path / "paused", tmp_path / "bundle", tmp_path / "partial-report")
    assert (
        not report["passed"]
        and report["reason"] == "model_episode_incomplete"
        and report["channels"] == {}
    )
    spec["policy"]["heat_power_w"]["wall.w0"] = 1
    create_job(spec, tmp_path / "different")
    run_job(tmp_path / "different")
    with pytest.raises(ValueError, match="different"):
        compare_job(tmp_path / "different", tmp_path / "bundle", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_event_missing_and_failed_error_thresholds_are_reported(fixture, tmp_path):
    csv = fixture.parent / "constant.csv"
    csv.write_text(
        "time_ms,wall_c,sigma_c\n0,26.85,.0001\n10,26.849,.0001\n20,26.848,.0001\n30,26.847,.0001\n"
    )
    change(fixture, lambda p: p["csv"].update(sha256=hashlib.sha256(csv.read_bytes()).hexdigest()))
    change(
        fixture,
        lambda p: p["criteria"].update(
            events=[
                dict(
                    id="warming",
                    channel="wall_temperature",
                    threshold=299.9985,
                    direction="down",
                    max_time_error_s=0.001,
                )
            ]
        ),
    )
    import_reference(fixture, tmp_path / "bundle")
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    report = compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    assert not report["passed"]
    assert report["events"][0]["status"] == "model_event_missing" and report["events"][0][
        "reference_time_s"
    ] == pytest.approx(0.015)
    assert report["events"][0]["error_s"] is None
    assert report["channels"]["wall_temperature"]["metrics"]["mae"] > 0.0015


def test_cli_import_inspect_compare_and_catalog(fixture, tmp_path):
    import json

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))

    def cli(*args, expected=0):
        p = subprocess.run(
            [sys.executable, "-m", "atomic_model_reference", *map(str, args)],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == expected, p.stderr
        return json.loads(p.stdout)

    cli("import", fixture, "--output", tmp_path / "bundle")
    assert cli("inspect", tmp_path / "bundle")["samples"] == 4
    assert cli("catalog", tmp_path / "bundle")["accepted"]
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    assert not cli(
        "compare",
        tmp_path / "job",
        tmp_path / "bundle",
        "--output",
        tmp_path / "report",
        expected=2,
    )["passed"]


def test_energy_identity_is_positive_numerical_control_with_explicit_units(fixture, tmp_path):
    energy = fixture.parent / "energy_reference.json"
    ref = import_reference(energy, tmp_path / "bundle")
    assert (
        ref["values"]["energy_residual"] == [0] * 4
        and ref["channels"]["energy_residual"]["canonical_unit"] == "J"
    )
    create_job(job_spec(energy), tmp_path / "job")
    run_job(tmp_path / "job")
    report = compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    assert report["passed"] and not report["validated"]
    assert report["channels"]["energy_residual"]["metric_units"]["integral_error"] == "(J) s"
    assert (
        report["channels"]["energy_residual"]["metrics"][
            "fraction_time_within_declared_uncertainty"
        ]
        is None
    )
    assert (
        import_reference(tmp_path / "report/reference_input.json", tmp_path / "reimported") == ref
    )


@pytest.mark.parametrize("case", ["thermal_failure", "energy_success"])
def test_report_replay_preserves_both_pass_and_fail_metrics(fixture, tmp_path, case):
    from atomic_model_reference.replay import replay_report

    source = fixture if case == "thermal_failure" else fixture.parent / "energy_reference.json"
    import_reference(source, tmp_path / "bundle")
    create_job(job_spec(source), tmp_path / "job")
    run_job(tmp_path / "job")
    expected = compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    result = replay_report(tmp_path / "report", tmp_path / "replayed")
    assert result["replay_exact"] and result["comparison_passed"] == expected["passed"]
    assert read(tmp_path / "replayed/report.json") == expected


def test_rehashed_metric_tampering_fails_recomputation_before_output(fixture, tmp_path):
    from atomic_model_reference.replay import replay_report

    import_reference(fixture, tmp_path / "bundle")
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    path = tmp_path / "report/report.json"
    data = read(path)
    data["channels"]["wall_temperature"]["metrics"]["mae"] = 0
    atomic_json(path, data)
    manifest = read(tmp_path / "report/manifest.json")
    manifest["artifacts"]["report.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(tmp_path / "report/manifest.json", manifest)
    with pytest.raises(ValueError, match="Recomputed"):
        replay_report(tmp_path / "report", tmp_path / "forged")
    assert not (tmp_path / "forged").exists()


def test_reference_row_and_byte_limits_are_enforced(fixture, tmp_path, monkeypatch):
    import atomic_model_reference.data as module

    monkeypatch.setattr(module, "MAX_ROWS", 3)
    with pytest.raises(ValueError, match="row"):
        import_reference(fixture, tmp_path / "rows")
    assert not (tmp_path / "rows").exists()
    monkeypatch.setattr(module, "MAX_ROWS", 20000)
    monkeypatch.setattr(module, "MAX_CSV_BYTES", 8)
    with pytest.raises(ValueError, match="hash limit"):
        import_reference(fixture, tmp_path / "bytes")
    assert not (tmp_path / "bytes").exists()


def test_case_id_guard_and_genuinely_distinct_condition_roles(fixture, tmp_path):
    import_reference(fixture, tmp_path / "validation")
    change(fixture, lambda p: p.update(reference_id="other-reference", role="calibration"))
    change(fixture, lambda p: p["problem"]["policy"]["heat_power_w"].update({"wall.w0": 10}))
    import_reference(fixture, tmp_path / "same-id")
    with pytest.raises(ValueError, match="leak"):
        check_catalog([tmp_path / "validation", tmp_path / "same-id"])
    change(
        fixture,
        lambda p: p.update(reference_id="other-physical-case", physical_case_id="heated-command"),
    )
    import_reference(fixture, tmp_path / "different-case")
    with pytest.raises(ValueError, match="leakage"):
        check_catalog([tmp_path / "validation", tmp_path / "different-case"])
    raw = fixture.parent / "constant.csv"
    raw.write_text(
        "time_ms,wall_c,sigma_c\n0,26.95,.01\n10,26.95,.01\n20,26.95,.01\n30,26.95,.01\n"
    )
    change(fixture, lambda p: p["csv"].update(sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
    change(fixture, lambda p: p.update(reference_id="independent-numerical-curve"))
    import_reference(fixture, tmp_path / "new-curve")
    assert check_catalog([tmp_path / "validation", tmp_path / "new-curve"])["accepted"]


def test_completed_model_with_insufficient_output_resolution_is_not_compared(fixture, tmp_path):
    change(fixture, lambda p: p["criteria"].update(max_model_gap_s=0.001))
    import_reference(fixture, tmp_path / "bundle")
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    result = compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    assert (
        not result["passed"]
        and result["reason"] == "model_interpolation_gap"
        and result["channels"] == {}
    )


def test_report_replay_cannot_read_an_untracked_csv(fixture, tmp_path):
    from atomic_model_reference.replay import replay_report

    import_reference(fixture, tmp_path / "bundle")
    create_job(job_spec(fixture), tmp_path / "job")
    run_job(tmp_path / "job")
    compare_job(tmp_path / "job", tmp_path / "bundle", tmp_path / "report")
    path = tmp_path / "report/reference_input.json"
    data = read(path)
    data["csv"]["path"] = "other.csv"
    atomic_json(path, data)
    shutil.copy2(tmp_path / "report/data.csv", tmp_path / "report/other.csv")
    manifest = read(tmp_path / "report/manifest.json")
    manifest["artifacts"]["reference_input.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(tmp_path / "report/manifest.json", manifest)
    with pytest.raises(ValueError, match="inventory"):
        replay_report(tmp_path / "report", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_unit_and_uncertainty_changes_do_not_hide_same_curve_across_roles(fixture, tmp_path):
    import_reference(fixture, tmp_path / "original")
    csv = fixture.parent / "constant.csv"
    csv.write_text("time_ms,wall_c,sigma_c\n0,300,.02\n10,300,.02\n20,300,.02\n30,300,.02\n")
    change(
        fixture,
        lambda p: p.update(reference_id="renamed", physical_case_id="other", role="calibration"),
    )
    change(fixture, lambda p: p["problem"]["policy"]["heat_power_w"].update({"wall.w0": 10}))
    change(fixture, lambda p: p["csv"].update(sha256=hashlib.sha256(csv.read_bytes()).hexdigest()))
    change(fixture, lambda p: p["channels"]["wall_temperature"].update(unit="K"))
    import_reference(fixture, tmp_path / "same-curve")
    with pytest.raises(ValueError, match="curve leakage"):
        check_catalog([tmp_path / "original", tmp_path / "same-curve"])


def test_original_declaration_is_retained_and_only_csv_path_is_rewritten(fixture, tmp_path):
    import_reference(fixture, tmp_path / "bundle")
    original = read(tmp_path / "bundle/source_input.json")
    portable = read(tmp_path / "bundle/input.json")
    assert original == read(fixture) and original["csv"]["path"] == "constant.csv"
    original["csv"]["path"] = "data.csv"
    assert original == portable


def test_rehashed_origin_metadata_tampering_is_not_silently_accepted(fixture, tmp_path):
    import_reference(fixture, tmp_path / "bundle")
    path = tmp_path / "bundle/source_input.json"
    value = read(path)
    value["provenance"]["license"] = "invented"
    atomic_json(path, value)
    manifest = read(tmp_path / "bundle/manifest.json")
    manifest["artifacts"]["source_input.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(tmp_path / "bundle/manifest.json", manifest)
    with pytest.raises(ValueError, match="Original"):
        load_bundle(tmp_path / "bundle")
