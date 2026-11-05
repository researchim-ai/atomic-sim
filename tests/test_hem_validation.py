"""Independent limited HEM friction evidence and fail-closed audit tests."""

import json
import math
import shutil
import subprocess
import sys

import pytest

from atomic_model_hem_validation import audit
from atomic_model_hem_validation.__main__ import main
from atomic_model_validation import digitize


def test_pdf_reextracts_frozen_digitization():
    assert digitize.extract(audit.DATA / "source.pdf") == audit.read(audit.DATA / "vectors.json")


def test_reference_metrics_and_limits():
    result = audit.evaluate()
    assert result["samples"] == 38
    assert result["passed"]
    baseline = result["comparisons"]["independent_haaland"]
    assert baseline["max_abs_relative_error"] == pytest.approx(0.025613155401812585, abs=1e-14)
    errors = [r["relative_error"] for r in baseline["rows"]]
    assert baseline["rms_relative_error"] == math.sqrt(sum(e * e for e in errors) / 38)
    assert sum(s["samples"] for s in baseline["series"].values()) == 38
    for key in (
        "validated",
        "dynamic_hem_validated",
        "two_phase_hem_validated",
        "experimental_uncertainty_complete",
        "dimensional_pressure_drop_experiment",
        "hem_implementation_tested",
        "calibrated",
        "plant_referenced",
    ):
        assert result[key] is False
    blocked = result["validation_matrix"][1:]
    assert {r["regime"] for r in blocked} == {"evaporation", "condensation", "flashing", "blowdown"}
    for row in blocked:
        if row["regime"] in {"flashing", "blowdown"}:
            assert row["status"] == "blocked_missing_matched_dynamic_data"
            assert row["candidate_id"] == "alamgir-kan-lienhard-1980-figure4"
            assert not row["candidate_metadata"]["experimental_acceptance"]
        else:
            assert row["status"] == "blocked_missing_real_data"
            assert row["evidence"] is None
    for row in baseline["rows"]:
        lo, hi = row["modeled_reading_interval"]
        assert lo <= row["modeled_darcy_f"] <= hi


def test_signed_dimensional_reference_only():
    # Arbitrary manufactured values check algebra/units, never experimental rows.
    assert audit.pressure_drop(10, 0.02, 1000, 2, 0.02) == 20000
    assert audit.pressure_drop(10, 0.02, 1000, -2, 0.02) == -20000
    assert audit.pressure_drop(10, 0.02, 1000, 0, 0.02) == 0
    with pytest.raises(ValueError):
        audit.pressure_drop(1, 0, 1000, 1, 0.02)
    for re in (0, 2000, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            audit.darcy_factor(re, 0)


@pytest.mark.parametrize("scale", [0.25, 4])
def test_factor_convention_error_fails(monkeypatch, scale):
    original = audit.darcy_factor
    monkeypatch.setattr(audit, "darcy_factor", lambda re, rr: scale * original(re, rr))
    result = audit.evaluate()
    baseline = result["comparisons"]["independent_haaland"]
    assert not baseline["passed"]
    assert result["passed"]  # Baseline cannot override production screening.
    assert all(not s["passed"] for s in baseline["series"].values())


@pytest.mark.parametrize(
    "name",
    [
        "source.pdf",
        "vectors.json",
        "protocol.json",
        "measurements.csv",
        "NOTICE.md",
        "manifest.json",
    ],
)
def test_frozen_inputs_rejected(tmp_path, name):
    target = tmp_path / "data"
    shutil.copytree(audit.DATA, target)
    with (target / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        audit.dataset(target)


@pytest.fixture
def output(tmp_path):
    target = tmp_path / "audit"
    audit.run(target)
    return target


def test_audit_cli_and_no_overwrite(output):
    assert main(["verify", str(output)]) == 0
    with pytest.raises(FileExistsError):
        audit.run(output)
    assert audit.verify(output)["passed"]


@pytest.mark.parametrize("fault", ["report", "matrix", "source", "extra", "symlink"])
def test_tampering_even_with_rehashed_report(output, fault):
    if fault in {"report", "matrix"}:
        report = audit.read(output / "report.json")
        if fault == "report":
            report["comparisons"]["production_colebrook"]["rows"][0]["modeled_darcy_f"] *= 2
        else:
            report["validation_matrix"][1]["status"] = "validated"
        audit.write(output / "report.json", report)
        manifest = audit.read(output / "manifest.json")
        manifest["artifacts"]["report.json"] = audit.sha(output / "report.json")
        audit.write(output / "manifest.json", manifest)
    elif fault == "source":
        manifest = audit.read(output / "manifest.json")
        manifest["source"]["implementation"] = {}
        audit.write(output / "manifest.json", manifest)
    elif fault == "extra":
        (output / "unexpected").write_text("x")
    else:
        (output / "data/source.pdf").unlink()
        (output / "data/source.pdf").symlink_to(audit.DATA / "source.pdf")
    assert main(["verify", str(output)]) == 2


def test_changed_runtime_formula_rejected(output, monkeypatch):
    original = audit.darcy_factor
    monkeypatch.setattr(audit, "darcy_factor", lambda re, rr: original(re, rr) * 1.001)
    with pytest.raises(ValueError, match="recomputation"):
        audit.verify(output)


def test_production_coefficient_is_compared_on_every_point():
    result = audit.evaluate()
    production = result["comparisons"]["production_colebrook"]
    baseline = result["comparisons"]["independent_haaland"]
    protocol = result["protocol"]
    rr = protocol["roughness_m"] / protocol["diameter_m"]
    assert production["samples"] == 38
    assert production["passed"] and production["max_abs_relative_error"] < 0.05
    assert result["production_coefficient_tested"] is True
    assert result["primary_comparison"] == "production_colebrook"
    for actual, reference in zip(production["rows"], baseline["rows"]):
        assert actual["modeled_darcy_f"] == audit.production_closures.darcy_factor(
            actual["reynolds"], rr
        )
        assert actual["modeled_darcy_f"] != reference["modeled_darcy_f"]
        assert actual["darcy_f"] == reference["darcy_f"]
    assert sum(s["samples"] for s in production["series"].values()) == 38


@pytest.mark.parametrize("scale", [0.25, 4])
def test_production_failure_not_hidden_by_baseline(monkeypatch, scale):
    original = audit.production_closures.darcy_factor
    monkeypatch.setattr(
        audit.production_closures, "darcy_factor", lambda re, rr: scale * original(re, rr)
    )
    result = audit.evaluate()
    assert not result["passed"]
    assert not result["comparisons"]["production_colebrook"]["passed"]
    assert result["comparisons"]["independent_haaland"]["passed"]


def test_runtime_production_change_rejected(output, monkeypatch):
    original = audit.production_closures.darcy_factor
    monkeypatch.setattr(
        audit.production_closures, "darcy_factor", lambda re, rr: original(re, rr) * 1.001
    )
    with pytest.raises(ValueError, match="recomputation"):
        audit.verify(output)


def test_closure_source_identity_change_rejected(output, monkeypatch):
    actual = audit.closure_sources()
    monkeypatch.setattr(audit, "closure_sources", lambda: {**actual, "__init__.py": "0" * 64})
    with pytest.raises(ValueError, match="provenance"):
        audit.verify(output)


def test_rehashed_closure_snapshot_rejected(output):
    snapshot = output / "closure_source/__init__.py"
    snapshot.write_text(snapshot.read_text() + "\n# tampered\n")
    manifest = audit.read(output / "manifest.json")
    manifest["artifacts"]["closure_source/__init__.py"] = audit.sha(snapshot)
    audit.write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="closure source"):
        audit.verify(output)


def test_restored_audit_checks_current_closure_and_recomputes(output, tmp_path):
    restored = tmp_path / "restored"
    shutil.copytree(output, restored)
    result = audit.verify(restored)
    assert result["source"]["production_closure_sources"] == audit.closure_sources()
    assert result["comparisons"]["production_colebrook"]["samples"] == 38
    assert result["comparisons"]["independent_haaland"]["samples"] == 38


def test_cli_returns_failure_for_production_screening(tmp_path, monkeypatch):
    original = audit.production_closures.darcy_factor
    monkeypatch.setattr(
        audit.production_closures, "darcy_factor", lambda re, rr: 4 * original(re, rr)
    )
    output = tmp_path / "failed"
    assert main(["run", str(output)]) == 2
    assert main(["verify", str(output)]) == 2
    assert audit.read(output / "report.json")["comparisons"]["independent_haaland"]["passed"]


def test_candidate_metadata_has_no_runtime_workspace_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Matrix must not read files")

    monkeypatch.setattr(audit.Path, "read_text", forbidden)
    monkeypatch.setattr(audit.Path, "read_bytes", forbidden)
    rows = {r["regime"]: r for r in audit.matrix()}
    for regime in ("flashing", "blowdown"):
        row = rows[regime]
        assert row["primary_url"] == (
            "https://engines.egr.uh.edu/sites/engines/files/talks/"
            "ExperimentalStudyOfDepressurization.pdf"
        )
        candidate = row["candidate_metadata"]
        assert candidate["runs"] == ["HW-18", "HW-17", "HW-20", "HW-27"]
        assert candidate["year"] == 1980
        assert not candidate["digitized_history_available"]
        assert not candidate["matched_dynamic_data_available"]
        assert not candidate["comparison_performed"]
        assert not candidate["experimental_acceptance"]
        assert not candidate["uncertainty_complete"]


def test_embedded_matrix_is_portable_outside_checkout(tmp_path):
    # The .py metadata module is shipped automatically with the package in wheels.
    # Exercise it alone in an isolated interpreter, without checkout/import paths.
    from atomic_model_hem_validation import evidence

    module = tmp_path / "evidence.py"
    shutil.copyfile(evidence.__file__, module)
    code = f"import json,runpy; print(json.dumps(runpy.run_path({str(module)!r})['matrix']()))"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == audit.matrix()


def test_forged_candidate_acceptance_rejected_after_rehash(output):
    report = audit.read(output / "report.json")
    row = next(r for r in report["validation_matrix"] if r["regime"] == "flashing")
    row["candidate_metadata"]["experimental_acceptance"] = True
    audit.write(output / "report.json", report)
    manifest = audit.read(output / "manifest.json")
    manifest["artifacts"]["report.json"] = audit.sha(output / "report.json")
    audit.write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="recomputation"):
        audit.verify(output)
