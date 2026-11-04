"""External figure provenance, independent re-extraction and fail-closed comparison."""

import copy
import hashlib
import shutil

import pytest

from atomic_model_validation import digitize, pipe
from atomic_model_validation.__main__ import main
from atomic_model_workspace.common import atomic_json, read


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    folder = tmp_path_factory.mktemp("nist") / "comparison"
    pipe.run(folder)
    return folder


def test_source_pdf_reextracts_all_symbols_and_axes():
    actual = digitize.extract(pipe.DATA / "source.pdf")
    expected = read(pipe.DATA / "vectors.json")
    assert actual == expected
    assert actual["counts"] == dict(test_1=14, test_2=9, test_3=9, test_4=6)
    assert actual["axes"]["re0"] == 0
    assert actual["axes"]["re1"] == 120000
    assert actual["axes"]["f0"] == 0.016
    assert actual["axes"]["f1"] == 0.030


def test_native_existing_closure_external_comparison(output):
    result = pipe.verify(output)
    assert result["samples"] == 38
    assert result["passed"] is True
    assert result["max_abs_relative_error"] < 0.05
    assert result["reynolds_range"][0] > 14000
    assert result["reynolds_range"][1] < 101000
    assert result["validated"] is False
    assert result["plant_referenced"] is False
    assert result["experimental_uncertainty_complete"] is False
    assert result["calibrated"] is False
    assert sum(r["samples"] for r in result["series"].values()) == 38
    for row in result["rows"]:
        assert row["reynolds_reading_bound"] > 0
        assert row["darcy_f_reading_bound"] > 0
        assert (
            row["modeled_reading_interval"][0]
            <= row["modeled_darcy_f"]
            <= row["modeled_reading_interval"][1]
        )


@pytest.mark.parametrize("multiplier", [0.25, 4.0])
def test_fanning_darcy_or_factor_error_cannot_pass(monkeypatch, multiplier):
    original = pipe.native_factor
    monkeypatch.setattr(
        pipe, "native_factor", lambda re, protocol: multiplier * original(re, protocol)
    )
    result = pipe.evaluate()
    assert result["passed"] is False
    assert all(not r["passed"] for r in result["series"].values())
    assert result["validated"] is False


def test_native_darcy_path_required(monkeypatch):
    original = pipe.closures.friction_darcy
    monkeypatch.setattr(pipe.closures, "friction_darcy", lambda *a: 4 * original(*a))
    with pytest.raises(ValueError, match="Darcy integration"):
        pipe.evaluate()


@pytest.mark.parametrize(
    "name",
    [
        "source.pdf",
        "protocol.json",
        "measurements.csv",
        "vectors.json",
        "NOTICE.md",
        "manifest.json",
    ],
)
def test_frozen_data_changes_rejected(tmp_path, name):
    target = tmp_path / "data"
    shutil.copytree(pipe.DATA, target)
    with (target / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        pipe.dataset(target)


def test_forged_report_rehashed_manifest_still_fails(output, tmp_path):
    target = tmp_path / "report"
    shutil.copytree(output, target)
    result = read(target / "report.json")
    result["rows"][0]["modeled_darcy_f"] *= 2
    atomic_json(target / "report.json", result)
    manifest = read(target / "manifest.json")
    manifest["artifacts"]["report.json"] = hashlib.sha256(
        (target / "report.json").read_bytes()
    ).hexdigest()
    atomic_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError, match="recomputation"):
        pipe.verify(target)


@pytest.mark.parametrize("fault", ["extra", "symlink", "source"])
def test_invalid_output_refused(output, tmp_path, fault):
    target = tmp_path / "report"
    shutil.copytree(output, target)
    if fault == "extra":
        (target / "extra").write_text("x")
    elif fault == "symlink":
        (target / "comparison.png").unlink()
        (target / "comparison.png").symlink_to(output / "comparison.png")
    else:
        manifest = read(target / "manifest.json")
        manifest["source"]["hydraulics_sha256"] = "0" * 64
        atomic_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError):
        pipe.verify(target)


def test_no_overwrite(output):
    with pytest.raises(FileExistsError):
        pipe.run(output)
    assert pipe.verify(output)["passed"]


def test_cli_verification_and_failure_exit(output, monkeypatch, capsys):
    assert main(["verify", str(output)]) == 0
    result = copy.deepcopy(pipe.evaluate())
    result["passed"] = False
    monkeypatch.setattr(pipe, "verify", lambda folder: result)
    assert main(["verify", str(output)]) == 2
    assert "False" in capsys.readouterr().out
