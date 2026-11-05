"""Real primary-data provenance, strict scope and bounded-error regressions."""

import io
import json
import shutil
import zipfile
from copy import deepcopy

import pytest

from atomic_model_hem_assessment import audit, importer
from atomic_model_hem_assessment.__main__ import main
from atomic_model_hem_qualified import eos
from atomic_model_workspace.common import digest


def test_primary_inventory_units_and_unmodified_quality():
    manifest, rows = audit.dataset()
    assert len(rows) == 12320
    assert len({r["run"] for r in rows}) == len(manifest["workbooks"]) == 176
    r = next(r for r in rows if r["row_id"] == "0170:03")
    assert r["pressure_pa"] == pytest.approx(165.91039750427794 * 1e5)
    assert r["inlet_temperature_k"] == pytest.approx(331.07525152779994 + 273.15)
    assert r["quality"] == pytest.approx(-0.16066021426923194)
    assert any(r["quality"] > 1 for r in rows)
    assert manifest["license"] == "CC-BY-SA-4.0"
    assert manifest["uncertainty"]["confidence_level"] is None


def test_all_primary_nominals_inside_published_asymmetric_bounds():
    _, rows = audit.dataset()
    for r in rows:
        for field in importer.BOUNDS:
            assert r[field + "_lower"] <= r[field] <= r[field + "_upper"]
    assert any(
        r["bulk_enthalpy_j_kg_upper"] - r["bulk_enthalpy_j_kg"]
        != r["bulk_enthalpy_j_kg"] - r["bulk_enthalpy_j_kg_lower"]
        for r in rows
    )


@pytest.mark.parametrize("kind", ["manifest", "data", "notice", "extra", "symlink"])
def test_dataset_tamper_is_rejected(tmp_path, kind):
    dest = tmp_path / "data"
    shutil.copytree(audit.DATA, dest)
    if kind == "manifest":
        (dest / "manifest.json").write_text("{}")
    elif kind == "data":
        (dest / "observations.csv.gz").write_bytes(b"invalid")
    elif kind == "notice":
        (dest / "NOTICE.md").write_text("different license")
    elif kind == "extra":
        (dest / "extra").write_text("unexpected")
    else:
        p = dest / "NOTICE.md"
        p.unlink()
        p.symlink_to(audit.DATA / "NOTICE.md")
    with pytest.raises(ValueError):
        audit.dataset(dest)


def test_import_is_deterministic_without_reducing_source_precision():
    _, rows = audit.dataset()
    encoded = importer.encode(rows[:2])
    assert encoded == importer.encode(deepcopy(rows[:2]))
    assert importer.sha(encoded) == importer.sha(importer.encode(rows[:2]))


@pytest.mark.parametrize("name", ["../data", "/data", "a\\b"])
def test_archive_traversal_rejected(name):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(name, "test")
    with pytest.raises(ValueError):
        importer.bounded_zip(out.getvalue())


def test_duplicate_archive_and_entity_injection_rejected():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("a", "first")
        with pytest.warns(UserWarning):
            z.writestr("a", "second")
    with pytest.raises(ValueError):
        importer.bounded_zip(out.getvalue())
    with pytest.raises(ValueError):
        importer.xml(b'<!DOCTYPE a [<!ENTITY x "test">]><a>&x;</a>')


def test_formula_caches_are_not_silently_treated_as_measurements():
    out = io.BytesIO()
    ns = importer.NS["s"]
    data = f'<worksheet xmlns="{ns}"><sheetData><row r="1"><c r="A1"><f>1+1</f><v>2</v></c></row></sheetData></worksheet>'
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", data)
        z.writestr("xl/worksheets/sheet2.xml", data)
    with pytest.raises(ValueError, match="Formula"):
        importer.workbook(out.getvalue())


def eligible_test_point():
    # A synthetic unit-test point, never registered as a primary observation.
    return dict(
        pressure_pa=1e6,
        quality=0.05,
        diameter_m=0.008,
        length_m=2.0,
        mass_flux_kg_m2_s=500.0,
        wall_temperature_k=eos.px(1e6, 0.05).temperature_k + 2,
        wall_superheat_k=2.0,
    )


def test_scope_uses_real_production_closure_and_keeps_system_separate():
    r = audit.admission(eligible_test_point())
    assert r["local_closure_eligible"]
    assert r["system_experiment_eligible"] is False
    assert "prescribed_steady_throughflow_not_closed_finite_graph" in r["system_reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("pressure_pa", 17e6, "outside_dynamic_pressure_domain"),
        ("quality", -0.01, "outside_wet_quality_domain"),
        ("length_m", 7.1, "outside_boiling_heated_length_domain"),
        ("wall_superheat_k", 30.0, "outside_wall_superheat_domain"),
        ("diameter_m", 0.1, "no_supported_chf_correlation_for_geometry_and_state"),
    ],
)
def test_scope_rejects_each_mismatch_without_expanding_correlations(field, value, reason):
    r = eligible_test_point()
    r[field] = value
    result = audit.admission(r)
    assert reason in result["reasons"]
    assert not result["local_closure_eligible"]


def test_all_primary_rows_censused_no_false_zero_error():
    r = audit.evaluate()
    assert r["observations"] == sum(x["observations"] for x in r["per_run"].values()) == 12320
    assert len(r["assessment"]) == 12320
    assert r["local_closure_eligible"] == r["matched_system_experiments"] == 0
    assert r["error_metrics"] is None
    assert not r["physical_acceptance"] and not r["comparison_performed"]
    assert r["rejection_counts"]["no_supported_chf_correlation_for_geometry_and_state"] == 12320
    assert digest(r) == digest(json.loads(json.dumps(r)))


@pytest.mark.parametrize(
    "measured,modeled,gap,overlap",
    [
        ((10, 8, 11), (13, 12, 14), 1, False),
        ((10, 8, 11), (11, 10, 12), 0, True),
        ((0, -1, 1), (0, 0, 0), 0, True),
        ((10, 8, 11), (7, 6, 8), 0, True),
        ((10, 8, 11), (6, 5, 7), 1, False),
    ],
)
def test_asymmetric_bounded_comparison(measured, modeled, gap, overlap):
    def wrap(x):
        return dict(zip(("value", "lower", "upper"), x))

    r = audit.envelope_comparison(wrap(measured), wrap(modeled))
    assert r["envelope_gap"] == gap and r["overlaps"] == overlap
    assert not r["physical_acceptance"]
    if measured[0] == 0:
        assert r["relative_residual"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "3"])
def test_invalid_intervals_refused(value):
    with pytest.raises(ValueError):
        audit.envelope_comparison(
            dict(value=value, lower=0, upper=4), dict(value=1, lower=0, upper=2)
        )


def target():
    return dict(
        id="declared-research-envelope",
        pressure_pa=[2e5, 15.5e6],
        temperature_k=[310.0, 620.0],
        requires_post_chf=False,
        requires_metastability=False,
        requires_open_throughflow=False,
    )


def test_target_domain_is_not_plant_validation():
    r = audit.assess_target(target())
    assert r["supported_envelope"] and not r["plant_referenced"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pressure_pa", [1e5, 17e6]),
        ("temperature_k", [300.0, 1000.0]),
        ("requires_post_chf", True),
        ("requires_metastability", True),
        ("requires_open_throughflow", True),
    ],
)
def test_uncovered_target_requirements_are_visible(field, value):
    request = target()
    request[field] = value
    r = audit.assess_target(request)
    assert not r["supported_envelope"] and r["reasons"]


def test_cli_recompute_digest_source_and_overwrite_guards(tmp_path):
    path = tmp_path / "report.json"
    assert main(["screen", "--output", str(path)]) == 0
    assert main(["audit", str(path)]) == 0
    with pytest.raises(SystemExit):
        main(["screen", "--output", str(path)])
    r = json.loads(path.read_text())
    r["payload"]["local_closure_eligible"] = 12320
    r["sha256"] = digest(r["payload"])
    path.write_text(json.dumps(r))
    with pytest.raises(SystemExit):
        main(["audit", str(path)])


def test_real_acoustics_screen_is_not_independent_eight_run_validation():
    from atomic_model_hem_assessment import acoustics

    result = acoustics.screen()
    assert result["independent_experiments"] == 4
    assert result["local_eos_estimates"] == len(result["rows"]) == 8
    assert result["maximum_absolute_relative_residual"] == pytest.approx(
        0.03455297289529957, rel=1e-10
    )
    assert result["acceptance_threshold"] is None
    assert result["wave_speed_measurement_uncertainty"] is None
    assert not result["physical_acceptance"]
    assert all(len(r["input_sensitivity_samples"]) == 4 for r in result["rows"])


def test_invalid_zip_has_a_controlled_failure():
    with pytest.raises(ValueError, match="Invalid"):
        importer.bounded_zip(b"not a zip")


@pytest.mark.parametrize("bound", [eos.P_MIN + 100, eos.P_MAX - 100])
def test_target_pressure_derivative_margin_is_strict(bound):
    request = target()
    request["pressure_pa"] = [bound, bound]
    assert not audit.assess_target(request)["supported_envelope"]
