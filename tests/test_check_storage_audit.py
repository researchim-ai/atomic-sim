"""Reject incomplete or substituted hybrid-valve acceptance evidence."""

import json
from copy import deepcopy

import pytest

from scripts.audit_check_storage import validate_series
from scripts.verify_check_storage import scenarios
from sdk.check_storage_experiment import prepare


@pytest.fixture
def series(tmp_path):
    records = []
    for index, (name, spec) in enumerate(scenarios()):
        policies = ("base", "tight", "bdf") if index < 2 else ("base", "tight")
        for policy in policies:
            candidate = deepcopy(spec)
            if policy != "base":
                candidate["solver"].update(rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
            if policy == "bdf":
                candidate["solver"]["method"] = "BDF"
            folder = tmp_path / f"{name}_{policy}"
            folder.mkdir()
            (folder / "input.json").write_text(json.dumps(prepare(candidate)[0]))
            records.append(dict(name=name, policy=policy, folder=str(folder)))
    return tmp_path, dict(cases=records, integrations=28, case_families=13)


def test_complete_matrix_accepts_reordered_records(series):
    folder, report = series
    report["cases"].reverse()
    validate_series(folder, report)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "count", "family"])
def test_rejects_forged_matrix(series, mutation):
    folder, report = series
    if mutation == "missing":
        report["cases"].pop()
        report["integrations"] -= 1
    elif mutation == "duplicate":
        report["cases"][-1] = deepcopy(report["cases"][0])
    elif mutation == "unknown":
        report["cases"][0]["policy"] = "unqualified"
    elif mutation == "count":
        report["integrations"] += 1
    else:
        report["case_families"] -= 1
    with pytest.raises(ValueError, match="acceptance matrix"):
        validate_series(folder, report)


@pytest.mark.parametrize("field", ["duration_s", "solver", "model", "timeline"])
def test_rejects_scenario_substitution(series, field):
    folder, report = series
    path = folder / "restart_base" / "input.json"
    spec = json.loads(path.read_text())
    if field == "duration_s":
        spec[field] = 0.1
    elif field == "solver":
        spec[field]["method"] = "BDF"
    elif field == "model":
        spec[field]["cracking_pressure_pa"] += 1
    else:
        spec[field] = []
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="declared acceptance scenario"):
        validate_series(folder, report)


def test_rejects_shared_bundle(series):
    folder, report = series
    report["cases"][1]["folder"] = report["cases"][0]["folder"]
    with pytest.raises(ValueError, match="Shared or external"):
        validate_series(folder, report)


def test_rejects_symlink_outside_evidence(series, tmp_path_factory):
    folder, report = series
    external = tmp_path_factory.mktemp("external-evidence")
    link = folder / "external-link"
    link.symlink_to(external, target_is_directory=True)
    report["cases"][0]["folder"] = str(link)
    with pytest.raises(ValueError, match="Shared or external"):
        validate_series(folder, report)
