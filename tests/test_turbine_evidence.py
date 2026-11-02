"""Physical-stop and provenance checks on rehashed steam-train bundles."""

import json
from copy import deepcopy

import pytest

from scripts.audit_turbine import audit
from scripts.verify_turbine import BUDGET, assess, matrix, sha
from sdk.experiment import _source_fingerprint
from sdk.turbine_experiment import run_turbine_experiment


@pytest.fixture
def bundle(tmp_path):
    spec = deepcopy(next(matrix())[2])
    spec.update(duration_s=0.2, step_s=0.1, timeline=[])
    folder = tmp_path / "bundle"
    run_turbine_experiment(spec, folder)
    return folder, json.loads(BUDGET.read_text())


def rehash(folder, name):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][name] = sha(folder / name)
    path.write_text(json.dumps(manifest))


def test_independent_assessment(bundle):
    assert assess(*bundle)["accepted"]


@pytest.mark.parametrize("defect", ["grid", "nan", "controls", "classification", "energy"])
def test_rehashed_substitution_rejected(bundle, defect):
    folder, budget = bundle
    path = folder / "trajectory.json"
    rows = json.loads(path.read_text())
    if defect == "grid":
        rows.pop(1)
    elif defect == "nan":
        rows[1]["omega_rad_s"] = float("nan")
    elif defect == "controls":
        rows[1]["controls"]["mass_flow_kg_s"] = 1
    elif defect == "classification":
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["status"] = "domain_exit"
        report_path.write_text(json.dumps(report))
        rehash(folder, "report.json")
    else:
        rows[1]["electric_energy_j"] += 10
    path.write_text(json.dumps(rows))
    rehash(folder, "trajectory.json")
    if defect == "energy":
        assert not assess(folder, budget)["accepted"]
    else:
        with pytest.raises(ValueError):
            assess(folder, budget)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "stale", "substitution"])
def test_campaign_integrity(tmp_path, defect):
    specs = list(matrix())
    records = [
        dict(name=name, policy=policy, folder=str(tmp_path / f"{name}_{policy}"))
        for name, policy, _ in specs
    ]
    report = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_turbine.py"),
        reference_sha256=sha("scripts/turbine_references.py"),
        budget_sha256=sha(BUDGET),
        cases=records,
        integrations=len(records),
        case_families=12,
        all_cases_accepted=True,
    )
    if defect == "missing":
        records.pop()
        report["integrations"] -= 1
    elif defect == "duplicate":
        records[-1] = deepcopy(records[0])
    elif defect == "stale":
        report["source_sha256"] = "0" * 64
    else:
        folder = tmp_path / f"{specs[0][0]}_{specs[0][1]}"
        folder.mkdir()
        spec = deepcopy(specs[0][2])
        spec["duration_s"] /= 2
        (folder / "input.json").write_text(json.dumps(spec))
    (tmp_path / "verification.json").write_text(json.dumps(report))
    (tmp_path / "budget.json").write_bytes(BUDGET.read_bytes())
    with pytest.raises(ValueError):
        audit(tmp_path)
