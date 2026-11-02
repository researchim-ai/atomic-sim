"""Acceptance evidence must fail on omissions, substitutions and nonfinite data."""

import json
from copy import deepcopy

import pytest

from scripts.audit_exchanger import audit
from scripts.verify_exchanger import BUDGET, assess, matrix, sha
from sdk.exchanger_experiment import run_exchanger_experiment
from sdk.experiment import _source_fingerprint


@pytest.fixture
def bundle(tmp_path):
    spec = deepcopy(next(matrix())[2])
    spec.update(duration_s=0.3, step_s=0.1, timeline=[])
    folder = tmp_path / "bundle"
    run_exchanger_experiment(spec, folder)
    return folder, json.loads(BUDGET.read_text())


def rehash(folder, name):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"][name] = sha(folder / name)
    path.write_text(json.dumps(manifest))


def test_independent_assessment_passes(bundle):
    folder, budget = bundle
    assert assess(folder, budget)["accepted"]


@pytest.mark.parametrize("defect", ["omission", "controls", "phase", "nan", "inf", "events"])
def test_rehashed_trajectory_tampering_is_rejected(bundle, defect):
    folder, budget = bundle
    path = folder / "trajectory.json"
    rows = json.loads(path.read_text())
    if defect == "omission":
        rows.pop(1)
    elif defect == "controls":
        rows[1]["controls"]["a_mass_flow_kg_s"] = -1
    elif defect == "phase":
        rows[1]["phase"] = "initial"
    elif defect in ("nan", "inf"):
        rows[1]["temperature_k"]["a"][0] = float(defect)
    else:
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["events"] = [dict(t=0)]
        report_path.write_text(json.dumps(report))
        rehash(folder, "report.json")
    path.write_text(json.dumps(rows))
    rehash(folder, "trajectory.json")
    with pytest.raises(ValueError):
        assess(folder, budget)


def test_unhashed_artifact_corruption_is_rejected(bundle):
    folder, budget = bundle
    with (folder / "trajectory.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        assess(folder, budget)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "unknown", "count", "stale", "substitution"]
)
def test_audit_rejects_forged_campaign_before_replay(tmp_path, defect):
    specs = list(matrix())
    records = [
        dict(name=name, policy=policy, folder=str(tmp_path / f"{name}_{policy}"))
        for name, policy, _ in specs
    ]
    report = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_exchanger.py"),
        reference_sha256=sha("scripts/exchanger_references.py"),
        budget_sha256=sha(BUDGET),
        cases=records,
        integrations=len(records),
        case_families=13,
        all_cases_accepted=True,
    )
    if defect == "missing":
        records.pop()
        report["integrations"] -= 1
    elif defect == "duplicate":
        records[-1] = deepcopy(records[0])
    elif defect == "unknown":
        records[0]["policy"] = "unqualified"
    elif defect == "count":
        report["case_families"] = 12
    elif defect == "stale":
        report["source_sha256"] = "0" * 64
    else:
        folder = tmp_path / f"{specs[0][0]}_{specs[0][1]}"
        folder.mkdir()
        spec = deepcopy(specs[0][2])
        spec["duration_s"] /= 2
        (folder / "input.json").write_text(json.dumps(spec))
    (tmp_path / "budget.json").write_bytes(BUDGET.read_bytes())
    (tmp_path / "verification.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(tmp_path)
