"""Reject incomplete or repackaged two-cycle acceptance evidence."""

import json
from copy import deepcopy

import pytest

from scripts.audit_coupled_exchanger import audit
from scripts.verify_coupled_exchanger import BUDGET, assess, matrix, sha
from sdk.coupled_exchanger_experiment import run_coupled_exchanger_experiment
from sdk.experiment import _source_fingerprint


@pytest.fixture
def bundle(tmp_path):
    spec = deepcopy(next(matrix())[2])
    spec.update(duration_s=0.2, step_s=0.1, timeline=[])
    folder = tmp_path / "bundle"
    run_coupled_exchanger_experiment(spec, folder)
    return folder, json.loads(BUDGET.read_text())


def rehash(folder, name):
    path = folder / "manifest.json"
    record = json.loads(path.read_text())
    record["artifacts"][name] = sha(folder / name)
    path.write_text(json.dumps(record))


def test_independent_reference_passes(bundle):
    folder, budget = bundle
    assert assess(folder, budget)["accepted"]


@pytest.mark.parametrize("defect", ["grid", "controls", "nan", "energy"])
def test_rehashed_trajectory_substitution_rejected(bundle, defect):
    folder, budget = bundle
    path = folder / "trajectory.json"
    rows = json.loads(path.read_text())
    if defect == "grid":
        rows.pop(1)
    elif defect == "controls":
        rows[1]["controls"]["a_motor_torque_nm"] = -1
    elif defect == "nan":
        rows[1]["wall_temperature_k"][0] = float("nan")
    else:
        rows[1]["shared_wall_energy_change_j"] += 10
    path.write_text(json.dumps(rows))
    rehash(folder, "trajectory.json")
    if defect == "energy":
        assert not assess(folder, budget)["accepted"]
    else:
        with pytest.raises(ValueError):
            assess(folder, budget)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "count", "stale", "substitution"])
def test_rejects_campaign_substitution(tmp_path, defect):
    specs = list(matrix())
    records = [
        dict(name=name, policy=policy, folder=str(tmp_path / f"{name}_{policy}"))
        for name, policy, _ in specs
    ]
    report = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_coupled_exchanger.py"),
        reference_sha256=sha("scripts/coupled_exchanger_references.py"),
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
    (tmp_path / "verification.json").write_text(json.dumps(report))
    (tmp_path / "budget.json").write_bytes(BUDGET.read_bytes())
    with pytest.raises(ValueError):
        audit(tmp_path)
