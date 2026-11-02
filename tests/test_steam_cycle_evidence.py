"""Fail closed on substituted dynamic-cycle campaigns and rehashed trajectories."""

import json
from copy import deepcopy

import pytest

from scripts.audit_steam_cycle import audit
from scripts.steam_cycle_references import reference
from scripts.verify_steam_cycle import BUDGET, assess, matrix, sha
from sdk.experiment import _source_fingerprint
from sdk.steam_cycle_experiment import run_steam_cycle_experiment


@pytest.fixture
def bundle(tmp_path):
    spec = deepcopy(next(matrix())[2])
    spec.update(duration_s=0.2, step_s=0.1, timeline=[])
    folder = tmp_path / "bundle"
    run_steam_cycle_experiment(spec, folder)
    return folder, json.loads(BUDGET.read_text())


def rehash(folder, name):
    path = folder / "manifest.json"
    data = json.loads(path.read_text())
    data["artifacts"][name] = sha(folder / name)
    path.write_text(json.dumps(data))


def test_independent_assessment(bundle):
    assert assess(*bundle)["accepted"]
    spec = json.loads((bundle[0] / "input.json").read_text())
    original = spec["controls"]["heater_power_w"]
    spec.update(
        duration_s=0.001,
        timeline=[
            dict(t=0.0005, controls=dict(spec["controls"], heater_power_w=original + 100000))
        ],
    )
    at, _ = reference(spec)
    assert at(0.000501)["state"][7] == pytest.approx(
        original * 0.0005 + (original + 100000) * 0.000001, abs=1e-8
    )


@pytest.mark.parametrize(
    "defect", ["grid", "nan", "controls", "energy", "classification", "inventory", "port"]
)
def test_rehashed_substitution_rejected(bundle, defect):
    folder, budget = bundle
    path = folder / "trajectory.json"
    rows = json.loads(path.read_text())
    if defect == "grid":
        rows.pop(1)
    elif defect == "nan":
        rows[1]["omega_rad_s"] = float("nan")
    elif defect == "controls":
        rows[1]["controls"]["valve_opening"] = 0.5
    elif defect == "energy":
        rows[1]["stored_energy_change_j"] += 10
    elif defect == "inventory":
        rows[1]["vessels"]["steam_generator"]["internal_energy_j"] += 10
    elif defect == "port":
        rows[1]["internal_enthalpy_ports_w"]["feed_to_hot"] += 100
    else:
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["status"] = "domain_exit"
        report_path.write_text(json.dumps(report))
        rehash(folder, "report.json")
    path.write_text(json.dumps(rows))
    rehash(folder, "trajectory.json")
    if defect in ("energy", "port"):
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
        script_sha256=sha("scripts/verify_steam_cycle.py"),
        reference_sha256=sha("scripts/steam_cycle_references.py"),
        helper_sha256=sha("scripts/verify_turbine.py"),
        budget_sha256=sha(BUDGET),
        cases=records,
        integrations=len(records),
        case_families=10,
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
