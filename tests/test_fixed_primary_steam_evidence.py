"""Source closure, declared matrix and rejection provenance for fixed boundaries."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import qualify_fixed_primary_steam as qualification
from scripts.solve_fixed_primary_steam import SteadySolveError


@pytest.fixture
def packet(tmp_path):
    budget = json.loads(qualification.BUDGET.read_text())
    captured = qualification.signature()
    (tmp_path / "drivers").mkdir()
    for path in qualification.driver_hashes(captured):
        shutil.copyfile(path, tmp_path / "drivers" / Path(path).name)
    (tmp_path / "budget.json").write_text(json.dumps(budget))
    report = dict(
        signature=captured,
        passed=True,
        cases=[],
        negative_cases=[],
        integrations=len(budget["cases"]) * len(budget["policies"]),
        case_families=len(budget["cases"]),
    )
    (tmp_path / "qualification.json").write_text(json.dumps(report))
    return tmp_path


@pytest.mark.parametrize(
    "driver",
    ["solve_fixed_primary_steam.py", "verify_primary_steam.py", "qualify_fixed_primary_steam.py"],
)
def test_frozen_driver_tampering_is_rejected_before_numerical_replay(packet, driver):
    (packet / "drivers" / driver).write_text("# tampered\n")
    with pytest.raises(ValueError, match="Frozen fixed-boundary driver"):
        qualification.audit(packet)


def test_incomplete_fixed_matrix_is_not_an_accepted_dossier(packet):
    with pytest.raises(ValueError, match="Incomplete or duplicated"):
        qualification.audit(packet)


def test_budget_tampering_is_not_rebound_to_a_captured_signature(packet):
    value = json.loads((packet / "budget.json").read_text())
    value["horizon_s"] = 1
    (packet / "budget.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="runtime, drivers or budget"):
        qualification.audit(packet)


def test_declared_matrix_uses_both_methods_full_horizon_and_explicit_anchors():
    budget = json.loads(qualification.BUDGET.read_text())
    rows = list(qualification.matrix(budget))
    identities = [(name, policy) for name, policy, _, _ in rows]
    assert len(set(identities)) == len(rows) == len(budget["cases"]) * len(budget["policies"])
    for _, _, spec, anchors in rows:
        assert spec["duration_s"] == 120 and spec["timeline"] == []
        assert set(anchors) == {"total_steam_mass_kg", "steam_generator_quality"}
        assert all(
            t == 1000
            for t in spec["primary"]["loop"]["initialization"]["fluid_temperature_k"].values()
        )
    assert "pressure_endpoint" not in {name for name, _, _, _ in rows}
    assert "pressure_endpoint" in budget["negative_cases"]


def test_every_declared_rejection_has_exact_criterion_and_unchanged_request():
    budget = json.loads(qualification.BUDGET.read_text())
    seen = set()
    for name, spec, anchors, criterion in qualification.negative_matrix(budget):
        before = deepcopy(spec)
        report = qualification.rejection(spec, anchors, criterion)
        assert not report["accepted"] and report["criterion"] == criterion and spec == before
        seen.add(name)
    assert seen == set(budget["negative_cases"])


def test_rejection_purity_check_does_not_trust_a_false_commands_changed_flag(monkeypatch):
    spec = {"controls": {"heater": 100}}

    def dishonest(value, **kwargs):
        value["controls"]["heater"] = 0
        raise SteadySolveError("expected", "rejected")

    monkeypatch.setattr(qualification, "solve_fixed_primary_steam", dishonest)
    with pytest.raises(ValueError, match="purity"):
        qualification.rejection(spec, {}, "expected")


def test_unexpected_acceptance_cannot_satisfy_negative_matrix(monkeypatch):
    monkeypatch.setattr(qualification, "solve_fixed_primary_steam", lambda *a, **k: ({}, {}))
    with pytest.raises(ValueError, match="incompatible point was accepted"):
        qualification.rejection({}, {}, "expected")
