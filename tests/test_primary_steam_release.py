"""Release selection rejects stale audits and mixed-source evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.primary_steam_release import selection


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def release(tmp_path):
    folder = tmp_path / "dist"
    folder.mkdir()
    (folder / "runtime.whl").write_bytes(b"wheel placeholder; selection test only")
    (folder / "runtime.tar.gz").write_bytes(b"source placeholder; no numerical test")
    drivers = {}
    for name in (
        "audit_primary_steam",
        "qualify_primary_steam",
        "primary_steam_references",
        "verify_primary_steam",
        "verify_turbine",
        "turbine_references",
        "design_primary_steam",
        "qualify_primary_steam_design",
    ):
        path = folder / "frozen_scripts" / (name + ".py")
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Frozen synthetic driver\n")
        drivers["scripts/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    budget = folder / "frozen_budgets/ai05_primary_steam_budget_test.json"
    write(budget, dict(scope="selection test"))
    digest = hashlib.sha256(budget.read_bytes()).hexdigest()
    assembly = dict(
        source_sha256="a" * 64,
        budget_sha256=digest,
        driver_sha256={
            k: v
            for k, v in drivers.items()
            if Path(k).stem
            not in ("turbine_references", "design_primary_steam", "qualify_primary_steam_design")
        },
    )
    signature = dict(
        assembly=assembly,
        design_budget_sha256=digest,
        designer_sha256=drivers["scripts/design_primary_steam.py"],
        qualification_sha256=drivers["scripts/qualify_primary_steam_design.py"],
    )
    producer = dict(passed=True, integrations=14, case_families=7, signature=signature)
    qualified = tmp_path / "artifacts/steady/qualification.json"
    write(qualified, producer)
    checked = dict(
        producer, qualification_sha256=hashlib.sha256(qualified.read_bytes()).hexdigest()
    )
    write(tmp_path / "audit.json", checked)
    return tmp_path


def collect(root, **kwargs):
    return selection(root, "dist", steady="artifacts/steady", steady_audit="audit.json", **kwargs)


def test_selects_frozen_drivers_budgets_and_exact_source(release):
    result = collect(release)
    assert result["runtime"]["source_sha256"] == "a" * 64
    assert result["metadata"]["primary_steam_replay"]["steady_qualification"] == "artifacts/steady"
    entries = {r["path"] for r in result["entries"]}
    assert "dist/frozen_scripts/turbine_references.py" in entries
    assert "dist/frozen_budgets/ai05_primary_steam_budget_test.json" in entries
    assert result["metadata"]["validated"] is False


@pytest.mark.parametrize("attack", ["audit_rejected", "wrong_count", "stale_producer", "signature"])
def test_stale_or_rejected_qualification_cannot_enter_release(release, attack):
    path = release / "audit.json"
    value = json.loads(path.read_text())
    if attack == "audit_rejected":
        value["passed"] = False
    elif attack == "wrong_count":
        value["integrations"] = 13
    elif attack == "stale_producer":
        value["qualification_sha256"] = "0" * 64
    else:
        value["signature"]["assembly"]["source_sha256"] = "b" * 64
    write(path, value)
    with pytest.raises(ValueError, match="disagree"):
        collect(release)


def test_changed_driver_cannot_be_labeled_as_frozen_accepted_version(release):
    (release / "dist/frozen_scripts/verify_primary_steam.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="driver differs"):
        collect(release)


def test_missing_accepted_budget_is_rejected(release):
    (release / "dist/frozen_budgets/ai05_primary_steam_budget_test.json").write_text(
        '{"changed":true}'
    )
    with pytest.raises(ValueError, match="budget missing"):
        collect(release)


def test_multiple_runtime_distributions_are_ambiguous(release):
    (release / "dist/other.whl").write_bytes(b"other")
    with pytest.raises(ValueError, match="one wheel"):
        collect(release)


def test_mixed_source_profiles_cannot_share_one_release(release):
    steady = json.loads((release / "audit.json").read_text())
    signature = dict(steady["signature"]["assembly"], source_sha256="b" * 64)
    producer = dict(passed=True, integrations=53, case_families=17, signature=signature)
    q = release / "artifacts/transient/qualification.json"
    write(q, producer)
    write(
        release / "transient-audit.json",
        dict(producer, qualification_sha256=hashlib.sha256(q.read_bytes()).hexdigest()),
    )
    with pytest.raises(ValueError, match="one runtime source identity"):
        collect(release, transient="artifacts/transient", transient_audit="transient-audit.json")


def test_empty_profile_and_missing_partner_are_rejected(release):
    with pytest.raises(ValueError, match="accepted profiles"):
        selection(release, "dist")
    with pytest.raises(ValueError, match="requires its accepted audit"):
        selection(release, "dist", steady="artifacts/steady")


def test_source_outside_selected_root_is_rejected(release):
    with pytest.raises(ValueError, match="inside"):
        selection(
            release,
            release.parent / "outside",
            steady="artifacts/steady",
            steady_audit="audit.json",
        )


def fixed_profile(root):
    current = json.loads((root / "audit.json").read_text())
    for name in ("solve_fixed_primary_steam", "qualify_fixed_primary_steam"):
        path = root / "dist/frozen_scripts" / (name + ".py")
        path.write_text("# fixed synthetic driver\n")
    driver = root / "dist/frozen_scripts/solve_fixed_primary_steam.py"
    producer = root / "artifacts/fixed/qualification.json"
    budget = dict(
        cases=[dict(name=str(i)) for i in range(10)],
        policies=dict(radau={}, bdf={}),
        negative_cases=["rejection_" + str(i) for i in range(9)],
    )
    write(root / "artifacts/fixed/budget.json", budget)
    write(root / "dist/frozen_budgets/ai05_fixed_boundary_qualification_budget_test.json", budget)
    signature = dict(
        assembly=current["signature"]["assembly"],
        qualification_sha256=hashlib.sha256(
            (root / "dist/frozen_scripts/qualify_fixed_primary_steam.py").read_bytes()
        ).hexdigest(),
        initializer_driver_sha256={
            "scripts/solve_fixed_primary_steam.py": hashlib.sha256(driver.read_bytes()).hexdigest()
        },
        stationary_verifier_sha256=current["signature"]["qualification_sha256"],
        steady_budget_sha256=current["signature"]["design_budget_sha256"],
        budget_sha256=hashlib.sha256(
            (root / "artifacts/fixed/budget.json").read_bytes()
        ).hexdigest(),
    )
    record = dict(passed=True, integrations=20, case_families=10, signature=signature)
    write(producer, record)
    checked = dict(
        record,
        qualification_sha256=hashlib.sha256(producer.read_bytes()).hexdigest(),
        negative_checks=9,
    )
    write(root / "fixed-audit.json", checked)


def test_fixed_profile_collects_initializer_and_both_budgets(release):
    fixed_profile(release)
    result = selection(release, "dist", fixed="artifacts/fixed", fixed_audit="fixed-audit.json")
    paths = {r["path"] for r in result["entries"]}
    assert "dist/frozen_scripts/solve_fixed_primary_steam.py" in paths
    assert "dist/frozen_scripts/qualify_fixed_primary_steam.py" in paths
    assert "dist/frozen_budgets/ai05_fixed_boundary_qualification_budget_test.json" in paths
    assert result["metadata"]["primary_steam_replay"]["fixed_qualification"] == "artifacts/fixed"


def test_fixed_profile_cannot_drop_negative_checks(release):
    fixed_profile(release)
    path = release / "fixed-audit.json"
    value = json.loads(path.read_text())
    value["negative_checks"] = 0
    write(path, value)
    with pytest.raises(ValueError, match="rejection matrix"):
        selection(release, "dist", fixed="artifacts/fixed", fixed_audit="fixed-audit.json")
