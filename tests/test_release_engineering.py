"""Executable five-task release gates; external availability is never a fake pass."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from atomic_model_agents import jobs
from atomic_model_release import checks, environment, lanes
from atomic_model_workspace.common import atomic_json, read
from atomic_model_workspace.project import load_project

ROOT = Path(__file__).resolve().parents[1]


def test_golden_is_frozen_and_replayed_with_declared_units():
    result = checks.golden()
    assert result["golden_passed"] and not result["validated"]
    oracle = read(checks.GOLDEN / "oracle.json")
    assert (
        oracle["baseline_source_sha256"]
        == "71fe3f4ce5c9f3e4dc86a36c17c1cdb32470ca19e0214ebc9c91af1c1a997d67"
    )
    assert oracle["temperature_abs_tol_k"] == 1e-9


def test_golden_refuses_mutated_oracle_and_nonfinite_actual(tmp_path, monkeypatch):
    target = tmp_path / "golden"
    shutil.copytree(checks.GOLDEN, target)
    oracle = read(target / "oracle.json")
    oracle["time_s"] = 1
    atomic_json(target / "oracle.json", oracle)
    monkeypatch.setattr(checks, "GOLDEN", target)
    with pytest.raises(ValueError, match="integrity"):
        checks.golden()
    with pytest.raises(ValueError):
        checks._near(float("nan"), 1, 1e-6)


def test_explicit_data_migration_preserves_every_model_value(tmp_path):
    before = load_project(checks.GOLDEN / "project/project.json")
    result = checks.migrate_project(checks.GOLDEN / "project/project.json", tmp_path / "migration")
    from atomic_model_study import store

    after = store.inspect(tmp_path / "migration/reimported")["project"]
    assert before == after and result["semantic_project_equal"]
    assert result["source_checkpoints_migrated"] is False
    with pytest.raises(FileExistsError):
        checks.migrate_project(checks.GOLDEN / "project/project.json", tmp_path / "migration")


@pytest.mark.parametrize("defect", ["future_schema", "missing", "unknown", "symlink", "traversal"])
def test_unsupported_or_bad_project_never_migrates(tmp_path, defect):
    folder = tmp_path / "project"
    shutil.copytree(checks.GOLDEN / "project", folder)
    value = read(folder / "project.json")
    if defect == "future_schema":
        value["schema_version"] = 2
    elif defect == "unknown":
        value["guess_units"] = True
    elif defect == "missing":
        (folder / "scenarios/uniform-train.json").unlink()
    elif defect == "symlink":
        item = folder / "scenarios/uniform-train.json"
        raw = item.read_bytes()
        item.unlink()
        (tmp_path / "other.json").write_bytes(raw)
        item.symlink_to(tmp_path / "other.json")
    else:
        value["scenarios"]["uniform-train"] = "../outside.json"
    atomic_json(folder / "project.json", value)
    with pytest.raises((ValueError, OSError)):
        checks.migrate_project(folder / "project.json", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.fixture
def runbook(tmp_path):
    folder = tmp_path / "case"
    report = checks.control_demo(folder)
    return folder, report


def test_runbook_and_readonly_diagnosis_are_concrete(runbook):
    folder, report = runbook
    assert (
        report["control_completed"]
        and report["agent_free_replay"]
        and report["accepted_decisions"] == 3
    )
    head = (folder / "job/HEAD.json").read_bytes()
    diagnostic = checks.diagnose(folder / "job")
    assert diagnostic["observed_reason"] == "time_limit" and diagnostic["agent_executed"] is False
    assert head == (folder / "job/HEAD.json").read_bytes()


def test_unknown_checkpoint_schema_is_not_reinterpreted(runbook, tmp_path):
    folder, _ = runbook
    head = read(folder / "job/HEAD.json")
    head["schema_version"] = 2
    atomic_json(folder / "job/HEAD.json", head)
    with pytest.raises(ValueError, match="schema"):
        checks.diagnose(folder / "job")


def test_doctor_never_deletes_or_takes_over_lock(tmp_path):
    marker = tmp_path / ".lock"
    marker.write_text(str(os.getpid()))
    result = environment.doctor(tmp_path)
    assert result["lock"]["pid_present"] and result["lock"]["removed"] is False
    assert marker.read_text() == str(os.getpid()) and result["external_runtime_qualified"] is False
    marker.write_text("unknown")
    assert environment.doctor(tmp_path)["lock"]["pid"] is None


@pytest.mark.parametrize(
    "file,name,expected",
    [
        ("test_mgxs_pipeline.py", "test_openmc_particle_run_reports_k_and_mgxs", "openmc"),
        ("test_batch.py", "test_performance", "hardware"),
        ("test_simulator.py", "test_any", "long"),
        ("test_integration.py", "test_any", "long"),
        ("test_transient_convergence.py", "test_any", "long"),
        ("test_muscl_total_initialization.py", "test_any", "long"),
        ("test_external_agents.py", "test_any", "portable"),
        ("test_model_arena.py", "test_any", "portable"),
    ],
)
def test_lanes_are_disjoint(file, name, expected):
    assert lanes.classify(file, name) == expected


@pytest.mark.parametrize("lane", ["openmc", "mpi", "hardware"])
def test_missing_external_prerequisite_is_failure(lane, monkeypatch):
    monkeypatch.setattr(lanes.shutil, "which", lambda name: None)
    monkeypatch.setattr(lanes.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
    monkeypatch.delenv("ATOMIC_HARDWARE_PROFILE", raising=False)
    with pytest.raises(ValueError, match="missing"):
        lanes.preflight(lane)


def test_presence_is_not_external_qualification(tmp_path, monkeypatch):
    marker = tmp_path / "cross.xml"
    marker.write_text("configured path only")
    Path(str(marker) + ".license.json").write_text("{}")
    monkeypatch.setenv("OPENMC_CROSS_SECTIONS", str(marker))
    monkeypatch.setattr(lanes.shutil, "which", lambda name: "/configured/" + name)
    value = lanes.preflight("openmc")
    assert value["prerequisites_present"] and not value["executed"] and not value["qualified"]


@pytest.mark.parametrize(
    "body,expected",
    [
        ("def test_real(): assert 1+1==2", 0),
        ('import pytest\n@pytest.mark.skip(reason="unavailable")\ndef test_missing(): pass', 1),
        ("# no tests", 5),
        ("def test_failed(): assert False", 1),
    ],
)
def test_pytest_lane_gate_cannot_pass_all_skipped(tmp_path, body, expected):
    (tmp_path / "test_probe.py").write_text(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test_probe.py",
            "-p",
            "atomic_model_release.pytest_gate",
            "--atomic-require-executed",
            "--atomic-no-skips",
            "--atomic-receipt",
            "receipt.json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    receipt = read(tmp_path / "receipt.json")
    assert receipt["exit_code"] == expected
    if expected == 1 and "skip" in body:
        assert receipt["gate_failures"]


def test_stubs_check_real_public_call_and_reject_bad_input(tmp_path):
    valid = 'from pathlib import Path\nfrom atomic_model_agents import jobs\nfrom atomic_model_contracts import Scenario,AgentDescriptor\ndef call(s: Scenario,a: AgentDescriptor)->None:\n jobs.create(s,a,Path("job"),["python"])\n'
    bad = 'from pathlib import Path\nfrom atomic_model_agents import jobs\njobs.create("not-scenario", "not-descriptor", Path("job"), ["python"])\n'
    for name, body, success in [("valid", valid, True), ("bad", bad, False)]:
        path = tmp_path / (name + ".py")
        path.write_text(body)
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "--follow-imports=silent", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        if not success:
            assert "arg-type" in result.stdout


def test_workflow_profiles_keep_manual_runtimes_separate():
    folder = ROOT / ".github/workflows"
    docs = {p.stem: yaml.safe_load(p.read_text()) for p in folder.glob("*.yml")}
    portable = docs["ci"]
    steps = portable["jobs"]["portable"]["steps"]
    command = "\n".join(s.get("run", "") for s in steps)
    assert (
        "not hardware_specific" in command
        and "not long_running" in command
        and "not external_integration" in command
    )
    assert "mypy --strict" in command and "install_check" in command
    for name in ["external", "hardware", "numerical"]:
        event = docs[name].get("on", docs[name].get(True))
        assert "workflow_dispatch" in event and "push" not in event and "pull_request" not in event
    external = docs["external"]["jobs"]
    assert "atomic-openmc" in external["openmc"]["runs-on"]
    assert "atomic-mpi" in external["mpi-runtime"]["runs-on"]
    assert "--atomic-no-skips" in str(external["openmc"])
    assert "mpiexec -n 2" in str(external["mpi-runtime"])


def test_doctor_cli_does_not_import_numerical_modules():
    code = 'import sys;from atomic_model_release.__main__ import main;main(["doctor"]);assert "numpy" not in sys.modules;assert "scipy" not in sys.modules'
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_release_cli_and_installed_entrypoints(runbook):
    result = subprocess.run(
        [sys.executable, "-m", "atomic_model_release", "diagnose", str(runbook[0] / "job")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0 and json.loads(result.stdout)["agent_executed"] is False, (
        result.stderr
    )


def test_source_bound_snapshot_is_not_migrated_or_relabelled(runbook, tmp_path):
    from atomic_model_workspace.common import digest

    folder = runbook[0] / "job"
    head = read(folder / "HEAD.json")
    value = read(folder / head["checkpoint"])
    value["payload"]["source"]["agent_runtime_sha256"] = "0" * 64
    value["sha256"] = digest(value["payload"])
    name = "revisions/" + value["sha256"] + ".json"
    atomic_json(folder / name, value)
    atomic_json(
        folder / "HEAD.json", dict(schema_version=1, checkpoint=name, sha256=value["sha256"])
    )
    with pytest.raises(ValueError, match="source"):
        checks.diagnose(folder)
    with pytest.raises(ValueError):
        checks.migrate_project(folder / name, tmp_path / "not-a-project")
    assert not (tmp_path / "not-a-project").exists()


def test_golden_rejects_unit_semantics_change(monkeypatch):
    from atomic_model_workspace import jobs as workspace_jobs

    original = workspace_jobs._row

    def wrong_units(*args, **kwargs):
        row = original(*args, **kwargs)
        row["observation"]["channels"][0]["unit"] = "degC"
        return row

    monkeypatch.setattr(workspace_jobs, "_row", wrong_units)
    with pytest.raises(ValueError, match="unit"):
        checks.golden()


def test_three_use_cases_are_installed_public_contracts():
    manifest = read(Path(checks.__file__).parent / "use_cases.json")
    assert manifest["model_id"] == "eos-thermal-assembly-v1"
    assert set(manifest["cases"]) == {"control", "diagnosis", "learning_data"}
    assert "synthetic" in manifest["fidelity"]
    assert "null" in manifest["cases"]["learning_data"]["outputs"]


def test_negative_diagnostic_demo_reports_failure_without_advancing(tmp_path):
    value = checks.diagnostic_demo(tmp_path / "diagnostic")
    assert value["demo_passed"] and value["expected_failure"] and value["simulator_unchanged"]
    assert (
        value["diagnosis"]["status"] == "failed" and value["diagnosis"]["accepted_decisions"] == 0
    )
    assert not value["expert_cause_claimed"]
    assert jobs.replay(tmp_path / "diagnostic/export")["agent_executed"] is False


def test_dataset_use_case_is_verified_and_never_invents_reward(tmp_path):
    value = checks.data_demo(tmp_path / "dataset")
    assert value["transitions"] == 48 and value["splits"] == {"train": 24, "dev": 12, "test": 12}
    assert value["terminated"] == 0 and value["truncated"] == 16 and not value["reward_defined"]
    assert value["agent_executed"] is True and value["audit_agent_executed"] is False
    assert checks.dataset_summary(tmp_path / "dataset/benchmark")["agent_executed"] is False
