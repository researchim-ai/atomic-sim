"""Durable lifecycle, constrained policies, campaign isolation and hostile archives."""

import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path

import pytest

from atomic_model_runtime import create_eos_episode
from atomic_model_workspace import common
from atomic_model_workspace.bundle import export_bundle, import_bundle, verify_bundle
from atomic_model_workspace.campaign import (
    cancel_campaign,
    create_campaign,
    load_campaign,
    run_campaign,
)
from atomic_model_workspace.common import Lock, atomic_json, digest, loads, read
from atomic_model_workspace.jobs import cancel_job, create_job, load_job, resume_job, run_job
from atomic_model_workspace.policies import decide, observation_frame, prepare_policy
from atomic_model_workspace.project import load_project, validate_project
from atomic_model_workspace.reports import (
    campaign_report,
    compare_campaigns,
    export_report,
    job_metrics,
)


@pytest.fixture
def project(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples/eos_workspace"
    shutil.copytree(root, tmp_path / "project")
    return tmp_path / "project/project.json"


def job_spec(project, policy="feedback"):
    p = load_project(project)
    return dict(
        schema_version=1,
        scenario=p["scenarios"]["mixed_dev"],
        policy=p["policies"][policy],
        decision_timeout_s=1,
    )


def change(path, fn):
    value = read(path)
    fn(value)
    atomic_json(path, value)


def test_project_portable_identity_and_public_limits(project, tmp_path, monkeypatch):
    a = validate_project(project)
    shutil.copytree(project.parent, tmp_path / "moved")
    monkeypatch.chdir(tmp_path)
    b = validate_project(tmp_path / "moved/project.json")
    assert a == b and a["jobs"] == 6 and a["model"]["validated"] is False
    assert "consume no RNG" in a["seed_semantics"]


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.json",
        "/tmp/absolute.json",
        "policies//hold.json",
        "policies/../policies/hold.json",
        "C:hold.json",
        "policies\\hold.json",
    ],
)
def test_project_rejects_unsafe_refs_before_output(project, tmp_path, bad):
    change(project, lambda p: p["policies"].update(hold=bad))
    with pytest.raises(ValueError):
        create_campaign(project, tmp_path / "campaign")
    assert not (tmp_path / "campaign").exists()


def test_project_rejects_symlink_reference(project, tmp_path):
    original = project.parent / "policies/hold.json"
    original.rename(project.parent / "outside.json")
    original.symlink_to("../outside.json")
    with pytest.raises(ValueError):
        load_project(project)


@pytest.mark.parametrize(
    "bad",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b'{"x":Infinity}',
    ],
)
def test_strict_json_rejects_duplicates_and_nonfinite(bad):
    with pytest.raises(ValueError):
        loads(bad)


@pytest.mark.parametrize(
    "defect",
    [
        "schema",
        "unknown",
        "seed",
        "split",
        "objective",
        "numerics",
        "policy",
        "mask",
        "future_policy",
        "budget",
    ],
)
def test_all_scenarios_and_policies_admitted_before_campaign(project, tmp_path, defect):
    scenario = project.parent / "scenarios/gravity_test.json"
    policy = project.parent / "policies/feedback.json"
    if defect == "schema":
        change(project, lambda p: p.update(schema_version=True))
    elif defect == "unknown":
        change(project, lambda p: p.update(unknown=1))
    elif defect == "seed":
        change(scenario, lambda p: p.update(seed=True))
    elif defect == "split":
        change(scenario, lambda p: p.update(split="evaluation"))
    elif defect == "objective":
        change(scenario, lambda p: p["objective"].update(sensor="hidden.temperature"))
    elif defect == "numerics":
        change(scenario, lambda p: p["config"].update(max_step_s=0.1))
    elif defect == "policy":
        change(policy, lambda p: p.update(kind="user.module:code"))
    elif defect == "mask":
        change(project.parent / "policies/hold.json", lambda p: p["heat_power_w"].pop("wall.w0"))
    elif defect == "future_policy":
        change(scenario, lambda p: p["config"].update(preset="uniform"))
    elif defect == "budget":
        change(project, lambda p: p["limits"].update(job_timeout_s=0))
    with pytest.raises(ValueError):
        create_campaign(project, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_feedback_only_observation_and_complete_bounded_action(project):
    spec = job_spec(project)
    e = create_eos_episode(spec["scenario"]["config"])
    frame = observation_frame(e)
    assert set(frame) == {"time_s", "channels"}
    assert all(
        set(c) == {"name", "unit", "value", "valid", "sample_time_s"} for c in frame["channels"]
    )
    policy = prepare_policy(spec["policy"], e)
    policy["gain_w_k"] = 1e6
    action = decide(policy, frame, e.config.action_channels)
    assert set(action) == set(e.config.action_channels) and action["wall.w0"] == 500
    assert sum(v != 0 for v in action.values()) == 1
    frame["channels"][-2]["valid"] = False
    with pytest.raises(ValueError):
        decide(policy, frame, e.config.action_channels)


def test_full_pause_resume_exact_native_state_and_policy_trace(project, tmp_path):
    spec = job_spec(project)
    create_job(spec, tmp_path / "full")
    create_job(spec, tmp_path / "split")
    full = run_job(tmp_path / "full")
    paused = run_job(tmp_path / "split", max_decisions=1)
    assert paused["status"] == "paused" and paused["episode"]["steps"] == 1
    split = run_job(tmp_path / "split")
    assert split["episode"] == full["episode"] and split["status"] == "completed"
    assert split["rows"][-1]["ledger"] == full["rows"][-1]["ledger"]
    direct = create_eos_episode(spec["scenario"]["config"])
    for row in full["rows"][1:]:
        direct.step([row["heat_power_w"][k] for k in direct.config.action_channels])
    assert direct.export_episode() == full["episode"]
    before = read(tmp_path / "full/HEAD.json")
    assert run_job(tmp_path / "full")["episode"] == full["episode"]
    assert read(tmp_path / "full/HEAD.json") == before


def test_cancel_and_explicit_resume_keep_accepted_prefix(project, tmp_path):
    folder = tmp_path / "job"
    initial = create_job(job_spec(project), folder)
    cancel_job(folder)
    cancelled = run_job(folder)
    assert cancelled["status"] == "cancelled" and cancelled["episode"] == initial["episode"]
    assert run_job(folder)["status"] == "cancelled"
    resume_job(folder)
    assert run_job(folder)["status"] == "completed"
    with pytest.raises(ValueError):
        cancel_job(folder)
    with pytest.raises(ValueError):
        resume_job(folder)


def test_soft_deadline_and_policy_deadline_do_not_apply_commands(project, tmp_path, monkeypatch):
    folder = tmp_path / "job"
    initial = create_job(job_spec(project), folder)
    paused = run_job(folder, deadline_s=1e-6)
    assert paused["status"] == "paused" and paused["episode"] == initial["episode"]
    import atomic_model_workspace.jobs as jobs

    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(jobs.time, "monotonic", lambda: next(clock))
    failed = run_job(folder)
    assert (
        failed["status"] == "failed"
        and failed["reason"] == "policy_deadline"
        and failed["episode"] == initial["episode"]
    )


def test_numerical_rejection_is_not_success_or_fabricated_trajectory(
    project, tmp_path, monkeypatch
):
    from atomic_model_runtime.episode import EOSEpisode, EpisodeRejected

    folder = tmp_path / "job"
    initial = create_job(job_spec(project), folder)

    def reject(*args, **kwargs):
        raise EpisodeRejected("admission_test", "reject")

    monkeypatch.setattr(EOSEpisode, "step", reject)
    result = run_job(folder)
    assert (
        result["status"] == "rejected"
        and result["episode"] == initial["episode"]
        and len(result["rows"]) == 1
    )
    assert not job_metrics(result)["completed"] and not job_metrics(result)["final_target_met"]


def test_checkpoint_publication_failure_leaves_exact_old_head(project, tmp_path, monkeypatch):
    folder = tmp_path / "job"
    initial = create_job(job_spec(project), folder)
    original = common.os.replace

    def fail_head(src, dst):
        if Path(dst).name == "HEAD.json":
            raise OSError("injected publication failure")
        return original(src, dst)

    with monkeypatch.context() as m:
        m.setattr(common.os, "replace", fail_head)
        with pytest.raises(OSError):
            run_job(folder, max_decisions=1)
    assert load_job(folder) == initial
    assert len(list((folder / "revisions").glob("*.json"))) == 2  # orphan not HEAD
    assert run_job(folder)["status"] == "completed"


def test_job_writer_lock_and_corrupt_head_are_rejected(project, tmp_path):
    folder = tmp_path / "job"
    create_job(job_spec(project), folder)
    with Lock(folder):
        with pytest.raises(ValueError):
            run_job(folder)
    change(folder / "HEAD.json", lambda p: p.update(sha256="0" * 64))
    with pytest.raises(ValueError):
        load_job(folder)


@pytest.mark.parametrize(
    "defect", ["schema", "source", "config", "trajectory", "policy_trace", "completion"]
)
def test_rehashed_semantic_checkpoint_corruption_rejected(project, tmp_path, defect):
    folder = tmp_path / "job"
    create_job(job_spec(project), folder)
    run_job(folder, max_decisions=1)
    head = read(folder / "HEAD.json")
    path = folder / head["checkpoint"]
    cp = read(path)
    s = cp["payload"]
    if defect == "schema":
        s["schema_version"] = 2
    elif defect == "source":
        s["provenance"]["workspace_sha256"] = "0" * 64
    elif defect == "config":
        s["spec"]["scenario"]["config"]["preset"] = "gravity"
        s["spec_sha256"] = digest(s["spec"])
    elif defect == "trajectory":
        s["rows"][-1]["ledger"]["energy_residual_j"] += 1
    elif defect == "policy_trace":
        s["rows"][-1]["policy_input"]["channels"][0]["value"] += 1
    elif defect == "completion":
        s["status"] = "completed"
    cp["sha256"] = digest(s)
    head["sha256"] = cp["sha256"]
    atomic_json(path, cp)
    atomic_json(folder / "HEAD.json", head)
    with pytest.raises(ValueError):
        load_job(folder)


def test_metrics_units_integrals_ledgers_and_failure_visibility(project, tmp_path):
    create_campaign(project, tmp_path / "campaign")
    state = load_campaign(tmp_path / "campaign")
    jobpath = tmp_path / "campaign" / state["entries"][0]["job"]
    done = run_job(jobpath)
    metric = job_metrics(done)
    rows = done["rows"]
    expected = sum(
        (b["time_s"] - a["time_s"]) * sum(abs(v) for v in b["heat_power_w"].values())
        for a, b in zip(rows, rows[1:])
    )
    assert metric["absolute_heat_input_j"] == expected and metric["completed_time_s"] == 0.03
    for row in rows:
        ledger = row["ledger"]
        assert (
            ledger["energy_residual_j"]
            == ledger["total_energy_change_j"]
            - ledger["external_heat_j"]
            + ledger["exported_drag_heat_j"]
        )
    report = export_report(tmp_path / "campaign", tmp_path / "report")
    assert len(report["jobs"]) == 6 and sum(j["metrics"]["completed"] for j in report["jobs"]) == 1
    manifest = read(tmp_path / "report/manifest.json")
    assert manifest["units"]["absolute_heat_input_j"] == "J"
    for p, h in manifest["artifacts"].items():
        assert hashlib.sha256((tmp_path / "report" / p).read_bytes()).hexdigest() == h


def test_campaign_spawn_pause_resume_and_report(project, tmp_path):
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    first = run_campaign(folder, max_jobs=1)
    assert first["status"] == "paused"
    result = run_campaign(folder, resume=True)
    assert result["status"] == "completed"
    report = campaign_report(folder)
    assert all(j["metrics"]["completed"] for j in report["jobs"])
    assert all(len(j["attempts"]) == 1 for j in report["jobs"])
    assert compare_campaigns(folder, folder)["comparable"]
    with pytest.raises(ValueError):
        cancel_campaign(folder)


def test_hard_worker_timeout_continues_and_resume_preserves_accepted_prefix(project, tmp_path):
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    state = run_campaign(folder, worker_timeout_s=0.001)
    assert state["status"] == "paused"
    assert all(e["attempts"][0]["outcome"] == "worker_deadline" for e in state["entries"])
    for entry in state["entries"]:
        assert load_job(folder / entry["job"])["episode"]["steps"] == 0
        assert not (folder / entry["job"] / ".lock").exists()
    result = run_campaign(folder, resume=True)
    assert result["status"] == "completed" and all(
        len(e["attempts"]) == 2 for e in result["entries"]
    )


def test_campaign_cancel_before_worker_spawn_then_resume(project, tmp_path):
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    cancel_campaign(folder)
    state = run_campaign(folder)
    assert state["status"] == "cancelled" and all(not e["attempts"] for e in state["entries"])
    assert run_campaign(folder, resume=True)["status"] == "completed"


def test_compare_rejects_different_numerics_and_keeps_unfinished_jobs(project, tmp_path):
    create_campaign(project, tmp_path / "a")
    change(
        project.parent / "scenarios/mixed_dev.json", lambda p: p["config"].update(max_step_s=0.001)
    )
    create_campaign(project, tmp_path / "b")
    with pytest.raises(ValueError):
        compare_campaigns(tmp_path / "a", tmp_path / "b")
    same = compare_campaigns(tmp_path / "a", tmp_path / "a")
    assert (
        all(not j["metrics"]["completed"] for j in same["left"]["jobs"]) and same["ranking"] is None
    )


def test_archive_roundtrip_current_checkpoint_and_deterministic_bytes(project, tmp_path):
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    entry = load_campaign(folder)["entries"][0]
    run_job(folder / entry["job"], max_decisions=1)
    a = export_bundle(project, tmp_path / "a.zip", campaign=folder)
    b = export_bundle(project, tmp_path / "b.zip", campaign=folder)
    assert a == b and (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    restored = import_bundle(tmp_path / "a.zip", tmp_path / "imported")
    assert load_project(restored["project"]) == load_project(project)
    assert load_campaign(restored["campaign"]) == load_campaign(folder)
    left = run_job(folder / entry["job"])
    right = run_job(Path(restored["campaign"]) / entry["job"])
    assert left["episode"] == right["episode"]
    with pytest.raises(FileExistsError):
        import_bundle(tmp_path / "a.zip", tmp_path / "imported")
    with pytest.raises(FileExistsError):
        export_bundle(project, tmp_path / "a.zip")


@pytest.mark.parametrize(
    "defect",
    ["traversal", "absolute", "duplicate", "symlink", "tamper", "extra", "collision", "schema"],
)
def test_hostile_archive_rejected_without_output(project, tmp_path, defect):
    archive = tmp_path / "good.zip"
    export_bundle(project, archive)
    with zipfile.ZipFile(archive) as z:
        files = {i.filename: z.read(i) for i in z.infolist()}
    if defect == "tamper":
        files["project/project.json"] += b" "
    elif defect == "extra":
        files["evil.py"] = b"print(1)"
    elif defect == "schema":
        manifest = json.loads(files["MANIFEST.json"])
        manifest["schema_version"] = 2
        files["MANIFEST.json"] = json.dumps(manifest).encode()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        if defect in ("traversal", "absolute", "collision"):
            z.writestr(
                {
                    "traversal": "../escape.json",
                    "absolute": "/tmp/escape.json",
                    "collision": "project",
                }[defect],
                b"{}",
            )
        elif defect == "duplicate":
            with pytest.warns(UserWarning):
                z.writestr("project/project.json", files["project/project.json"])
        elif defect == "symlink":
            info = zipfile.ZipInfo("link.json")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, b"project/project.json")
    with pytest.raises(ValueError):
        import_bundle(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_project_only_archive_no_runtime_execution(project, tmp_path):
    export_bundle(project, tmp_path / "project.zip")
    result = verify_bundle(tmp_path / "project.zip")
    assert not result["manifest"]["contains_campaign"]
    assert all(n.endswith(".json") for n in result["files"])


def crash_after_decision(folder):
    """Real isolated process failure after a durable accepted decision."""
    run_job(folder, max_decisions=1)
    raise SystemExit(17)


def test_worker_exit_retains_prefix_continues_other_jobs_and_can_resume(
    project, tmp_path, monkeypatch
):
    import atomic_model_workspace.campaign as campaign

    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    with monkeypatch.context() as m:
        m.setattr(campaign, "worker", crash_after_decision)
        state = run_campaign(folder)
    assert state["status"] == "paused"
    assert all(e["attempts"][0]["exitcode"] == 17 for e in state["entries"])
    assert all(load_job(folder / e["job"])["episode"]["steps"] == 1 for e in state["entries"])
    assert run_campaign(folder, resume=True)["status"] == "completed"


def test_archive_preserves_pending_cancellation(project, tmp_path):
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    entry = load_campaign(folder)["entries"][0]
    cancel_job(folder / entry["job"])
    cancel_campaign(folder)
    export_bundle(project, tmp_path / "cancelled.zip", campaign=folder)
    restored = import_bundle(tmp_path / "cancelled.zip", tmp_path / "restored")
    assert (Path(restored["campaign"]) / "CANCEL.json").exists()
    assert (Path(restored["campaign"]) / entry["job"] / "CANCEL.json").exists()
    assert run_campaign(restored["campaign"])["status"] == "cancelled"


def test_baseline_agent_tools_and_job_share_exact_episode(project, tmp_path):
    from atomic_agent_tools import AgentToolServer

    spec = job_spec(project)
    create_job(spec, tmp_path / "job")
    complete = run_job(tmp_path / "job")
    server = AgentToolServer()
    view = server.invoke(
        "create_eos_episode", dict(session_id="baseline", config=spec["scenario"]["config"])
    )
    for row in complete["rows"][1:]:
        view = server.invoke(
            "step_episode",
            dict(
                session_id="baseline",
                expected_revision=view["revision"],
                observation_id=view["observation_id"],
                action=row["heat_power_w"],
            ),
        )
    assert server.export_session("baseline")["payload"]["episode"] == complete["episode"]


def test_compare_rejects_execution_environment_mismatch(project, tmp_path):
    create_campaign(project, tmp_path / "a")
    create_campaign(project, tmp_path / "b")
    change(
        tmp_path / "b/campaign.json",
        lambda p: p["execution_environment"].update(cpu_model="different-machine"),
    )
    with pytest.raises(ValueError, match="environment"):
        compare_campaigns(tmp_path / "a", tmp_path / "b")


def test_archive_expanded_size_bound(project, tmp_path, monkeypatch):
    import atomic_model_workspace.bundle as bundle

    export_bundle(project, tmp_path / "a.zip")
    monkeypatch.setattr(bundle, "MAX_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        import_bundle(tmp_path / "a.zip", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_validate_initialize_pause_inspect_and_data_export(project, tmp_path):
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "atomic_model_workspace", *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert cli("validate", project)["jobs"] == 6
    cli("initialize", project, "--output", tmp_path / "campaign")
    assert len(cli("inspect", tmp_path / "campaign")["jobs"]) == 6
    job = load_campaign(tmp_path / "campaign")["entries"][0]["job"]
    assert cli("job-run", tmp_path / "campaign" / job, "--max-decisions", "1")["status"] == "paused"
    cli("export", project, "--campaign", tmp_path / "campaign", "--output", tmp_path / "a.zip")
    assert cli("verify", tmp_path / "a.zip")["contains_campaign"]
    assert cli("import", tmp_path / "a.zip", "--output", tmp_path / "imported")["campaign"]


def test_explicit_job_pause_marker_survives_archive_and_resume(project, tmp_path):
    from atomic_model_workspace.jobs import pause_job

    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    entry = load_campaign(folder)["entries"][0]
    path = folder / entry["job"]
    before = load_job(path)["episode"]
    pause_job(path)
    paused = run_job(path)
    assert (
        paused["status"] == "paused"
        and paused["reason"] == "pause_requested"
        and paused["episode"] == before
    )
    export_bundle(project, tmp_path / "paused.zip", campaign=folder)
    restored = import_bundle(tmp_path / "paused.zip", tmp_path / "restore")
    other = Path(restored["campaign"]) / entry["job"]
    assert (other / "PAUSE.json").exists()
    resume_job(other)
    assert run_job(other)["status"] == "completed"
    with pytest.raises(ValueError):
        pause_job(other)


def test_campaign_explicit_pause_before_spawn(project, tmp_path):
    from atomic_model_workspace.campaign import pause_campaign

    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    pause_campaign(folder)
    state = run_campaign(folder)
    assert state["status"] == "paused" and all(not e["attempts"] for e in state["entries"])
    assert run_campaign(folder, resume=True, max_jobs=1)["status"] == "paused"


def test_active_supervisor_pause_keeps_durable_decision_and_resumes(project, tmp_path):
    import threading
    import time

    from atomic_model_workspace.campaign import pause_campaign

    change(
        project,
        lambda p: p.update(
            scenarios={"mixed_dev": p["scenarios"]["mixed_dev"]},
            policies={"hold": p["policies"]["hold"]},
        ),
    )
    change(
        project.parent / "scenarios/mixed_dev.json",
        lambda p: p["config"].update(episode_horizon_s=0.2),
    )
    folder = tmp_path / "campaign"
    create_campaign(project, folder)
    entry = load_campaign(folder)["entries"][0]
    requested = []
    stop = threading.Event()

    def request_after_commit():
        end = time.monotonic() + 30
        while not stop.is_set() and time.monotonic() < end:
            if load_job(folder / entry["job"])["episode"]["steps"] >= 1:
                pause_campaign(folder)
                requested.append(True)
                return
            stop.wait(0.01)

    thread = threading.Thread(target=request_after_commit, daemon=True)
    thread.start()
    try:
        state = run_campaign(folder)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert requested and state["status"] == "paused"
    accepted = load_job(folder / entry["job"])["episode"]["steps"]
    assert 1 <= accepted < 20
    assert run_campaign(folder, resume=True)["status"] == "completed"
