"""Model/data CAS, leakage refusal, portable snapshots and real supervised comparisons."""

import copy
import hashlib
from pathlib import Path

import pytest

from atomic_model_study import evaluate, store
from atomic_model_study.manager import Manager
from atomic_model_workspace.bundle import _project_files
from atomic_model_workspace.common import atomic_json, read

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study(tmp_path):
    spec = read(ROOT / "examples/eos_reference/energy_reference.json")
    project = dict(
        schema_version=1,
        project_id="reference-study",
        model_id=spec["problem"]["model_id"],
        scenarios={
            "uniform_dev": dict(
                schema_version=1,
                config=spec["problem"]["config"],
                seed=17,
                split="dev",
                objective=dict(sensor="wall.w0", target_k=300, tolerance_k=1),
            )
        },
        policies={"hold": spec["problem"]["policy"]},
        limits=dict(decision_timeout_s=1, job_timeout_s=30),
    )
    for name, raw in _project_files(project).items():
        p = tmp_path / "input" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    folder = tmp_path / "study"
    store.create(tmp_path / "input/project/project.json", folder)
    return folder


def request(study, name="energy"):
    value, _, _ = store.load(study)
    spec = read(
        ROOT
        / "examples/eos_reference"
        / ("energy_reference.json" if name == "energy" else "reference.json")
    )
    raw = (ROOT / "examples/eos_reference" / spec["csv"]["path"]).read_bytes()
    return dict(
        schema_version=1,
        expected_revision=value["payload"]["revision"],
        expected_checkpoint_sha256=value["sha256"],
        id=name,
        scenario_id="uniform_dev",
        policy_id="hold",
        declaration=spec,
        note="bound reference",
    ), raw


def attach(study, name="energy"):
    req, raw = request(study, name)
    return store.attach_data(study, req, raw)


def edit_request(study, operation):
    v, _, _ = store.load(study)
    return dict(
        schema_version=1,
        expected_revision=v["payload"]["revision"],
        expected_project_sha256=v["payload"]["project_sha256"],
        operations=[operation],
        note="edited",
    )


def test_preview_no_writes_and_stale_cas(study):
    req, raw = request(study)
    before = {str(p.relative_to(study)): p.read_bytes() for p in study.rglob("*") if p.is_file()}
    preview = store.preview_data(study, req, raw)
    assert preview["committed"] is False
    assert before == {
        str(p.relative_to(study)): p.read_bytes() for p in study.rglob("*") if p.is_file()
    }
    store.attach_data(study, req, raw)
    head = (study / "HEAD.json").read_bytes()
    with pytest.raises(ValueError, match="Stale"):
        store.attach_data(study, req, raw)
    assert (study / "HEAD.json").read_bytes() == head
    assert store.inspect(study)["references"]["energy"]["role"] == "validation"


@pytest.mark.parametrize(
    "defect", ["csv_hash", "role", "problem", "policy", "path", "unknown", "schema", "note"]
)
def test_attach_refusals_are_atomic(study, defect):
    req, raw = request(study)
    if defect == "csv_hash":
        raw += b"\n"
    elif defect == "role":
        req["declaration"]["role"] = "test"
    elif defect == "problem":
        req["declaration"]["problem"]["config"]["episode_horizon_s"] = 0.04
    elif defect == "policy":
        req["policy_id"] = "missing"
    elif defect == "path":
        req["declaration"]["csv"]["path"] = "../outside.csv"
    elif defect == "unknown":
        req["extra"] = True
    elif defect == "schema":
        req["schema_version"] = True
    else:
        req["note"] = "x" * 513
    head = (study / "HEAD.json").read_bytes()
    with pytest.raises((ValueError, KeyError)):
        store.attach_data(study, req, raw)
    assert head == (study / "HEAD.json").read_bytes()
    assert not list((study / "objects").iterdir())


def test_bound_model_change_requires_detach_and_restore_includes_data(study):
    attach(study)
    op = dict(op="set_scenario_config", id="uniform_dev", value=dict(episode_horizon_s=0.04))
    req = edit_request(study, op)
    before = (study / "HEAD.json").read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        store.preview(study, req)
    with pytest.raises(ValueError, match="does not match"):
        store.edit(study, req)
    assert before == (study / "HEAD.json").read_bytes()
    v, _, _ = store.load(study)
    store.detach(study, expected_revision=1, expected_checkpoint_sha256=v["sha256"], id="energy")
    store.edit(study, edit_request(study, op))
    v, _, _ = store.load(study)
    store.restore(
        study,
        expected_revision=3,
        expected_project_sha256=v["payload"]["project_sha256"],
        target_revision=1,
    )
    view = store.inspect(study)
    assert view["revision"] == 4 and "energy" in view["references"]
    assert view["project"]["scenarios"]["uniform_dev"]["config"]["episode_horizon_s"] == 0.03


def test_duplicate_reference_id_refused(study):
    attach(study)
    req, raw = request(study)
    req["id"] = "another"
    with pytest.raises(ValueError, match="already bound"):
        store.attach_data(study, req, raw)


def test_export_import_exact_model_data_no_extra_files(study, tmp_path):
    attach(study)
    attach(study, "thermal")
    v, _, _ = store.load(study)
    obj = store.object_path(study, v["payload"]["references"]["energy"]["object_sha256"])
    (obj / "unrelated.tmp").write_text("do not export")
    result = store.export(study, tmp_path / "export")
    assert not list((tmp_path / "export").rglob("unrelated.tmp"))
    store.import_snapshot(result["study"], tmp_path / "copy")
    copied, _, metadata = store.load(tmp_path / "copy")
    assert copied["payload"]["project"] == v["payload"]["project"]
    assert copied["payload"]["references"] == v["payload"]["references"]
    assert set(metadata) == {"energy", "thermal"}
    assert copied["payload"]["revision"] == 0


@pytest.mark.parametrize(
    "defect", ["hash", "missing", "extra_inventory", "path", "symlink", "object"]
)
def test_snapshot_refusals_before_output(study, tmp_path, defect):
    attach(study)
    out = tmp_path / "snapshot"
    store.export(study, out)
    manifest = read(out / "manifest.json")
    name = "project/scenarios/uniform_dev.json"
    if defect == "hash":
        (out / name).write_text("{}")
    elif defect == "missing":
        del manifest["artifacts"][name]
    elif defect == "extra_inventory":
        (out / "extra").write_bytes(b"data")
        manifest["artifacts"]["extra"] = hashlib.sha256(b"data").hexdigest()
    elif defect == "path":
        manifest["artifacts"]["../outside"] = "0" * 64
    elif defect == "symlink":
        raw = (out / name).read_bytes()
        (out / name).unlink()
        (tmp_path / "other").write_bytes(raw)
        (out / name).symlink_to(tmp_path / "other")
    else:
        spec = read(out / "study.json")
        spec["references"]["energy"]["object_sha256"] = "0" * 64
        atomic_json(out / "study.json", spec)
        manifest["artifacts"]["study.json"] = hashlib.sha256(
            (out / "study.json").read_bytes()
        ).hexdigest()
    atomic_json(out / "manifest.json", manifest)
    with pytest.raises((ValueError, OSError)):
        store.import_snapshot(out / "study.json", tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def start(manager, study, binding="energy"):
    v, _, _ = store.load(study)
    return manager.start(
        binding, expected_revision=v["payload"]["revision"], expected_checkpoint_sha256=v["sha256"]
    )


def test_supervised_positive_negative_curves_replay_and_stale(study, tmp_path):
    attach(study)
    attach(study, "thermal")
    with Manager(study) as manager:
        state = start(manager, study)
        positive = manager.wait(state["evaluation_id"])
        assert positive["status"] == "completed", positive
        assert positive["comparison_passed"] is True
        summary = evaluate.summary(study, positive["evaluation_id"])
        assert summary["passed"] and not summary["validated"]
        curve = evaluate.curve(study, positive["evaluation_id"], "energy_residual")
        assert curve["decimated"] is False and curve["unit"] == "J"
        assert curve["reference"]["value"] == [0] * 4
        negative = manager.wait(start(manager, study, "thermal")["evaluation_id"])
        assert negative["status"] == "completed" and negative["comparison_passed"] is False
        req = edit_request(
            study,
            dict(
                op="set_objective",
                id="uniform_dev",
                value=dict(sensor="wall.w0", target_k=301, tolerance_k=1),
            ),
        )
        store.edit(study, req)
        assert manager.status(positive["evaluation_id"])["stale"]
        # Saved old results remain inspectable against their pinned input.
        assert evaluate.summary(study, positive["evaluation_id"])["passed"]
        with pytest.raises(ValueError):
            manager.resume(positive["evaluation_id"])
        (
            evaluate.evaluation_path(study, positive["evaluation_id"])
            / positive["report"]
            / "report.json"
        ).write_text("{}")
        with pytest.raises(ValueError, match="identity"):
            evaluate.summary(study, positive["evaluation_id"])


def test_cancel_resume_single_writer_and_lock_cleanup(study):
    attach(study)
    with Manager(study) as manager:
        state = start(manager, study)
        with Manager(study) as other:
            with pytest.raises(ValueError, match="owns|running"):
                start(other, study)
        cancelled = manager.cancel(state["evaluation_id"])
        assert cancelled["status"] == "cancelled"
        assert not (study / "EVALUATION.lock").exists()
        resumed = manager.resume(state["evaluation_id"])
        final = manager.wait(resumed["evaluation_id"])
        assert final["status"] == "completed", final
        assert final["attempt"] == 1 and final["comparison_passed"]


def test_deadline_terminates_worker(study):
    attach(study)
    # Job deadline includes cold child startup and native module imports.
    # Change only project limits by creating a new portable project first.
    project = copy.deepcopy(store.load(study)[0]["payload"]["project"])
    project["limits"]["job_timeout_s"] = 0.1
    root = study.parent / "short_input"
    for name, raw in _project_files(project).items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    short = study.parent / "short"
    store.create(root / "project/project.json", short)
    attach(short)
    with Manager(short) as manager:
        result = manager.wait(start(manager, short)["evaluation_id"])
        assert result["status"] == "timed_out"
        assert not (short / "EVALUATION.lock").exists()


def test_rss_limit_and_no_report_for_killed_worker(study):
    attach(study)
    with Manager(study, rss_limit_bytes=32 * 1024**2) as manager:
        state = manager.wait(start(manager, study)["evaluation_id"])
        assert state["status"] == "rss_limit", state
        assert state["comparison_passed"] is None and state["report"] is None
        with pytest.raises(ValueError, match="no complete"):
            evaluate.summary(study, state["evaluation_id"])


def test_source_guard_and_revision_corruption(study, monkeypatch):
    original = store.source_identity
    monkeypatch.setattr(store, "source_identity", lambda: dict(changed=True))
    with pytest.raises(ValueError, match="source"):
        store.load(study)
    monkeypatch.setattr(store, "source_identity", original)
    head = read(study / "HEAD.json")
    p = study / head["checkpoint"]
    value = read(p)
    value["payload"]["note"] = "corrupt"
    atomic_json(p, value)
    with pytest.raises(ValueError, match="hash"):
        store.load(study)


def test_corrupted_object_rejected_on_inspection(study):
    attach(study)
    value, _, _ = store.load(study)
    sha = value["payload"]["references"]["energy"]["object_sha256"]
    (store.object_path(study, sha) / "data.csv").write_text("broken")
    with pytest.raises(ValueError):
        store.inspect(study)


def test_recovery_after_supervisor_exit_and_no_live_pid_kill(study):
    import os
    import subprocess
    import sys
    import time

    attach(study)
    script = """
import os,sys
from atomic_model_study.manager import Manager
from atomic_model_study import store
from atomic_model_workspace.common import atomic_json
folder=sys.argv[1]
m=Manager(folder)
v=store.inspect(folder)
s=m.start("energy",expected_revision=v["revision"],expected_checkpoint_sha256=v["checkpoint_sha256"])
atomic_json(folder+"/crashed-evaluation.json",s)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(study)], check=True, timeout=15)
    state = read(study / "crashed-evaluation.json")
    with Manager(study) as manager:
        path = evaluate.evaluation_path(study, state["evaluation_id"])
        original = read(path / "status.json")
        altered = dict(original, pid=os.getpid())
        atomic_json(path / "status.json", altered)
        with pytest.raises(ValueError, match="still exists"):
            manager.recover(state["evaluation_id"])
        assert (study / "EVALUATION.lock").exists()
        atomic_json(path / "status.json", original)
        deadline = time.monotonic() + 10
        while True:
            try:
                recovered = manager.recover(state["evaluation_id"])
                break
            except ValueError as exc:
                if "still exists" not in str(exc) or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        assert recovered["status"] == "interrupted"
        resumed = manager.resume(state["evaluation_id"])
        assert manager.wait(resumed["evaluation_id"])["comparison_passed"] is True


def test_cli_create_attach_evaluate_and_inspect(study, tmp_path):
    import json
    import subprocess
    import sys

    req, _ = request(study)
    del req["declaration"]
    atomic_json(tmp_path / "request.json", req)

    def cli(*args, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "atomic_model_study", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == expected, result.stderr
        return json.loads(result.stdout)

    cli(
        "attach",
        study,
        tmp_path / "request.json",
        ROOT / "examples/eos_reference/energy_reference.json",
    )
    view = cli("inspect", study)
    state = cli(
        "evaluate",
        study,
        "energy",
        "--expected-revision",
        1,
        "--expected-sha256",
        view["checkpoint_sha256"],
    )
    assert state["comparison_passed"] is True
    assert cli("summary", study, state["evaluation_id"])["passed"]


def test_cross_role_physical_leakage_rejected_in_study(study):
    attach(study)
    scenario = copy.deepcopy(store.inspect(study)["project"]["scenarios"]["uniform_dev"])
    scenario["split"] = "train"
    store.edit(
        study, edit_request(study, dict(op="add_scenario", id="uniform_train", value=scenario))
    )
    req, raw = request(study)
    req.update(id="calibration", scenario_id="uniform_train")
    req["declaration"].update(reference_id="calibration-control", role="calibration")
    before = (study / "HEAD.json").read_bytes()
    with pytest.raises(ValueError, match="leak|role|split"):
        store.attach_data(study, req, raw)
    assert before == (study / "HEAD.json").read_bytes()


def test_cancel_after_accepted_decisions_preserves_prefix(study):
    import time

    from atomic_model_workspace.jobs import load_job

    store.edit(
        study,
        edit_request(
            study,
            dict(
                op="set_scenario_config",
                id="uniform_dev",
                value=dict(episode_horizon_s=0.06, action_dt_s=0.001),
            ),
        ),
    )
    req, raw = request(study)
    req["declaration"]["problem"]["config"] = store.inspect(study)["project"]["scenarios"][
        "uniform_dev"
    ]["config"]
    store.attach_data(study, req, raw)
    with Manager(study) as manager:
        state = start(manager, study)
        job = evaluate.evaluation_path(study, state["evaluation_id"]) / "job"
        deadline = time.monotonic() + 15
        while True:
            if (job / "HEAD.json").exists():
                prefix = load_job(job)
                if len(prefix["rows"]) >= 3:
                    break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        cancelled = manager.cancel(state["evaluation_id"])
        assert cancelled["status"] == "cancelled"
        accepted = load_job(job)["rows"]
        assert accepted[: len(prefix["rows"])] == prefix["rows"]
        manager.resume(state["evaluation_id"])
        final = manager.wait(state["evaluation_id"])
        assert final["status"] == "completed" and final["comparison_passed"] is True, final
        assert load_job(job)["rows"][: len(accepted)] == accepted
