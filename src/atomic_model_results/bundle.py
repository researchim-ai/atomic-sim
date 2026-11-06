"""Bounded data-only result bundles. Integrity is not publisher authentication."""

import hashlib
import shutil
from pathlib import Path

from atomic_model_reference.bundle import load_bundle
from atomic_model_reference.replay import FILES, replay_report
from atomic_model_results import analysis
from atomic_model_runtime.episode import EOSEpisode
from atomic_model_study import evaluate, store
from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, number, read
from atomic_model_workspace.jobs import _row, load_job

MAX_RUNS = 32
MAX_FILES = 2048
MAX_TOTAL = 512 * 1024**2
MAX_FILE = 32 * 1024**2
STATUSES = {
    "completed",
    "paused",
    "cancelled",
    "timed_out",
    "rss_limit",
    "failed",
    "interrupted",
    "rejected",
}


def source_identity():
    return dict(
        results_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        study=store.source_identity(),
    )


def _selected(ids):
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_RUNS:
        raise ValueError("Select 1..32 evaluations")
    for id in ids:
        evaluate.evaluation_path(".", id)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation ID")
    return sorted(ids)


def _status(state, p, id):
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or type(state.get("schema_version")) is not int
        or state.get("evaluation_id") != id
        or state.get("study_checkpoint_sha256") != p["study_checkpoint_sha256"]
        or type(state.get("study_revision")) is not int
        or state.get("study_revision") != p["study"]["revision"]
        or state.get("binding_id") != p["binding_id"]
        or state.get("status") not in STATUSES
        or type(state.get("attempt")) is not int
        or not 0 <= state["attempt"] < 32
    ):
        raise ValueError("Evaluation status/input mismatch or unfinished worker")
    if state.get("comparison_passed") is not None and type(state["comparison_passed"]) is not bool:
        raise ValueError("Invalid comparison outcome")
    expected = "attempts/" + str(state["attempt"]) + "/report"
    if state.get("report") is not None and state["report"] != expected:
        raise ValueError("Noncanonical report path")
    if state["status"] == "completed" and state.get("report") is None:
        raise ValueError("Missing completed comparison")
    if state["status"] != "completed" and (
        state.get("report") is not None or state.get("comparison_passed") is not None
    ):
        raise ValueError("Incomplete evaluation cannot carry metrics")


def _job(folder, state):
    (folder / "revisions").mkdir(parents=True)
    atomic_json(folder / "revisions/snapshot.json", dict(payload=state, sha256=digest(state)))
    atomic_json(
        folder / "HEAD.json",
        dict(schema_version=1, checkpoint="revisions/snapshot.json", sha256=digest(state)),
    )


def _copy_report(source, target):
    target.mkdir(parents=True)
    for name in sorted(FILES | {"manifest.json"}):
        shutil.copy2(local_file(source, name), target / name)


def _write_manifest(folder):
    files = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
    if (
        len(files) > MAX_FILES
        or any(p.stat().st_size > MAX_FILE for p in files.values())
        or sum(p.stat().st_size for p in files.values()) > MAX_TOTAL
    ):
        raise ValueError("Result bundle resource limit")
    atomic_json(
        folder / "manifest.json",
        dict(
            schema_version=1,
            source=source_identity(),
            artifacts={
                name: hashlib.sha256(p.read_bytes()).hexdigest()
                for name, p in sorted(files.items())
            },
            validated=False,
            plant_referenced=False,
        ),
    )


def export(study, ids, output):
    ids = _selected(ids)
    output = Path(output)
    with Lock(study):
        current, _, _ = store.load(study)
        if (Path(study) / "EVALUATION.lock").exists():
            raise ValueError("Wait until the study worker releases its lock")
        output.mkdir(parents=True, exist_ok=False)
        try:
            object_ids = {b["object_sha256"] for b in current["payload"]["references"].values()}
            for id in ids:
                p, path = evaluate.load_input(study, id)
                state = read(local_file(path, "status.json"))
                _status(state, p, id)
                object_ids.update(b["object_sha256"] for b in p["study"]["references"].values())
                if len(object_ids) > 64:
                    raise ValueError("Result bundle reference object limit")
                target = output / "evaluations" / id
                target.mkdir(parents=True)
                shutil.copy2(local_file(path, "input.json"), target / "input.json")
                atomic_json(target / "status.json", state)
                if (path / "job/HEAD.json").exists():
                    job = load_job(path / "job")
                    _job(target / "job", job)
                    (target / "trajectory.csv").write_bytes(analysis.trajectory(job))
                if state["report"] is not None:
                    _, report, _, _ = evaluate.result(study, id)
                    _copy_report(report, target / state["report"])
            for sha in sorted(object_ids):
                store._copy_bundle(store.object_path(study, sha), store.object_path(output, sha))
            atomic_json(
                output / "bundle.json",
                dict(
                    schema_version=1,
                    format="atomic-study-results-v1",
                    source=source_identity(),
                    study=current["payload"],
                    study_checkpoint_sha256=current["sha256"],
                    evaluations=ids,
                ),
            )
            # Reconstruct and validate everything before publishing derived summaries.
            runs = _runs(output, read(output / "bundle.json"))
            atomic_json(output / "summary.json", analysis.summary(runs, current["sha256"]))
            _write_manifest(output)
            verify(output)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        schema_version=1,
        bundle=str(output / "bundle.json"),
        evaluations=ids,
        summary_sha256=digest(read(output / "summary.json")),
        validated=False,
        plant_referenced=False,
    )


def audit_native(job):
    """Reintegrate accepted actions and compare every retained row and final CP."""
    env = EOSEpisode(job["spec"]["scenario"]["config"])
    if _row(env) != job["rows"][0]:
        raise ValueError("Native initial trajectory row mismatch")
    for row in job["rows"][1:]:
        commands = row["heat_power_w"]
        env.step([commands[name] for name in env.config.action_channels])
        if _row(env, commands, row["policy_input"], row["policy_wall_s"]) != row:
            raise ValueError("Native replay trajectory row mismatch")
    if env.export_episode() != job["episode"]:
        raise ValueError("Native replay final checkpoint mismatch")


def _study(value):
    keys(
        value,
        (
            "schema_version",
            "revision",
            "parent_sha256",
            "project",
            "project_sha256",
            "references",
            "source",
            "note",
            "operation",
        ),
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["revision"]) is not int
        or not 0 <= value["revision"] < 64
        or value["project_sha256"] != digest(value["project"])
        or value["source"] != store.source_identity()
        or not isinstance(value["note"], str)
        or len(value["note"]) > 512
        or value["operation"] not in ("create", "edit", "attach", "detach", "restore")
    ):
        raise ValueError("Study schema/source/project identity mismatch")
    if value["revision"] == 0:
        if value["parent_sha256"] is not None or value["operation"] != "create":
            raise ValueError("Invalid study origin")
    else:
        store._hash(value["parent_sha256"])
        if value["operation"] == "create":
            raise ValueError("Invalid study operation")


def _runs(folder, spec):
    runs = {}
    for id in _selected(spec["evaluations"]):
        p, path = evaluate.load_input(folder, id)
        study = p["study"]
        _study(study)
        if p["wall_limit_s"] != study["project"]["limits"]["job_timeout_s"]:
            raise ValueError("Evaluation wall budget differs from study limits")
        number(p["wall_limit_s"], 0.1, 3600, "wall_limit_s")
        if (
            type(p["rss_limit_bytes"]) is not int
            or not 32 * 1024**2 <= p["rss_limit_bytes"] <= 16 * 1024**3
            or not isinstance(p["cpus"], list)
            or len(p["cpus"]) != 1
            or type(p["cpus"][0]) is not int
            or not 0 <= p["cpus"][0] <= 65535
            or type(p["parent_pid"]) is not int
            or p["parent_pid"] <= 0
        ):
            raise ValueError("Invalid retained worker budget/identity")
        state = read(local_file(path, "status.json"))
        _status(state, p, id)
        binding = p["study"]["references"][p["binding_id"]]
        ref, _ = load_bundle(store.object_path(folder, binding["object_sha256"]))
        job = load_job(path / "job") if (path / "job/HEAD.json").exists() else None
        if job:
            audit_native(job)
            project = p["study"]["project"]
            if job["spec"] != dict(
                schema_version=1,
                scenario=project["scenarios"][binding["scenario_id"]],
                policy=project["policies"][binding["policy_id"]],
                decision_timeout_s=project["limits"]["decision_timeout_s"],
            ):
                raise ValueError("Retained job does not match pinned study policy/scenario")
            if local_file(path, "trajectory.csv").read_bytes() != analysis.trajectory(job):
                raise ValueError("Trajectory differs from native accepted job")
        report = None
        if state["report"] is not None:
            report, _, _, _ = evaluate.result(folder, id)
            if state["comparison_passed"] is not report["passed"]:
                raise ValueError("Outcome differs from replayed report")
        runs[id] = (p, state, job, report, ref)
    return runs


def verify(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(manifest, ("schema_version", "source", "artifacts", "validated", "plant_referenced"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source"] != source_identity()
        or manifest["validated"] is not False
        or manifest["plant_referenced"] is not False
        or not isinstance(manifest["artifacts"], dict)
        or not 2 <= len(manifest["artifacts"]) <= MAX_FILES
    ):
        raise ValueError("Result bundle schema/source/inventory mismatch")
    total = local_file(folder, "manifest.json").stat().st_size
    for name, sha in manifest["artifacts"].items():
        if name == "manifest.json":
            raise ValueError("Manifest cannot inventory itself")
        with local_file(folder, name).open("rb") as stream:
            raw = stream.read(MAX_FILE + 1)
        total += len(raw)
        if len(raw) > MAX_FILE or total > MAX_TOTAL or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Result bundle bytes/hash limit mismatch")
    spec = read(local_file(folder, "bundle.json"))
    keys(
        spec,
        ("schema_version", "format", "source", "study", "study_checkpoint_sha256", "evaluations"),
    )
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["format"] != "atomic-study-results-v1"
        or spec["source"] != source_identity()
        or digest(spec["study"]) != spec["study_checkpoint_sha256"]
    ):
        raise ValueError("Result study/source identity mismatch")
    _study(spec["study"])
    store.validate_state(folder, spec["study"]["project"], spec["study"]["references"])
    runs = _runs(folder, spec)
    expected = {"bundle.json", "summary.json"}
    objects = {b["object_sha256"] for b in spec["study"]["references"].values()}
    for id, (p, state, job, report, ref) in runs.items():
        prefix = "evaluations/" + id + "/"
        expected.update(prefix + name for name in ["input.json", "status.json"])
        objects.update(b["object_sha256"] for b in p["study"]["references"].values())
        if job:
            expected.update(
                prefix + name
                for name in ["job/HEAD.json", "job/revisions/snapshot.json", "trajectory.csv"]
            )
        if report:
            expected.update(
                prefix + state["report"] + "/" + name for name in FILES | {"manifest.json"}
            )
    if len(objects) > 64:
        raise ValueError("Result reference object limit")
    for sha in objects:
        _, m = load_bundle(store.object_path(folder, sha))
        expected.update(
            "objects/" + sha + "/" + name for name in (*m["artifacts"], "manifest.json")
        )
    if set(manifest["artifacts"]) != expected:
        raise ValueError("Missing or undeclared result artifacts")
    actual = {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != expected | {"manifest.json"}:
        raise ValueError("Unexpected files in result bundle")
    result = analysis.summary(runs, spec["study_checkpoint_sha256"])
    if read(local_file(folder, "summary.json")) != result:
        raise ValueError("Recomputed result summary differs")
    return result


def pair(folder, left, right, channel):
    verify(folder)
    spec = read(Path(folder) / "bundle.json")
    return analysis.pair(_runs(Path(folder), spec), left, right, channel)


def replay(folder, output):
    result = verify(folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(output / "summary.json", result)
        manifest = read(Path(folder) / "manifest.json")
        # Independent per-report comparator replay also rebuilds all metrics.
        for id, state in result["entries"].items():
            if state["status"] == "completed":
                status = read(Path(folder) / "evaluations" / id / "status.json")
                replay_report(Path(folder) / "evaluations" / id / status["report"], output / id)
        atomic_json(
            output / "replay.json",
            dict(
                schema_version=1,
                replay_exact=True,
                manifest_sha256=digest(manifest),
                evaluations=len(result["entries"]),
                validated=False,
                plant_referenced=False,
            ),
        )
    except BaseException:
        shutil.rmtree(output)
        raise
    return dict(
        schema_version=1,
        replay_exact=True,
        evaluations=len(result["entries"]),
        validated=False,
        plant_referenced=False,
    )
