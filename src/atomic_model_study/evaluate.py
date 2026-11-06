"""Pinned study evaluations; a finished comparison may legitimately fail criteria."""

import hashlib
import tempfile
from pathlib import Path

from atomic_model_reference.compare import compare_job, signal_value
from atomic_model_reference.metrics import grid, interpolate
from atomic_model_reference.replay import replay_report
from atomic_model_study import store
from atomic_model_workspace.common import atomic_json, digest, keys, local_file, read
from atomic_model_workspace.jobs import create_job, load_job, run_job


def evaluation_path(folder, id):
    if not isinstance(id, str) or len(id) != 32 or any(c not in "0123456789abcdef" for c in id):
        raise ValueError("Invalid evaluation ID")
    return Path(folder) / "evaluations" / id


def load_input(folder, id):
    path = evaluation_path(folder, id)
    value = read(local_file(path, "input.json"))
    keys(value, ("payload", "sha256"))
    p = value["payload"]
    keys(
        p,
        (
            "schema_version",
            "study_checkpoint_sha256",
            "study",
            "binding_id",
            "source",
            "cpus",
            "wall_limit_s",
            "rss_limit_bytes",
            "parent_pid",
        ),
    )
    if (
        type(p["schema_version"]) is not int
        or p["schema_version"] != 1
        or digest(p) != value["sha256"]
        or digest(p["study"]) != p["study_checkpoint_sha256"]
        or p["source"] != store.source_identity()
    ):
        raise ValueError("Evaluation input/source identity mismatch")
    store.validate_state(folder, p["study"]["project"], p["study"]["references"])
    if p["binding_id"] not in p["study"]["references"]:
        raise ValueError("Missing pinned reference binding")
    return p, path


def run(folder, id, attempt):
    if type(attempt) is not int or not 0 <= attempt < 32:
        raise ValueError("Invalid evaluation attempt")
    p, path = load_input(folder, id)
    binding = p["study"]["references"][p["binding_id"]]
    project = p["study"]["project"]
    scenario = project["scenarios"][binding["scenario_id"]]
    policy = project["policies"][binding["policy_id"]]
    job = path / "job"
    if not (job / "HEAD.json").exists():
        if job.exists():
            # A kill during first publication can leave a directory without
            # any accepted checkpoint. Preserve it; never treat it as progress.
            job.rename(path / "attempts" / str(attempt) / "incomplete-job")
        create_job(
            dict(
                schema_version=1,
                scenario=scenario,
                policy=policy,
                decision_timeout_s=project["limits"]["decision_timeout_s"],
            ),
            job,
        )
    result = run_job(job, deadline_s=p["wall_limit_s"])
    if result["status"] != "completed":
        out = dict(
            schema_version=1,
            study_checkpoint_sha256=p["study_checkpoint_sha256"],
            binding_id=p["binding_id"],
            attempt=attempt,
            job_status=result["status"],
            comparison_passed=None,
            reason=result["reason"],
            report=None,
        )
    else:
        report = path / "attempts" / str(attempt) / "report"
        comparison = compare_job(job, store.object_path(folder, binding["object_sha256"]), report)
        out = dict(
            schema_version=1,
            study_checkpoint_sha256=p["study_checkpoint_sha256"],
            binding_id=p["binding_id"],
            attempt=attempt,
            job_status=result["status"],
            comparison_passed=comparison["passed"],
            reason=comparison["reason"],
            report=str(report.relative_to(path)),
            report_sha256=hashlib.sha256((report / "report.json").read_bytes()).hexdigest(),
        )
    atomic_json(path / "attempts" / str(attempt) / "result.json", out)
    return out


def result(folder, id):
    p, path = load_input(folder, id)
    state = read(local_file(path, "status.json"))
    if state["status"] != "completed" or state.get("report") is None:
        raise ValueError("Evaluation has no complete comparison report")
    relative = state["report"]
    report = local_file(path, relative + "/report.json").parent
    if hashlib.sha256((report / "report.json").read_bytes()).hexdigest() != state["report_sha256"]:
        raise ValueError("Evaluation report identity mismatch")
    with tempfile.TemporaryDirectory(prefix="atomic-study-report-") as tmp:
        replay_report(report, Path(tmp) / "verified")
    data = read(report / "report.json")
    binding = p["study"]["references"][p["binding_id"]]
    if (
        data["reference_sha256"]
        != store.load_bundle(store.object_path(folder, binding["object_sha256"]))[1][
            "normalized_reference_sha256"
        ]
    ):
        raise ValueError("Report does not match pinned reference")
    job = load_job(path / "job")
    if data["job_payload_sha256"] != digest(job):
        raise ValueError("Report does not match accepted model job")
    return data, report, p, job


def summary(folder, id):
    data, _, p, job = result(folder, id)
    return dict(
        schema_version=1,
        evaluation_id=id,
        study_revision=p["study"]["revision"],
        study_checkpoint_sha256=p["study_checkpoint_sha256"],
        binding_id=p["binding_id"],
        reference_id=data["reference_id"],
        kind=data["evidence_kind"],
        role=data["role"],
        provenance=data["provenance"],
        passed=data["passed"],
        reason=data["reason"],
        channels=data["channels"],
        events=data["events"],
        criteria=data["criteria"],
        model=data["model"],
        source=data["source"],
        job_status=job["status"],
        job_reason=job["reason"],
        final_time_s=job["rows"][-1]["time_s"],
        global_ledger=job["rows"][-1]["ledger"],
        local_ledgers=job["rows"][-1]["circuits"],
        validated=False,
        plant_referenced=False,
    )


def curve(folder, id, channel):
    data, report, p, job = result(folder, id)
    if channel not in data["channels"]:
        raise ValueError("Unknown compared channel")
    ref = read(report / "reference_snapshot.json")
    binding = ref["channels"][channel]
    times = [r["time_s"] for r in job["rows"]]
    values = [signal_value(r, binding["signal"]) for r in job["rows"]]
    start, end = data["criteria"]["window_s"]
    mt = grid(times, start, end)
    rt = grid(ref["times_s"], start, end)
    return dict(
        schema_version=1,
        channel=channel,
        unit=binding["canonical_unit"],
        interpolation="piecewise linear",
        decimated=False,
        model=dict(time_s=mt, value=[interpolate(times, values, t) for t in mt]),
        reference=dict(
            time_s=rt, value=[interpolate(ref["times_s"], ref["values"][channel], t) for t in rt]
        ),
        validated=False,
        plant_referenced=False,
    )
