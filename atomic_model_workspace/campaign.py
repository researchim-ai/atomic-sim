"""Sequential spawn-isolated campaigns with durable per-decision job recovery.

One worker at a time: no throughput/scaling claim. Hard deadlines include spawn.
"""

import multiprocessing
import time
from pathlib import Path

from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, number, read
from atomic_model_workspace.jobs import (
    TERMINAL,
    clear_exited_worker_lock,
    create_job,
    interrupt_job,
    load_job,
    resume_job,
    worker,
)
from atomic_model_workspace.project import (
    execution_environment,
    load_project,
    normalize,
    provenance,
)


def _normalized(value):
    return normalize(value, value["scenarios"], value["policies"])


def create_campaign(project_path, output):
    project = load_project(project_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "jobs").mkdir()
    entries = []
    for sid, scenario in project["scenarios"].items():
        for pid, policy in project["policies"].items():
            relative = "jobs/" + digest([sid, pid])
            create_job(
                dict(
                    schema_version=1,
                    scenario=scenario,
                    policy=policy,
                    decision_timeout_s=project["limits"]["decision_timeout_s"],
                ),
                output / relative,
            )
            entries.append(dict(scenario_id=sid, policy_id=pid, job=relative, attempts=[]))
    state = dict(
        schema_version=1,
        project=project,
        project_sha256=digest(project),
        execution_environment=execution_environment(),
        provenance=provenance(),
        status="initialized",
        entries=entries,
    )
    atomic_json(output / "campaign.json", state)
    return state


def load_campaign(output):
    output = Path(output)
    state = read(output / "campaign.json")
    keys(
        state,
        (
            "schema_version",
            "project",
            "project_sha256",
            "execution_environment",
            "provenance",
            "status",
            "entries",
        ),
    )
    project = _normalized(state["project"])
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or digest(project) != state["project_sha256"]
        or project != state["project"]
        or state["provenance"] != provenance()
        or state["status"]
        not in (
            "initialized",
            "paused",
            "cancelled",
            "completed",
            "completed_with_failures",
            "running",
        )
    ):
        raise ValueError("Campaign identity/lifecycle mismatch")
    expected = {(s, p) for s in project["scenarios"] for p in project["policies"]}
    seen = set()
    if not isinstance(state["entries"], list):
        raise ValueError("Malformed campaign entries")
    for entry in state["entries"]:
        keys(entry, ("scenario_id", "policy_id", "job", "attempts"))
        pair = (entry["scenario_id"], entry["policy_id"])
        if (
            pair not in expected
            or pair in seen
            or entry["job"] != "jobs/" + digest(list(pair))
            or not isinstance(entry["attempts"], list)
        ):
            raise ValueError("Campaign entry identity mismatch")
        seen.add(pair)
        # local_file also excludes symlinks along all path components.
        job = load_job(local_file(output, entry["job"] + "/HEAD.json").parent)
        expected_spec = dict(
            schema_version=1,
            scenario=project["scenarios"][pair[0]],
            policy=project["policies"][pair[1]],
            decision_timeout_s=project["limits"]["decision_timeout_s"],
        )
        if job["spec"] != expected_spec:
            raise ValueError("Campaign job specification mismatch")
    if seen != expected:
        raise ValueError("Campaign is missing jobs")
    return state


def cancel_campaign(output):
    state = load_campaign(output)
    if state["status"] in ("completed", "completed_with_failures"):
        raise ValueError("Terminal campaign cannot be cancelled")
    atomic_json(Path(output) / "CANCEL.json", dict(schema_version=1, request="cancel"))
    return dict(requested=True)


def pause_campaign(output):
    state = load_campaign(output)
    if state["status"] in ("completed", "completed_with_failures"):
        raise ValueError("Terminal campaign cannot be paused")
    atomic_json(Path(output) / "PAUSE.json", dict(schema_version=1, request="pause"))
    return dict(requested=True)


def _stop(process):
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)
    if process.is_alive():
        raise RuntimeError("Worker could not be stopped; no lock recovery attempted")


def run_campaign(output, *, max_jobs=None, resume=False, worker_timeout_s=None):
    if max_jobs is not None and (type(max_jobs) is not int or not 1 <= max_jobs <= 1024):
        raise ValueError("Invalid campaign job budget")
    if worker_timeout_s is not None:
        number(worker_timeout_s, 0.001, 3600, "worker timeout")
    output = Path(output)
    with Lock(output):
        state = load_campaign(output)
        if resume:
            (output / "CANCEL.json").unlink(missing_ok=True)
            (output / "PAUSE.json").unlink(missing_ok=True)
        elif state["status"] == "cancelled":
            return state
        state["status"] = "running"
        atomic_json(output / "campaign.json", state)
        processed = 0
        for entry in state["entries"]:
            jobpath = output / entry["job"]
            job = load_job(jobpath)
            if job["status"] in TERMINAL:
                continue
            if (output / "CANCEL.json").exists():
                state["status"] = "cancelled"
                break
            if (output / "PAUSE.json").exists():
                state["status"] = "paused"
                break
            if max_jobs is not None and processed >= max_jobs:
                state["status"] = "paused"
                break
            if job["status"] == "cancelled" or (jobpath / "PAUSE.json").exists():
                if not resume:
                    state["status"] = "paused"
                    break
                resume_job(jobpath)
            initial_time = job["rows"][-1]["time_s"]
            process = multiprocessing.get_context("spawn").Process(
                target=worker, args=(str(jobpath),)
            )
            limit = (
                worker_timeout_s
                if worker_timeout_s is not None
                else state["project"]["limits"]["job_timeout_s"]
            )
            started = time.monotonic()
            outcome = "exited"
            process.start()
            try:
                while process.is_alive():
                    if (output / "CANCEL.json").exists():
                        # Cooperative marker first, bounded supervisor stop if needed.
                        atomic_json(
                            jobpath / "CANCEL.json", dict(schema_version=1, request="cancel")
                        )
                        outcome = "cancelled"
                        break
                    if (output / "PAUSE.json").exists():
                        atomic_json(jobpath / "PAUSE.json", dict(schema_version=1, request="pause"))
                        outcome = "pause_requested"
                        break
                    if time.monotonic() - started >= limit:
                        outcome = "worker_deadline"
                        break
                    process.join(timeout=0.02)
            finally:
                _stop(process)
                clear_exited_worker_lock(jobpath, process)
            elapsed = time.monotonic() - started
            if outcome != "exited" or process.exitcode != 0:
                interrupt_job(
                    jobpath,
                    outcome if outcome != "exited" else "worker_exit_" + str(process.exitcode),
                    cancelled=outcome == "cancelled",
                )
            job = load_job(jobpath)
            entry["attempts"].append(
                dict(
                    outcome=outcome,
                    execution_environment=execution_environment(),
                    exitcode=process.exitcode,
                    wall_s=elapsed,
                    worker_timeout_s=limit,
                    job_status=job["status"],
                    accepted_decisions=job["episode"]["steps"],
                )
            )
            attempt = entry["attempts"][-1]
            attempt["simulated_s"] = job["rows"][-1]["time_s"] - initial_time
            attempt["simulated_seconds_per_wall_second"] = (
                attempt["simulated_s"] / elapsed if elapsed else None
            )
            processed += 1
            atomic_json(output / "campaign.json", state)
            process.close()
            if outcome in ("cancelled", "pause_requested"):
                state["status"] = "cancelled" if outcome == "cancelled" else "paused"
                break
        else:
            statuses = [load_job(output / e["job"])["status"] for e in state["entries"]]
            state["status"] = (
                "completed"
                if all(s == "completed" for s in statuses)
                else "completed_with_failures"
                if all(s in TERMINAL for s in statuses)
                else "paused"
            )
        atomic_json(output / "campaign.json", state)
        return state
