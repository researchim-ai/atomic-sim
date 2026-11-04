"""Quiescent checkpoint exports and native-recomputed rollout quality summaries."""

import hashlib
import shutil
from pathlib import Path

from atomic_model_reference.metrics import compare_curve
from atomic_model_results.bundle import audit_native
from atomic_model_rollouts import store
from atomic_model_workspace import jobs
from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, read


def summarize(p, natives):
    entries = {}
    for id, item in p["entries"].items():
        job = natives.get(id)
        quality = None
        if job:
            rows = job["rows"]
            objective = job["spec"]["scenario"]["objective"]
            times = [r["time_s"] for r in rows]
            values = [
                next(
                    c["value"]
                    for c in r["observation"]["channels"]
                    if c["name"] == objective["sensor"]
                )
                for r in rows
            ]
            if times[-1] > times[0]:
                quality = compare_curve(
                    times,
                    values,
                    [times[0], times[-1]],
                    [objective["target_k"]] * 2,
                    [times[0], times[-1]],
                )
        entries[id] = dict(
            scenario_id=item["scenario_id"],
            policy_id=item["policy_id"],
            seed=item["seed"],
            status=item["status"],
            reason=item["reason"],
            native_validated=job is not None,
            complete=bool(job and job["status"] == "completed" and item["status"] == "completed"),
            accepted_decisions=job["episode"]["steps"] if job else item["accepted_decisions"],
            time_s=job["rows"][-1]["time_s"] if job else None,
            quality=quality,
            quality_metric_units={
                k: (
                    "1"
                    if k in ("comparison_knots", "fraction_time_within_declared_uncertainty")
                    else "s"
                    if k.endswith("_s")
                    else "K s"
                    if "integral" in k
                    else "K"
                )
                for k in quality
            }
            if quality
            else None,
            final_global_ledger=job["rows"][-1]["ledger"] if job else None,
            final_local_ledgers=job["rows"][-1]["circuits"] if job else None,
            attempts=item["attempts"],
        )
    return dict(
        schema_version=1,
        project_sha256=p["project_sha256"],
        campaign_status=p["status"],
        entries=entries,
        counts=dict(
            completed=sum(e["complete"] for e in entries.values()),
            native_unavailable=sum(not e["native_validated"] for e in entries.values()),
            total=len(entries),
        ),
        effective_workers=min(
            p["config"]["workers"],
            p["config"]["total_rss_bytes"] // p["config"]["worker_rss_bytes"],
        ),
        worker_cpus=p["cpus"],
        semantics=dict(
            seed="Recorded seed grid; EOS and built-in policies deterministic, seeds do not create independent physical cases",
            memory="Observed main-worker RSS and dispatch reservation; supervisor/audit memory excluded, not OS/cgroup quota",
            cpu="Worker CPU affinity and BLAS threads=1; supervisor/audit not included in worker CPU budget",
            quality="Exact piecewise-linear objective-temperature errors on accepted samples; incomplete horizon never treated as complete",
            cost="Per-attempt spawn/run/join wall plus separate audit wall; no speedup goal or advantage claim",
        ),
        validated=False,
        plant_referenced=False,
    )


def report(folder):
    value = store.load(folder)
    if value["payload"]["status"] == "running" or (Path(folder) / "CONTROLLER.json").exists():
        raise ValueError("Wait for quiescent campaign")
    native = {}
    for id in value["payload"]["entries"]:
        try:
            native[id] = store.native(folder, id)
        except (ValueError, OSError, KeyError, TypeError):
            native[id] = None
    return summarize(value["payload"], native)


def export(folder, output):
    folder = Path(folder)
    output = Path(output)
    with Lock(folder):
        value = store.load(folder)
        if value["payload"]["status"] == "running" or (folder / "CONTROLLER.json").exists():
            raise ValueError("Active rollout export forbidden")
        output.mkdir(parents=True, exist_ok=False)
        try:
            native = {}
            for id in value["payload"]["entries"]:
                try:
                    job = store.native(folder, id)
                except (ValueError, OSError, KeyError, TypeError):
                    native[id] = None
                    continue
                native[id] = job
                target = output / "jobs" / id
                (target / "revisions").mkdir(parents=True)
                atomic_json(
                    target / "revisions/snapshot.json", dict(payload=job, sha256=digest(job))
                )
                atomic_json(
                    target / "HEAD.json",
                    dict(
                        schema_version=1, checkpoint="revisions/snapshot.json", sha256=digest(job)
                    ),
                )
            atomic_json(output / "campaign.json", value)
            atomic_json(output / "report.json", summarize(value["payload"], native))
            files = [p for p in output.rglob("*") if p.is_file()]
            if (
                len(files) > 256
                or any(p.stat().st_size > 32 * 1024**2 for p in files)
                or sum(p.stat().st_size for p in files) > 512 * 1024**2
            ):
                raise ValueError("Rollout export size limit")
            atomic_json(
                output / "manifest.json",
                dict(
                    schema_version=1,
                    source=store.source_identity(),
                    artifacts={
                        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files
                    },
                    validated=False,
                    plant_referenced=False,
                ),
            )
            verify(output)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        schema_version=1,
        output=str(output),
        campaign_sha256=value["sha256"],
        validated=False,
        plant_referenced=False,
    )


def verify(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(manifest, ("schema_version", "source", "artifacts", "validated", "plant_referenced"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source"] != store.source_identity()
        or manifest["validated"] is not False
        or manifest["plant_referenced"] is not False
        or not isinstance(manifest["artifacts"], dict)
        or not 2 <= len(manifest["artifacts"]) <= 256
    ):
        raise ValueError("Rollout export schema/source")
    total = 0
    for name, sha in manifest["artifacts"].items():
        with local_file(folder, name).open("rb") as stream:
            raw = stream.read(32 * 1024**2 + 1)
        total += len(raw)
        if (
            len(raw) > 32 * 1024**2
            or total > 512 * 1024**2
            or hashlib.sha256(raw).hexdigest() != sha
        ):
            raise ValueError("Rollout artifact hash/size")
    value = read(local_file(folder, "campaign.json"))
    keys(value, ("payload", "sha256"))
    if digest(value["payload"]) != value["sha256"]:
        raise ValueError("Rollout snapshot hash")
    p = store.validate(value["payload"])
    if p["status"] == "running" or p["controller"] is not None:
        raise ValueError("Active rollout snapshot")
    wanted = {"campaign.json", "report.json"}
    native = {}
    expected = store.expected(p["project"], p["config"])
    for id in p["entries"]:
        head = "jobs/" + id + "/HEAD.json"
        if head not in manifest["artifacts"]:
            native[id] = None
            continue
        job = jobs.load_job(local_file(folder, head).parent)
        audit_native(job)
        if job["spec"] != expected[id]["spec"]:
            raise ValueError("Rollout snapshot job spec")
        native[id] = job
        wanted.update([head, "jobs/" + id + "/revisions/snapshot.json"])
    result = summarize(p, native)
    if read(local_file(folder, "report.json")) != result:
        raise ValueError("Rollout report recomputation differs")
    if set(manifest["artifacts"]) != wanted or {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()
    } != wanted | {"manifest.json"}:
        raise ValueError("Rollout snapshot inventory")
    return result
