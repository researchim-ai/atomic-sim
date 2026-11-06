"""Fixed-work pool trials with bounded worker lifetimes and per-task admission."""

import hashlib
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from atomic_model_bench.harness import _accept, percentile, rss_bytes, stop, trial
from atomic_model_bench.spec import THREAD_NAMES, environment, prepare
from atomic_model_pool.identity import identity
from atomic_model_workspace.common import atomic_json, digest, keys, number, read


def prepare_pool(value):
    keys(value, ("schema_version", "performance", "max_cases_per_worker"))
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Expected pool comparison schema1")
    cap = value["max_cases_per_worker"]
    if type(cap) is not int or not 1 <= cap <= 64:
        raise ValueError("Worker task cap outside 1..64")
    return dict(
        schema_version=1, performance=prepare(value["performance"]), max_cases_per_worker=cap
    )


@contextmanager
def cpu_budget(count):
    previous = {}
    cpus = sorted(os.sched_getaffinity(0))[:count]
    try:
        for path in Path("/proc/self/task").iterdir():
            tid = int(path.name)
            try:
                previous[tid] = os.sched_getaffinity(tid)
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                previous.pop(tid, None)
        yield cpus
    finally:
        for tid, mask in previous.items():
            try:
                os.sched_setaffinity(tid, mask)
            except ProcessLookupError:
                pass


def _start(folder, index, env):
    log = (folder / f"worker-{index:04d}.log").open("w")
    launched = time.perf_counter()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "atomic_model_pool.worker"],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except BaseException:
        log.close()
        raise
    return dict(process=process, log=log, tasks=0, busy=None, index=index, launched=launched)


def _close(item):
    process = item["process"]
    try:
        if process.stdin:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            stop(process)
    finally:
        item["log"].close()


def pooled_trial(spec, output, workers, cpus, source, cap):
    folder = Path(output)
    folder.mkdir()
    perf = spec["performance"]
    queue = list(enumerate(perf["cases"]))
    slots = []
    rows = []
    spawned = 0
    peak = 0
    env = dict(os.environ, **{n: "1" for n in THREAD_NAMES})
    started = time.perf_counter()
    try:
        while queue or any(s["busy"] for s in slots):
            for slot in list(slots):
                if slot["busy"] is None and (
                    slot["tasks"] >= cap or slot["process"].poll() is not None
                ):
                    _close(slot)
                    slots.remove(slot)
            while queue and len(slots) < workers:
                slot = _start(folder, spawned, env)
                spawned += 1
                slots.append(slot)
            for slot in slots:
                if not queue or slot["busy"] is not None:
                    continue
                index, case = queue.pop(0)
                req = dict(schema_version=1, case=case, cpus=cpus, profile=False)
                req["request_sha256"] = digest(req)
                path = folder / f"case-{index:04d}.input.json"
                result = folder / f"case-{index:04d}.result.json"
                atomic_json(path, req)
                command = dict(
                    request=str(path.resolve()), result=str(result.resolve()), token=digest(req)
                )
                payload = (__import__("json").dumps(command) + "\n").encode()
                if len(payload) > 8192:
                    raise ValueError("Pool mailbox path size limit")
                slot["busy"] = dict(
                    case=case,
                    request=req,
                    result_path=result,
                    token=command["token"],
                    started=slot.get("launched", time.perf_counter())
                    if slot["tasks"] == 0
                    else time.perf_counter(),
                )
                try:
                    slot["process"].stdin.write(payload)
                    slot["process"].stdin.flush()
                except (BrokenPipeError, OSError):
                    stop(slot["process"])
            total_rss = 0
            for slot in list(slots):
                process = slot["process"]
                rss = rss_bytes(process.pid)
                total_rss += rss
                item = slot["busy"]
                if item is None:
                    continue
                elapsed = time.perf_counter() - item["started"]
                outcome = None
                if elapsed >= perf["worker_timeout_s"]:
                    outcome = "deadline"
                if rss > perf["worker_rss_limit_mib"] * 1024**2:
                    outcome = "rss_limit"
                result = None
                error = None
                if outcome:
                    stop(process)
                elif item["result_path"].exists():
                    try:
                        envelope = read(item["result_path"])
                        keys(
                            envelope,
                            ("schema_version", "token", "pool", "startup_wall_s", "result"),
                        )
                        if (
                            type(envelope["schema_version"]) is not int
                            or envelope["schema_version"] != 1
                            or envelope["token"] != item["token"]
                            or envelope["pool"] != source
                        ):
                            raise ValueError("Pool completion identity mismatch")
                        number(envelope["startup_wall_s"], 0, 1e6, "startup wall time")
                        result = envelope["result"]
                        _accept(result, item["case"], item["request"], source["baseline"], cpus)
                        number(result["peak_rss_bytes"], 0, 1e15, "worker RSS")
                        outcome = (
                            "completed"
                            if result["peak_rss_bytes"] <= perf["worker_rss_limit_mib"] * 1024**2
                            else "rss_limit"
                        )
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        outcome = "invalid_result"
                        error = str(exc)
                elif process.poll() is not None:
                    outcome = "worker_exit"
                else:
                    continue
                row = dict(
                    case_id=item["case"]["id"],
                    status=outcome,
                    worker_index=slot["index"],
                    pid=process.pid,
                    task_ordinal=slot["tasks"] + 1,
                    latency_wall_s=time.perf_counter() - item["started"],
                    error=error,
                )
                if result is not None:
                    row["result"] = dict(
                        result_path=item["result_path"].name,
                        sha256=hashlib.sha256(item["result_path"].read_bytes()).hexdigest(),
                        episode_sha256=result["episode_sha256"],
                        peak_rss_bytes=result["peak_rss_bytes"],
                        simulated_s=result["simulated_s"],
                        import_wall_s=result["import_wall_s"],
                        worker_startup_wall_s=envelope["startup_wall_s"],
                        phases=result["phases"],
                    )
                rows.append(row)
                slot["tasks"] += 1
                slot["busy"] = None
                atomic_json(folder / "progress.json", dict(rows=rows))
                if outcome != "completed":
                    stop(process)
                    _close(slot)
                    slots.remove(slot)
            peak = max(peak, total_rss)
            if queue or any(s["busy"] for s in slots):
                time.sleep(0.005)
    finally:
        for slot in slots:
            if slot["busy"] is not None:
                stop(slot["process"])
            _close(slot)
    wall = time.perf_counter() - started
    completed = [r for r in rows if r["status"] == "completed"]
    simulated = sum(r["result"]["simulated_s"] for r in completed)
    result = dict(
        schema_version=1,
        mode="pool",
        workers=workers,
        worker_processes_spawned=spawned,
        max_cases_per_worker=cap,
        shared_cpu_set=cpus,
        cases=len(perf["cases"]),
        completed=len(completed),
        all_completed=len(completed) == len(perf["cases"]),
        wall_s=wall,
        simulated_s=simulated,
        episodes_per_second=len(completed) / wall,
        simulated_seconds_per_wall_second=simulated / wall,
        latency_p50_s=percentile([r["latency_wall_s"] for r in rows], 0.5),
        latency_p95_s=percentile([r["latency_wall_s"] for r in rows], 0.95),
        sampled_peak_aggregate_rss_bytes=peak,
        rows=sorted(rows, key=lambda r: r["case_id"]),
    )
    atomic_json(folder / "trial.json", result)
    return result


def compare(value, output):
    spec = prepare_pool(value)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "input.json", spec)
    source = identity()
    trials = []
    reference = {}
    exact = True
    with cpu_budget(spec["performance"]["cpu_count"]) as cpus:
        host = environment()
        for repeat in range(spec["performance"]["repeats"]):
            for workers in spec["performance"]["workers"]:
                modes = ("classic", "pool") if repeat % 2 == 0 else ("pool", "classic")
                for mode in modes:
                    path = folder / f"repeat-{repeat:02d}-workers-{workers}-{mode}"
                    if mode == "classic":
                        result = trial(spec["performance"], path, workers, cpus, source["baseline"])
                    else:
                        result = pooled_trial(
                            spec, path, workers, cpus, source, spec["max_cases_per_worker"]
                        )
                    result.update(mode=mode, repeat=repeat, artifact=path.name + "/trial.json")
                    for row in result["rows"]:
                        if row["status"] != "completed":
                            continue
                        fp = row["result"]["episode_sha256"]
                        if row["case_id"] in reference and reference[row["case_id"]] != fp:
                            exact = False
                        reference.setdefault(row["case_id"], fp)
                    trials.append(result)
                    atomic_json(
                        folder / "progress.json",
                        dict(trials=trials, numerical_identity_exact=exact),
                    )
        summaries = []
        for workers in spec["performance"]["workers"]:
            classic = [t for t in trials if t["workers"] == workers and t["mode"] == "classic"]
            pooled = [t for t in trials if t["workers"] == workers and t["mode"] == "pool"]
            eligible = exact and all(t["all_completed"] for t in classic + pooled)
            a = percentile([t["episodes_per_second"] for t in classic], 0.5)
            b = percentile([t["episodes_per_second"] for t in pooled], 0.5)
            summaries.append(
                dict(
                    workers=workers,
                    eligible=eligible,
                    classic_median_episodes_per_second=a,
                    pool_median_episodes_per_second=b,
                    pool_speedup_at_same_worker_count=b / a if eligible else None,
                )
            )
        passed = exact and all(t["all_completed"] for t in trials)
        report = dict(
            schema_version=1,
            source=source,
            request_sha256=digest(spec),
            environment=host,
            fixed_cpu_set=cpus,
            thread_settings={n: "1" for n in THREAD_NAMES},
            trials=trials,
            summaries=summaries,
            numerical_identity_exact=exact,
            passed=passed,
            two_x_vs_classic_same_workers=passed
            and any(s["pool_speedup_at_same_worker_count"] >= 2 for s in summaries),
            validated=False,
            plant_referenced=False,
            notes=[
                "Every task constructs fresh EOSEpisode; only process and Python imports reused",
                "New processes for every trial; cold first tasks and final shutdown included in trial wall time",
                "Classic/pool order alternates across repeats; identical cases and shared supervisor/worker CPU set",
                "Worker restarts on rejection, deadline, RSS limit or invalid completion; task cap limits reuse lifetime",
                "RSS is observed per-worker process high-water, not current-case incremental allocation or hard OS quota",
                "Timing quality is workload/hardware specific; all failures remain visible; no physical or universal speed claim",
            ],
        )
        atomic_json(folder / "report.json", report)
        atomic_json(
            folder / "manifest.json",
            dict(
                schema_version=1,
                source=source,
                artifacts={
                    str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(folder.rglob("*"))
                    if p.is_file()
                },
                validated=False,
                plant_referenced=False,
            ),
        )
    return report
