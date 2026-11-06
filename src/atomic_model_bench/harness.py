"""Fixed-work Linux subprocess trials; timing never changes numerical fidelity."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from atomic_model_bench.spec import THREAD_NAMES, environment, prepare, source_identity
from atomic_model_workspace.common import atomic_json, digest, read


def rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _accept(result, case, request, source, cpus):
    from atomic_model_runtime.episode import EOSEpisode

    if (
        result["schema_version"] != 1
        or result["case_sha256"] != digest(case)
        or result["request_sha256"] != request["request_sha256"]
        or result["source"] != source
        or result["affinity_cpus"] != cpus
        or result["threads"] != {n: "1" for n in THREAD_NAMES}
        or result["profiled"] is not request["profile"]
    ):
        raise ValueError("Worker result identity mismatch")
    restored = EOSEpisode.from_episode(result["episode"])
    if (
        not restored._done
        or restored.config.as_dict() != case["config"]
        or digest(result["episode"]) != result["episode_sha256"]
        or result["simulated_s"] != case["config"]["episode_horizon_s"]
        or result["decisions"] != restored._steps
    ):
        raise ValueError("Worker did not complete the exact declared episode")
    if any(row["commands"] != case["heat_power_w"] for row in result["episode"]["action_log"]):
        raise ValueError("Worker action history mismatch")


def trial(spec, output, workers, cpus, source, *, profiled=False):
    output = Path(output)
    output.mkdir()
    cases = spec["cases"][:1] if profiled else spec["cases"]
    queued = list(enumerate(cases))
    active = {}
    rows = []
    peak = 0
    env = dict(os.environ, **{name: "1" for name in THREAD_NAMES})
    # Retain normal module discovery for checkout execution. Cold replay proves
    # installed discovery separately with PYTHONPATH removed and cwd outside repo.
    started = time.perf_counter()
    try:
        while queued or active:
            while queued and len(active) < workers:
                index, case = queued.pop(0)
                request = dict(schema_version=1, case=case, cpus=cpus, profile=profiled)
                request["request_sha256"] = digest(request)
                prefix = output / f"case-{index:04d}"
                request_path = prefix.with_suffix(".input.json")
                result_path = prefix.with_suffix(".result.json")
                atomic_json(request_path, request)
                log = prefix.with_suffix(".log").open("w")
                t = time.perf_counter()
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "atomic_model_bench.worker",
                            str(request_path.resolve()),
                            str(result_path.resolve()),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                active[process.pid] = dict(
                    process=process,
                    case=case,
                    request=request,
                    result_path=result_path,
                    log=log,
                    started=t,
                )
            aggregate = 0
            for pid, item in list(active.items()):
                process = item["process"]
                elapsed = time.perf_counter() - item["started"]
                rss = rss_bytes(pid)
                aggregate += rss
                outcome = None
                if elapsed >= spec["worker_timeout_s"]:
                    outcome = "deadline"
                if rss > spec["worker_rss_limit_mib"] * 1024**2:
                    outcome = "rss_limit"
                if outcome:
                    stop(process)
                elif process.poll() is None:
                    continue
                else:
                    outcome = "completed" if process.returncode == 0 else "worker_exit"
                row = dict(
                    case_id=item["case"]["id"],
                    status=outcome,
                    exitcode=process.returncode,
                    latency_wall_s=time.perf_counter() - item["started"],
                    observed_rss_at_exit_bytes=rss,
                )
                if outcome == "completed":
                    try:
                        result = read(item["result_path"])
                        _accept(result, item["case"], item["request"], source, cpus)
                        if result["peak_rss_bytes"] > spec["worker_rss_limit_mib"] * 1024**2:
                            row["status"] = "rss_limit"
                        row["result"] = dict(
                            result_path=item["result_path"].name,
                            sha256=hashlib.sha256(item["result_path"].read_bytes()).hexdigest(),
                            episode_sha256=result["episode_sha256"],
                            peak_rss_bytes=result["peak_rss_bytes"],
                            simulated_s=result["simulated_s"],
                            phases=result["phases"],
                            import_wall_s=result["import_wall_s"],
                        )
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        row.update(status="invalid_result", error=str(exc))
                item["log"].close()
                del active[pid]
                rows.append(row)
                atomic_json(output / "progress.json", dict(rows=rows))
            peak = max(peak, aggregate)
            if active:
                time.sleep(0.01)
    finally:
        for item in active.values():
            stop(item["process"])
            item["log"].close()
    wall = time.perf_counter() - started
    complete = [r for r in rows if r["status"] == "completed"]
    simulated = sum(r["result"]["simulated_s"] for r in complete)
    result = dict(
        schema_version=1,
        workers=workers,
        shared_cpu_set=cpus,
        profiled=profiled,
        wall_s=wall,
        cases=len(cases),
        completed=len(complete),
        all_completed=len(complete) == len(cases),
        simulated_s=simulated,
        simulated_seconds_per_wall_second=simulated / wall,
        episodes_per_second=len(complete) / wall,
        latency_p50_s=percentile([r["latency_wall_s"] for r in rows], 0.5),
        latency_p95_s=percentile([r["latency_wall_s"] for r in rows], 0.95),
        sampled_peak_aggregate_rss_bytes=peak,
        rows=sorted(rows, key=lambda r: r["case_id"]),
    )
    atomic_json(output / "trial.json", result)
    return result


def _benchmark(spec, output, *, profile=True):
    spec = prepare(spec)
    if type(profile) is not bool:
        raise ValueError("profile must be boolean")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input.json", spec)
    source = source_identity()
    host = environment()
    cpus = host["available_cpus"][: spec["cpu_count"]]
    trials = []
    reference = {}
    exact = True
    for repeat in range(spec["repeats"]):
        for workers in spec["workers"]:
            folder = output / f"repeat-{repeat:02d}-workers-{workers}"
            result = trial(spec, folder, workers, cpus, source)
            result["artifact"] = folder.name + "/trial.json"
            result["repeat"] = repeat
            for row in result["rows"]:
                if row["status"] != "completed":
                    continue
                numerical = row["result"]["episode_sha256"]
                if row["case_id"] in reference and reference[row["case_id"]] != numerical:
                    exact = False
                reference.setdefault(row["case_id"], numerical)
            trials.append(result)
            atomic_json(
                output / "progress.json", dict(trials=trials, numerical_identity_exact=exact)
            )
    summaries = []
    baseline = percentile(
        [t["episodes_per_second"] for t in trials if t["workers"] == 1 and t["all_completed"]], 0.5
    )
    for workers in spec["workers"]:
        values = [t for t in trials if t["workers"] == workers]
        eligible = (
            all(t["all_completed"] for t in values)
            and exact
            and all(t["all_completed"] for t in trials if t["workers"] == 1)
        )
        throughput = percentile([t["episodes_per_second"] for t in values], 0.5)
        speedup = throughput / baseline if eligible and baseline else None
        summaries.append(
            dict(
                workers=workers,
                repeats=len(values),
                eligible=eligible,
                median_episodes_per_second=throughput,
                median_simulated_seconds_per_wall_second=percentile(
                    [t["simulated_seconds_per_wall_second"] for t in values], 0.5
                ),
                speedup_vs_one=speedup,
                efficiency_speedup_per_worker=speedup / workers if speedup is not None else None,
            )
        )
    profile_result = None
    if profile:
        profile_result = trial(spec, output / "profile", 1, cpus, source, profiled=True)
        for row in profile_result["rows"]:
            if (
                row["status"] == "completed"
                and reference.get(row["case_id"]) != row["result"]["episode_sha256"]
            ):
                exact = False
    report = dict(
        schema_version=1,
        source=source,
        environment=host,
        request_sha256=digest(spec),
        fixed_cpu_set=cpus,
        thread_settings={n: "1" for n in THREAD_NAMES},
        trials=trials,
        summaries=summaries,
        profile=profile_result,
        numerical_identity_exact=exact,
        passed=exact
        and all(t["all_completed"] for t in trials)
        and (not profile or profile_result["all_completed"]),
        two_x_goal_met=exact
        and all(t["all_completed"] for t in trials)
        and (not profile or profile_result["all_completed"])
        and any(s["eligible"] and s["speedup_vs_one"] >= 2 for s in summaries),
        validated=False,
        plant_referenced=False,
        notes=[
            "Linux trusted workers only; sampled per-process RSS limit is supervised, not an OS hard memory quota",
            "All modes share the same CPU set and workload; each worker uses one BLAS/OpenMP thread; oversubscription is measured",
            "Trials include interpreter/import/startup, compute, checkpoint output and supervisor admission; profile is separate and never used for speedup",
            "p50/p95 interpolate case end-to-end latencies; small samples do not establish population tail latency",
            "Seeds are identities; fixed EOS actions use no RNG; native balances/admission tolerances unchanged",
            "Self profile times are additive; cumulative profile times overlap and must not be summed",
            "2x is a measured fixed-work end-to-end goal, not a universal simulator performance claim",
        ],
    )
    atomic_json(output / "report.json", report)
    files = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    atomic_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            artifacts=files,
            units=dict(
                wall_s="s",
                latency_p50_s="s",
                latency_p95_s="s",
                peak_rss_bytes="byte",
                sampled_peak_aggregate_rss_bytes="byte",
                simulated_s="s",
                episodes_per_second="1/s",
            ),
            source=source,
            validated=False,
            plant_referenced=False,
        ),
    )
    return report


def benchmark(spec, output, *, profile=True):
    spec = prepare(spec)
    if type(profile) is not bool:
        raise ValueError("profile must be boolean")
    cpus = sorted(os.sched_getaffinity(0))[: spec["cpu_count"]]
    previous = {}
    try:
        # Existing numerical-library pools may predate this call. Restrict all
        # supervisor threads, not only its main thread. New threads inherit it.
        for path in Path("/proc/self/task").iterdir():
            tid = int(path.name)
            try:
                previous[tid] = os.sched_getaffinity(tid)
                os.sched_setaffinity(tid, cpus)
            except ProcessLookupError:
                previous.pop(tid, None)
        return _benchmark(spec, output, profile=profile)
    finally:
        for tid, affinity in previous.items():
            try:
                os.sched_setaffinity(tid, affinity)
            except ProcessLookupError:
                pass
