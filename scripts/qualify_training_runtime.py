"""Execute the predeclared short-core benchmark, exact parity and tier replay."""

import argparse
import cProfile
import json
import os
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from atomic_model_training.episode import identity, run_episode
from atomic_model_training.runner import envelope
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_training.template import Template
from atomic_model_workspace.common import digest


def case(name, commands=None, tier="training"):
    return dict(
        id=name,
        family=name,
        split="validation",
        seed=17,
        tier=tier,
        steps=[dict(duration_s=0.1, commands=commands or {}) for _ in range(2)],
    )


def qualify(protocol_path, output, admission_path=None):
    protocol = read_json(protocol_path)
    admission = read_json(admission_path) if admission_path else None
    if admission is not None:
        if (
            admission["protocol_sha256"] != digest(protocol)
            or admission["source_sha256"] != digest(identity())
            or admission["system"] != platform.system()
            or admission["machine"] != platform.machine()
            or admission["cpu"] not in os.sched_getaffinity(0)
        ):
            raise ValueError("Frozen machine/source/protocol admission differs")
        os.sched_setaffinity(0, {admission["cpu"]})
    if protocol["workload"] != dict(episodes=6, steps_per_episode=2, duration_s=0.1, commands={}):
        raise ValueError("This executable implements only the declared fixed workload")
    baseline, optimized, equality = [], [], []
    checkpoint_witnesses = []
    profiler = cProfile.Profile()
    profiler.enable()
    run_episode(case("profile"))
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        if any(
            token in filename.lower() + function.lower()
            for token in ("eos", "jacob", "radau", "initialization", "ledger", "rho_u")
        ):
            functions.append(
                dict(
                    file=Path(filename).name,
                    line=line,
                    function=function,
                    primitive_calls=primitive,
                    calls=calls,
                    self_s=own,
                    cumulative_s=cumulative,
                )
            )
    # Alternate order to reduce cache/thermal ordering bias. Include template preparation.
    for repetition in range(3):
        records = {}
        for mode in ("baseline", "optimized") if repetition % 2 == 0 else ("optimized", "baseline"):
            start = time.perf_counter()
            template = Template() if mode == "optimized" else None
            rows = [
                run_episode(case(f"episode-{i}"), template=template)
                for i in range(protocol["workload"]["episodes"])
            ]
            elapsed = time.perf_counter() - start
            (baseline if mode == "baseline" else optimized).append(elapsed)
            records[mode] = [r["checkpoint"] for r in rows]
        equality.append(records["baseline"] == records["optimized"])
        checkpoint_witnesses.append(
            dict(
                baseline=[p["sha256"] for p in records["baseline"]],
                optimized=[p["sha256"] for p in records["optimized"]],
                checkpoint=records["baseline"][0],
            )
        )
    tiers = []
    for name, commands in [
        ("nominal", {}),
        ("load", {"throttle": 0.9}),
        ("scram", {"scram": True}),
        ("coast", {"motor_off": True}),
    ]:
        low = run_episode(case(name, commands))
        high = run_episode(case(name, commands, "reference"))
        a, b = np.array(low["state"]), np.array(high["state"])
        error = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))
        balances = max(
            low["balances"][k] for k in ("energy_relative", "mass_relative", "boron_relative")
        )
        event_keys = ("events", "protection_event_s")
        event_match = all(
            all(x[k] == y[k] for k in event_keys)
            for x, y in zip(
                low["checkpoint"]["payload"]["history"], high["checkpoint"]["payload"]["history"]
            )
        )
        tiers.append(
            dict(
                case=name,
                training_checkpoint_sha256=low["checkpoint"]["sha256"],
                reference_checkpoint_sha256=high["checkpoint"]["sha256"],
                scaled_state_error=error,
                balance_relative=balances,
                events_equal=event_match,
                accepted=error <= protocol["tier_state_scaled_error_maximum"]
                and balances <= protocol["balance_relative_maximum"]
                and event_match,
            )
        )
    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import atomic_model_training.episode"], check=True)
    import_start = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    speedup = statistics.median(baseline) / statistics.median(optimized)
    report = dict(
        protocol_sha256=digest(protocol),
        admission_sha256=None if admission is None else digest(admission),
        executable_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        source=identity(),
        machine=dict(
            system=platform.system(),
            machine=platform.machine(),
            processor=platform.processor(),
            python=platform.python_version(),
            cpu_affinity=sorted(__import__("os").sched_getaffinity(0)),
        ),
        baseline_s=baseline,
        optimized_s=optimized,
        median_speedup=speedup,
        performance_accepted=speedup >= protocol["performance_minimum_speedup"],
        exact_checkpoint_parity=equality,
        checkpoint_witnesses=checkpoint_witnesses,
        tiers=tiers,
        profile=dict(
            cold_import_process_s=import_start,
            user_cpu_s=usage.ru_utime,
            system_cpu_s=usage.ru_stime,
            peak_rss_kib=usage.ru_maxrss,
            functions=sorted(functions, key=lambda row: -row["cumulative_s"]),
            note="cumulative times overlap; self times are disjoint; RSS is process high-water mark",
        ),
        validated=False,
        plant_referenced=False,
    )
    report["accepted"] = (
        report["performance_accepted"] and all(equality) and all(t["accepted"] for t in tiers)
    )
    atomic_json(output, envelope(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="validation/ai091012_training_protocol_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--admission")
    args = parser.parse_args()
    report = qualify(args.protocol, args.output, args.admission)
    print(
        json.dumps(
            {k: report[k] for k in ("accepted", "median_speedup", "exact_checkpoint_parity")}
        )
    )
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
