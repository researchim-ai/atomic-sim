"""Controlled fresh-read fingerprint optimization: same-tree legacy vs scandir.

Worker override exists only in this explicit qualification process. Production
SDK has one implementation and never caches source bytes or filesystem metadata.
"""

import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def legacy_fingerprint():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for package in (
        "sdk",
        "core",
        "plants",
        "physics",
        "components",
        "control",
        "instrumentation",
        "agent_envs",
        "data",
        "scenarios",
    ):
        for path in sorted((root / package).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".yaml"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(strategy, source, target):
    import sdk.experiment as sdk
    from atomic_model_workspace.common import atomic_json, read

    if any(
        name in sys.modules
        for name in ("scripts.inertial_graph", "scripts.compressible_storage_graph")
    ):
        raise ValueError("Qualification must choose implementation before native driver imports")
    if strategy == "legacy":
        sdk._source_fingerprint = legacy_fingerprint
    from atomic_model_bench.worker import run

    atomic_json(target, run(read(source)))


def run(spec, output):
    from atomic_model_bench import harness
    from atomic_model_bench.spec import environment, prepare, source_identity
    from atomic_model_pool.runner import cpu_budget
    from atomic_model_workspace.common import atomic_json, digest
    from sdk.experiment import _source_fingerprint

    spec = prepare(spec)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "input.json", spec)
    source = source_identity()
    trials = []
    exact = True
    reference = {}
    with cpu_budget(spec["cpu_count"]) as cpus:
        host = environment()
        micro = []
        for repeat in range(10):
            order = ("legacy", "optimized") if repeat % 2 == 0 else ("optimized", "legacy")
            row = {}
            for strategy in order:
                fn = legacy_fingerprint if strategy == "legacy" else _source_fingerprint
                start = time.perf_counter()
                hashes = [fn() for _ in range(5)]
                row[strategy + "_wall_s"] = (time.perf_counter() - start) / 5
                if any(h != _source_fingerprint() for h in hashes):
                    raise ValueError("Fingerprint byte-stream mismatch")
            micro.append(row)
        legacy_median = harness.percentile([x["legacy_wall_s"] for x in micro], 0.5)
        optimized_median = harness.percentile([x["optimized_wall_s"] for x in micro], 0.5)
        real_subprocess = harness.subprocess
        for repeat in range(spec["repeats"]):
            for workers in spec["workers"]:
                order = ("legacy", "optimized") if repeat % 2 == 0 else ("optimized", "legacy")
                for strategy in order:

                    def launch(argv, **kwargs):
                        if argv[1:3] != ["-m", "atomic_model_bench.worker"]:
                            raise ValueError("Unexpected qualification worker")
                        return real_subprocess.Popen(
                            [
                                argv[0],
                                "-m",
                                "scripts.qualify_source_scan",
                                "worker",
                                strategy,
                                argv[3],
                                argv[4],
                            ],
                            **kwargs,
                        )

                    harness.subprocess = SimpleNamespace(
                        Popen=launch,
                        TimeoutExpired=subprocess.TimeoutExpired,
                        STDOUT=subprocess.STDOUT,
                    )
                    path = folder / f"repeat-{repeat:02d}-workers-{workers}-{strategy}"
                    try:
                        result = harness.trial(spec, path, workers, cpus, source)
                    finally:
                        harness.subprocess = real_subprocess
                    result.update(
                        strategy=strategy, repeat=repeat, artifact=path.name + "/trial.json"
                    )
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
        for workers in spec["workers"]:
            old = [t for t in trials if t["workers"] == workers and t["strategy"] == "legacy"]
            new = [t for t in trials if t["workers"] == workers and t["strategy"] == "optimized"]
            eligible = exact and all(t["all_completed"] for t in old + new)
            a = harness.percentile([t["episodes_per_second"] for t in old], 0.5)
            b = harness.percentile([t["episodes_per_second"] for t in new], 0.5)
            summaries.append(
                dict(
                    workers=workers,
                    eligible=eligible,
                    legacy_median_episodes_per_second=a,
                    optimized_median_episodes_per_second=b,
                    speedup_same_workers=b / a if eligible else None,
                )
            )
        passed = exact and all(t["all_completed"] for t in trials)
        report = dict(
            schema_version=1,
            source=source,
            qualification_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_sha256=digest(spec),
            environment=host,
            fixed_cpu_set=cpus,
            trials=trials,
            summaries=summaries,
            microbenchmark=dict(
                samples=micro,
                legacy_median_s=legacy_median,
                optimized_median_s=optimized_median,
                speedup=legacy_median / optimized_median,
            ),
            numerical_identity_exact=exact,
            passed=passed,
            two_x_same_workers=passed and any(s["speedup_same_workers"] >= 2 for s in summaries),
            validated=False,
            plant_referenced=False,
            scope="Same current source tree and fresh-byte digest on both strategies; only qualification workers select old traversal. New processes and identical wrapper/import order for both modes; no persistent workers or numerical shortcuts.",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("worker")
    p.add_argument("strategy", choices=["legacy", "optimized"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("run")
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "worker":
        worker(args.strategy, args.input, args.output)
        return 0
    from atomic_model_workspace.common import canonical, read

    try:
        result = run(read(args.input), args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        canonical(
            {
                k: result[k]
                for k in (
                    "passed",
                    "numerical_identity_exact",
                    "two_x_same_workers",
                    "summaries",
                    "microbenchmark",
                )
            }
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
