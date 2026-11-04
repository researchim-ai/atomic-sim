"""Strict workloads, CPU allocation and measurement provenance."""

import hashlib
import os
import platform
from pathlib import Path

from atomic_model_workspace.common import digest, identifier, keys, number

THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def prepare(value):
    from atomic_model_runtime.episode import EOSEpisode

    keys(
        value,
        (
            "schema_version",
            "cases",
            "workers",
            "repeats",
            "cpu_count",
            "worker_timeout_s",
            "worker_rss_limit_mib",
        ),
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Expected EOS performance schema1")
    if (
        platform.system() != "Linux"
        or not hasattr(os, "sched_getaffinity")
        or not Path("/proc/self/status").is_file()
    ):
        raise ValueError("This measured RSS/affinity backend requires Linux /proc")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("Require 1..64 fixed cases")
    normalized = []
    seen = set()
    for case in cases:
        keys(case, ("id", "config", "heat_power_w", "seed"))
        name = identifier(case["id"])
        if name in seen or type(case["seed"]) is not int or not 0 <= case["seed"] < 2**32:
            raise ValueError("Duplicate case or invalid seed")
        seen.add(name)
        env = EOSEpisode(case["config"])
        normalized.append(
            dict(
                id=name,
                config=env.config.as_dict(),
                heat_power_w=env.validate_action(case["heat_power_w"]),
                seed=case["seed"],
            )
        )
    workers = value["workers"]
    if (
        not isinstance(workers, list)
        or not 1 <= len(workers) <= 4
        or any(type(w) is not int or w not in (1, 2, 4, 8) for w in workers)
        or len(set(workers)) != len(workers)
        or 1 not in workers
    ):
        raise ValueError("Unique worker counts from 1/2/4/8, including baseline1 required")
    for name, low, high in (("repeats", 1, 5), ("cpu_count", 1, len(os.sched_getaffinity(0)))):
        if type(value[name]) is not int or not low <= value[name] <= high:
            raise ValueError("Invalid " + name)
    if any(w > len(cases) for w in workers):
        raise ValueError("Each worker-count trial needs at least that many cases")
    return dict(
        schema_version=1,
        cases=normalized,
        workers=sorted(workers),
        repeats=value["repeats"],
        cpu_count=value["cpu_count"],
        worker_timeout_s=number(value["worker_timeout_s"], 0.01, 3600, "worker timeout"),
        worker_rss_limit_mib=number(value["worker_rss_limit_mib"], 64, 16384, "RSS limit MiB"),
    )


def environment():
    model = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    return dict(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu_model=model,
        available_cpus=sorted(os.sched_getaffinity(0)),
        logical_cpu_count=os.cpu_count(),
    )


def source_identity():
    from atomic_model_runtime.episode import runtime_provenance

    return dict(
        harness_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        runtime=runtime_provenance(),
    )
