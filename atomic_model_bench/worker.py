"""Trusted subprocess entry: apply affinity before importing numerical libraries."""

import cProfile
import os
import pstats
import resource
import sys
import time
from pathlib import Path

from atomic_model_bench.spec import THREAD_NAMES, source_identity
from atomic_model_workspace.common import atomic_json, digest, keys, read


def profile_group(filename, function):
    if filename == "scripts/compressible_storage_graph.py" and function in (
        "rho_T",
        "rho_u",
        "p_T",
    ):
        return "synthetic_eos"
    if (
        filename in ("scripts/conservative_eos_loop.py", "scripts/terminal_eos_chambers.py")
        and function == "point"
    ):
        return "caloric_state_and_entropy"
    if filename == "scripts/controlled_eos_assembly.py" and function == "_transaction":
        return "explicit_ssprk_transaction_self"
    if filename.startswith("scipy/integrate/") and (
        "jac" in function or function in ("num_jac", "_dense_num_jac", "_sparse_num_jac")
    ):
        return "scipy_jacobian"
    if filename.startswith("scipy/integrate/"):
        return "scipy_integrator_other"
    if filename.startswith("scripts/"):
        return "native_equations_and_admission"
    return "other"


def run(request):
    keys(request, ("schema_version", "case", "cpus", "profile", "request_sha256"))
    payload = {k: v for k, v in request.items() if k != "request_sha256"}
    if (
        type(request["schema_version"]) is not int
        or request["schema_version"] != 1
        or type(request["profile"]) is not bool
        or request["request_sha256"] != digest(payload)
    ):
        raise ValueError("Worker input identity mismatch")
    cpus = request["cpus"]
    if (
        not isinstance(cpus, list)
        or not cpus
        or any(type(c) is not int for c in cpus)
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("Worker CPU allocation unavailable")
    os.sched_setaffinity(0, cpus)
    if any(os.environ.get(n) != "1" for n in THREAD_NAMES):
        raise ValueError("Numerical thread configuration must be fixed before import")
    imported = time.perf_counter()
    import numpy
    import scipy

    from atomic_model_runtime.episode import EOSEpisode

    startup = time.perf_counter() - imported
    case = request["case"]
    profiler = cProfile.Profile()
    if request["profile"]:
        profiler.enable()
    phases = {}
    t = time.perf_counter()
    env = EOSEpisode(case["config"])
    phases["initialize_s"] = time.perf_counter() - t
    action = env.validate_action(case["heat_power_w"])
    vector = [action[k] for k in env.config.action_channels]
    t = time.perf_counter()
    while not env._done:
        env.step(vector)
    phases["integrate_and_admit_s"] = time.perf_counter() - t
    t = time.perf_counter()
    episode = env.export_episode()
    phases["checkpoint_export_s"] = time.perf_counter() - t
    t = time.perf_counter()
    s = env._model.snapshot()
    phases["snapshot_report_s"] = time.perf_counter() - t
    if request["profile"]:
        profiler.disable()
    profile = []
    if request["profile"]:
        for (filename, line, function), (
            primitive,
            total,
            self_s,
            cumulative,
            callers,
        ) in pstats.Stats(profiler).stats.items():
            normalized = filename.replace("\\", "/")
            for marker in (
                "atomic_model_runtime/",
                "scripts/",
                "scipy/",
                "numpy/",
                "core/",
                "physics/",
            ):
                if marker in normalized:
                    normalized = marker + normalized.split(marker, 1)[1]
                    break
            else:
                normalized = Path(filename).name
            profile.append(
                dict(
                    file=normalized,
                    line=line,
                    function=function,
                    primitive_calls=primitive,
                    total_calls=total,
                    self_s=self_s,
                    cumulative_s=cumulative,
                )
            )
        profile.sort(key=lambda r: (-r["self_s"], r["file"], r["line"], r["function"]))
    groups = {}
    for row in profile:
        group = profile_group(row["file"], row["function"])
        groups[group] = groups.get(group, 0.0) + row["self_s"]
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(
        profile_exclusive_self_s_by_group=groups,
        numerical_method="native explicit SSPRK3; Jacobian not applicable",
        cpu_user_s_before_result_write=usage.ru_utime,
        cpu_system_s_before_result_write=usage.ru_stime,
        schema_version=1,
        case_id=case["id"],
        case_sha256=digest(case),
        request_sha256=request["request_sha256"],
        source=source_identity(),
        affinity_cpus=sorted(os.sched_getaffinity(0)),
        threads={n: os.environ[n] for n in THREAD_NAMES},
        versions=dict(numpy=numpy.__version__, scipy=scipy.__version__),
        import_wall_s=startup,
        phases=phases,
        simulated_s=s["time_s"],
        decisions=env._steps,
        episode_sha256=digest(episode),
        episode=episode,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        balances={k: s[k] for k in ("energy_residual_j", "entropy_residual_j_k")},
        profile=profile,
        profiled=request["profile"],
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("worker requires request and output paths")
    result = run(read(sys.argv[1]))
    atomic_json(sys.argv[2], result)


if __name__ == "__main__":
    main()
