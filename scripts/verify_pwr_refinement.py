"""Full-horizon adaptive-step/tolerance sensitivity of the reduced controlled PWR.

Finite numerical references only; no experimental validation or Richardson order.
Each native campaign retains its checkpoint, balances, failures and observations.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from atomic_model_controlled_plant.qualification import CASES, protocol
from atomic_model_controlled_plant.session import identity
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest
from scripts.verify_pwr_horizons import admit_result
from scripts.run_pwr_windowed_reference import CASE as WINDOWED_CASE, RESET_INTERVAL_S, SETTINGS as WINDOWED_SETTINGS

DEFAULT_CASES = ("scram", "coastdown", "natural", "load_rejection_equilibrium")
PROFILES = {
    "cap-2": dict(method="Radau", max_step_s=2.0, rtol=1e-6, atol=1e-5, output_step_s=5.0),
    "cap-1": dict(method="Radau", max_step_s=1.0, rtol=1e-6, atol=1e-5, output_step_s=5.0),
    "cap-0.5": dict(method="Radau", max_step_s=0.5, rtol=1e-6, atol=1e-5, output_step_s=5.0),
    "reference": dict(method="Radau", max_step_s=0.5, rtol=1e-8, atol=1e-7, output_step_s=5.0),
}
# Predeclared numerical comparison budgets in SI, not experimental uncertainties.
BUDGETS = dict(
    fission_power_w=3.1e5, thermal_power_w=3.1e5,
    fuel_temperature_k=0.1, core_temperature_k=0.1, cold_leg_temperature_k=0.1,
    primary_pressure_pa=1600.0, sg_pressure_pa=640.0,
    mass_flow_kg_s=1.8, feed_flow_kg_s=0.1, steam_flow_kg_s=0.1,
    dump_flow_kg_s=0.1, rod_insertion=1e-5, reactivity=1e-6, omega_rad_s=0.01,
)
EVENT_BUDGET_S = 0.02


def declaration(cases):
    return dict(
        schema_version=1, source=identity(), native_protocol=protocol(),
        driver_sha256=sha(__file__), admission_sha256=sha(Path(__file__).with_name("verify_pwr_horizons.py")),
        cases=list(cases), profiles=PROFILES, absolute_budgets_si=BUDGETS,
        event_time_budget_s=EVENT_BUDGET_S, output_sampling_s=5.0,
        comparison="All identical output times including t=0; exact native event roots separately",
        reference="Finite adaptive Radau calculation; shared equations/EOS, no continuum error bound",
        interpretation="Maximum internal step is a cap, not a fixed step; tolerances refined separately",
        reference_execution={WINDOWED_CASE: dict(reset_interval_s=RESET_INTERVAL_S,
            worker_sha256=sha(Path(__file__).with_name("run_pwr_windowed_reference.py")))} if WINDOWED_CASE in cases else {},
        validated=False, plant_referenced=False,
    )


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_samples(rows, horizon):
    expected = [0.0]
    while expected[-1] < horizon:
        expected.append(min(expected[-1] + 5.0, horizon))
    if len(rows) != len(expected):
        raise ValueError("Missing full-horizon samples")
    for row, t in zip(rows, expected, strict=True):
        if not numeric(row.get("time_s")) or row["time_s"] != t:
            raise ValueError("Nonidentical or incomplete time grid")
        if any(not numeric(row.get(key)) for key in BUDGETS):
            raise ValueError("Nonfinite or missing physical observable")
        if type(row.get("scram_latched")) is not bool:
            raise ValueError("Missing discrete state")


def event_roots(result):
    roots = []
    for transition in result["events"]:
        for event in transition["events"]:
            if not numeric(event.get("t")):
                raise ValueError("Invalid event root")
            roots.append((event["kind"], event["t"]))
        t = transition["protection_event_s"]
        if t is not None:
            if not numeric(t):
                raise ValueError("Invalid protection root")
            roots.append(("protection", t))
    return sorted(roots, key=lambda event: (event[1], event[0]))


def validate_window_journal(execution, history, trace, horizon):
    expected = list(range(int(RESET_INTERVAL_S), int(horizon), int(RESET_INTERVAL_S)))
    if (execution["driver_sha256"] != sha(Path(__file__).with_name("run_pwr_windowed_reference.py"))
        or execution["reset_interval_s"] != RESET_INTERVAL_S
        or execution["source_unchanged"] is not True
        or [r["time_s"] for r in execution["resets"]] != expected):
        raise ValueError("Reference window execution differs")
    by_time = {r["time_s"]: r for r in trace}
    for reset in execution["resets"]:
        t = reset["time_s"]
        if (not numeric(t) or reset["state_sha256_before"] != reset["state_sha256_after"]
            or reset["state_sha256_before"] != by_time[t]["state_sha256"]
            or history[int(t / 5)]["reset_solver_before_step"] is not True):
            raise ValueError("Reference reset changed state or lacks replay journal")


def compare(left, right, left_result, right_result, horizon):
    validate_samples(left, horizon)
    validate_samples(right, horizon)
    differences = {}
    for key, budget in BUDGETS.items():
        errors = [abs(a[key] - b[key]) for a, b in zip(left, right, strict=True)]
        index = max(range(len(errors)), key=errors.__getitem__)
        differences[key] = dict(
            maximum_absolute=errors[index], time_s=left[index]["time_s"], budget=budget,
            budget_fraction=errors[index] / budget, passed=errors[index] <= budget,
            final_absolute=errors[-1],
            peak_value_difference=abs(max(r[key] for r in left) - max(r[key] for r in right)),
            minimum_value_difference=abs(min(r[key] for r in left) - min(r[key] for r in right)),
        )
    a, b = event_roots(left_result), event_roots(right_result)
    if any(not 0 <= root[1] <= horizon for root in [*a, *b]):
        raise ValueError("Event outside comparison horizon")
    topology = [e[0] for e in a] == [e[0] for e in b]
    event_error = max((abs(x[1] - y[1]) for x, y in zip(a, b)), default=0.0) if topology else None
    discrete = all(x["scram_latched"] == y["scram_latched"] for x, y in zip(left, right, strict=True))
    passed = all(d["passed"] for d in differences.values()) and topology and discrete
    passed = passed and event_error is not None and event_error <= EVENT_BUDGET_S
    return dict(
        passed=bool(passed), common_samples=len(left), differences=differences,
        discrete_states_equal=discrete, event_topology_equal=topology,
        event_time_maximum_absolute_s=event_error, left_events=a, reference_events=b,
        sampled_extrema_only=True,
    )


def load_evidence(directory, case, profile, source):
    directory = Path(directory)
    result = admit_result(directory, case, source)
    if result["solver"] != PROFILES[profile]:
        raise ValueError("Solver profile differs")
    samples_path = directory / "observations.jsonl"
    if result.get("observations_sha256") != sha(samples_path):
        raise ValueError("Observation checksum differs")
    rows = [json.loads(line) for line in samples_path.read_text().splitlines()]
    validate_samples(rows, CASES[case]["duration_s"])
    checkpoint = unseal(unseal(read_json(directory / "checkpoint-envelope.json")))
    history = checkpoint["history"]
    if checkpoint["config"]["maximum_internal_step_s"] != PROFILES[profile]["max_step_s"]:
        raise ValueError("Checkpoint internal step differs")
    if len(history) != len(rows) - 1:
        raise ValueError("Sample/checkpoint count differs")
    trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    if len(trace) != len(history):
        raise ValueError("Trace/checkpoint count differs")
    if case == WINDOWED_CASE and profile == "reference":
        validate_window_journal(result["reference_execution"], history, trace, CASES[case]["duration_s"])
    events = [r for r in history if r["events"] or r["protection_event_s"] is not None]
    if events != result["events"]:
        raise ValueError("Event history differs")
    clock = 0.0
    changes = dict(CASES[case]["changes"])
    for entry, row, native in zip(history, rows[1:], trace, strict=True):
        if entry["commands"] != changes.get(clock, {}):
            raise ValueError("Command history differs")
        if any(entry["options"][key] != PROFILES[profile][key] for key in ("method", "rtol", "atol")):
            raise ValueError("Checkpoint solver options differ")
        clock += entry["duration_s"]
        if clock != row["time_s"] or clock != native["time_s"] or entry["state_sha256"] != native["state_sha256"]:
            raise ValueError("Checkpoint clock differs")
    return result, rows


def verify(output, *, cases=DEFAULT_CASES, jobs=2, timeout_s=7200, reuse_from=None, reference_from=None):
    if not cases or len(set(cases)) != len(cases) or any(c not in DEFAULT_CASES for c in cases):
        raise ValueError("Choose distinct supported full-horizon cases")
    if type(jobs) is not int or not 1 <= jobs <= 4:
        raise ValueError("Worker budget 1..4")
    if type(timeout_s) is not int or not 1 <= timeout_s <= 14400:
        raise ValueError("Timeout budget 1..14400 seconds")
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output cannot be a symlink")
    declared = declaration(cases)
    reference = Path(reference_from).absolute() if reference_from is not None else None
    if reference is not None and (not reference.is_dir() or WINDOWED_CASE not in cases):
        raise ValueError("Windowed reference requires an existing directory and load-rejection case")
    reuse = Path(reuse_from).absolute() if reuse_from is not None else None
    if reuse is not None:
        previous = unseal(read_json(reuse / "protocol.json"))
        # The checker may be fixed between runs; every native result is admitted
        # again. Physical source, solver profiles and criteria must be identical.
        # Compare canonical JSON: native CASES contains tuples, disk has lists.
        checked_per_run = {"driver_sha256", "reference_execution"}
        if digest({k: v for k, v in previous.items() if k not in checked_per_run}) != digest({k: v for k, v in declared.items() if k not in checked_per_run}):
            raise ValueError("Reuse requires identical source and refinement protocol")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "protocol.json", seal(declared))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def execute(case, profile):
        started = time.monotonic()
        name = case + "--" + profile
        directory = output / name
        record = dict(case=case, profile=profile, passed=False, evidence_directory=str(directory))
        candidate = reference if reference is not None and case == WINDOWED_CASE and profile == "reference" else (reuse / name if reuse is not None else None)
        if candidate is not None and (candidate / "result.json").is_file():
            try:
                load_evidence(candidate, case, profile, declared["source"])
                record.update(passed=True, evidence_directory=str(candidate), reused=True)
                if candidate == reference:
                    record["reference_worker_sha256"] = declared["reference_execution"][case]["worker_sha256"]
                else:
                    record["previous_checker_sha256"] = previous["driver_sha256"]
                return record
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record["reuse_rejected"] = str(exc)
        settings = PROFILES[profile]
        command = [sys.executable, "-m", "atomic_model_controlled_plant", "run", case, "--output", str(directory)]
        for argument, key in [("--max-step", "max_step_s"), ("--rtol", "rtol"), ("--atol", "atol"), ("--output-step", "output_step_s"), ("--method", "method")]:
            command.extend([argument, str(settings[key])])
        if case == WINDOWED_CASE and profile == "reference":
            if settings != WINDOWED_SETTINGS:
                raise ValueError("Strict windowed solver settings differ")
            command = [sys.executable, "-m", "scripts.run_pwr_windowed_reference", "--output", str(directory)]
        print(name, "started", flush=True)
        try:
            with (output / (name + ".log")).open("xb") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            record["exit_code"] = process.returncode
            if process.returncode:
                raise ValueError("Native full-horizon campaign refused")
            load_evidence(directory, case, profile, declared["source"])
            record["passed"] = True
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            record["error"] = dict(type=type(exc).__name__, message=str(exc))
        record["wall_s"] = time.monotonic() - started
        print(name, "finished", record["passed"], flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # Short cases first; each profile still has its entire declared horizon.
        futures = [pool.submit(execute, case, profile) for case in cases for profile in PROFILES]
        for future in as_completed(futures):
            records.append(future.result())
            atomic_json(output / "progress.json", seal(dict(completed=False, runs=records)))
    comparisons = {}
    for case in cases:
        selected = {r["profile"]: r for r in records if r["case"] == case}
        if not all(r["passed"] for r in selected.values()):
            comparisons[case] = dict(passed=False, reason="Incomplete or refused native campaign")
            continue
        results = {key: load_evidence(r["evidence_directory"], case, key, declared["source"]) for key, r in selected.items()}
        reference, reference_rows = results["reference"]
        pairs = {key: compare(rows, reference_rows, result, reference, CASES[case]["duration_s"]) for key, (result, rows) in results.items() if key != "reference"}
        adjacent = {}
        for coarse, fine in [("cap-2", "cap-1"), ("cap-1", "cap-0.5")]:
            a, ar = results[coarse]
            b, br = results[fine]
            adjacent[coarse + "_vs_" + fine] = compare(ar, br, a, b, CASES[case]["duration_s"])
        accepted = [PROFILES[key]["max_step_s"] for key, result in pairs.items() if result["passed"]]
        comparisons[case] = dict(
            passed=all(r["passed"] for r in [*pairs.values(), *adjacent.values()]),
            versus_tight_reference=pairs, adjacent_caps=adjacent,
            accepted_caps_s=accepted,
            maximum_accepted_cap_s=max(accepted) if accepted else None,
            empirical_order=None,
            order_reason="Adaptive caps at fixed tolerances do not imply fixed-step Richardson order",
        )
    unchanged = declaration(cases) == declared
    report = dict(
        schema_version=1, profile="local-pwr-full-horizon-refinement-v2", completed=True,
        passed=unchanged and all(c["passed"] for c in comparisons.values()),
        complete_default_matrix=list(cases) == list(DEFAULT_CASES), source_unchanged=unchanged,
        protocol_sha256=sha(output / "protocol.json"), runs=records, comparisons=comparisons,
        validated=False, plant_referenced=False,
        scope="Sampled adaptive-step and tolerance sensitivity; strict load reference uses journalled 500s solver-history windows; no unsampled peak or experimental error bound",
    )
    atomic_json(output / "report.json", seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--reference-from", type=Path, help="Completed strict windowed load reference; always readmitted")
    args = parser.parse_args(argv)
    report = verify(args.output, cases=args.cases, jobs=args.jobs, timeout_s=args.timeout, reuse_from=args.reuse_from, reference_from=args.reference_from)
    print(json.dumps(dict(passed=report["passed"], output=str(args.output))))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
