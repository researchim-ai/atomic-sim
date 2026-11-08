"""Local full-horizon reduced-PWR campaigns and exact replay; no shortened cases."""

import argparse
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from atomic_model_controlled_plant.qualification import CASES, CRITERIA, protocol
from atomic_model_controlled_plant.session import identity
from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest

DEFAULT_CASES = (
    "nominal",
    "scram",
    "coastdown",
    "load_rejection_equilibrium",
    "feed_disturbance",
    "sink_disturbance",
    "natural",
    "natural_perturbation",
)


def admit_result(directory, name, expected_source):
    directory = Path(directory)
    declared = unseal(read_json(directory / "protocol.json"))
    result = unseal(read_json(directory / "result.json"))
    envelope = unseal(read_json(directory / "checkpoint-envelope.json"))
    checkpoint = unseal(envelope)
    horizon = CASES[name]["duration_s"]
    expected_checks = {
        "energy_relative",
        "mass_relative",
        "boron_relative",
        "feed_nonnegative",
        "primary_scope",
    }
    if name == "nominal":
        expected_checks.add("stationary_fission")
    if name in ("scram", "coastdown"):
        expected_checks.add("shutdown_fission")
    if name.startswith("natural"):
        expected_checks.update(
            ("zero_shaft_and_head", "positive_primary_flow", "buoyancy_drives_friction")
        )
    if name.startswith("load_rejection"):
        expected_checks.update(
            ("settled_final_rate", "settled_power", "settled_primary_flow", "settled_feed_flow")
        )
    if (
        digest(declared) != digest(protocol())
        or declared["source"] != expected_source
        or result["source"] != expected_source
        or result["case"] != name
        or result["protocol_sha256"] != digest(declared)
        or type(result["full_horizon_accepted"]) is not bool
        or not result["full_horizon_accepted"]
        or result["requested_horizon_s"] != horizon
        or result["completed_horizon_s"] != horizon
        or result["failure"] is not None
        or set(result["checks"]) != expected_checks
        or not all(type(value) is bool and value for value in result["checks"].values())
        or result["trace_sha256"] != sha(directory / "trace.jsonl")
        or result["checkpoint_sha256"] != envelope["sha256"]
        or checkpoint["revision"] != result["steps"]
        or result["validated"] is not False
        or result["plant_referenced"] is not False
    ):
        raise ValueError("Incomplete or inconsistent full-horizon evidence: " + name)
    for key in ("energy_relative", "mass_relative", "boron_relative"):
        value = result["ledger"][key]
        if type(value) not in (int, float) or not 0 <= value <= CRITERIA[key]:
            raise ValueError("Horizon ledger budget failed: " + key)
    return result


def evidence_record(directory, name, source):
    result = admit_result(directory, name, source)
    replay_path = directory / "replay.json"
    proof = unseal(read_json(replay_path))
    if (
        proof["replay_equal"] is not True
        or proof["checkpoint_sha256"] != result["checkpoint_sha256"]
        or proof["revision"] != result["steps"]
    ):
        raise ValueError("Replay result differs")
    return dict(
        passed=True,
        completed_horizon_s=result["completed_horizon_s"],
        checks=result["checks"],
        ledger=result["ledger"],
        result_sha256=sha(directory / "result.json"),
        replay_sha256=sha(replay_path),
        evidence_directory=str(directory),
    )


def verify(output, *, cases=DEFAULT_CASES, timeout_s=1800, jobs=1, reuse_from=None):
    if (
        not cases
        or len(set(cases)) != len(cases)
        or any(name not in DEFAULT_CASES for name in cases)
    ):
        raise ValueError("Choose distinct supported full-horizon cases")
    if type(timeout_s) is not int or not 1 <= timeout_s <= 14400:
        raise ValueError("Timeout domain 1..14400 seconds")
    if type(jobs) is not int or not 1 <= jobs <= 4:
        raise ValueError("Local worker budget 1..4")
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output must not be a symlink")
    reuse = Path(reuse_from).absolute() if reuse_from is not None else None
    if reuse is not None and (not reuse.is_dir() or reuse == output):
        raise ValueError("Reuse requires a distinct existing evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    source = identity()
    verifier_sha = sha(__file__)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def execute(name):
        directory = output / name
        started = time.monotonic()
        record = dict(case=name, requested_horizon_s=CASES[name]["duration_s"], passed=False)
        if reuse is not None and (reuse / name / "replay.json").is_file():
            try:
                record.update(evidence_record(reuse / name, name, source))
                record.update(reused_from=str(reuse / name), wall_s=time.monotonic() - started)
                print(name, "reused after evidence/source verification", flush=True)
                return record
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record["reuse_rejected"] = str(exc)
        print(name, CASES[name]["duration_s"], flush=True)
        try:
            with (output / f"{name}.log").open("xb") as log:
                command = [
                    sys.executable,
                    "-m",
                    "atomic_model_controlled_plant",
                    "run",
                    name,
                    "--output",
                    str(directory),
                ]
                run = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s
                )
                record["run_exit_code"] = run.returncode
                if run.returncode:
                    raise ValueError("Campaign process failed")
                admit_result(directory, name, source)
                replay = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "atomic_model_controlled_plant",
                        "replay",
                        str(directory / "checkpoint-envelope.json"),
                        "--output",
                        str(directory / "replay.json"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
                record["replay_exit_code"] = replay.returncode
                if replay.returncode:
                    raise ValueError("Checkpoint replay failed")
                record.update(evidence_record(directory, name, source))
        except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            record["error"] = dict(type=type(exc).__name__, message=str(exc))
        record["wall_s"] = time.monotonic() - started
        return record

    finished = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(execute, name): name for name in cases}
        for future in as_completed(futures):
            record = future.result()
            finished[record["case"]] = record
            records = [finished[name] for name in cases if name in finished]
            atomic_json(output / "progress.json", seal(dict(completed=False, cases=records)))
    records = [finished[name] for name in cases]
    unchanged = source == identity() and verifier_sha == sha(__file__)
    report = dict(
        schema_version=1,
        profile="local-pwr-full-horizons-v2",
        completed=True,
        passed=unchanged and all(r["passed"] for r in records),
        source=source,
        verifier_sha256=verifier_sha,
        source_unchanged=unchanged,
        cases=records,
        selected_cases=list(cases),
        complete_default_matrix=set(cases) == set(DEFAULT_CASES),
        jobs=jobs,
        reuse_from=str(reuse) if reuse is not None else None,
        reuse_policy="Recheck native evidence and prior source-bound replay; do not rerun reused trajectories",
        ci_required=False,
        validated=False,
        plant_referenced=False,
        scope="Full predeclared numerical horizons and source-bound replay; no new experimental validation",
    )
    atomic_json(output / "report.json", seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--phase-timeout-s", type=int, default=1800)
    parser.add_argument("--jobs", type=int, default=1, help="Local independent case workers, 1..4")
    parser.add_argument(
        "--reuse-from", type=Path, help="Recheck and reuse completed source-matching cases"
    )
    args = parser.parse_args(argv)
    output = args.output or str(Path("artifacts") / ("pwr-horizons-" + uuid.uuid4().hex[:10]))
    try:
        report = verify(
            output,
            cases=args.cases,
            timeout_s=args.phase_timeout_s,
            jobs=args.jobs,
            reuse_from=args.reuse_from,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"passed={report['passed']} complete_default_matrix={report['complete_default_matrix']} output={output}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
