"""Profile a single reproducible VVER step without altering physical constants."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

from core.manifest import input_manifest
from core.status import DomainError, SolverError, classify_model_error
from sdk.api import create_session, export_snapshot, restore_snapshot
from sdk.experiment import _source_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--method", choices=("BDF", "Radau"), default="BDF")
    parser.add_argument("--jacobian-relative-step", type=float)
    parser.add_argument("--first-step-s", type=float)
    parser.add_argument("--wall-limit-s", type=float, default=30.0)
    args = parser.parse_args()
    payload = json.loads(args.checkpoint.read_text())
    plant = create_session()
    try:
        restore_snapshot(plant, payload)
    except (DomainError, SolverError) as exc:
        # Холодное обращение EOS при restore может отказать до первого RHS.
        # Сохраняем этот дефект отдельно, не выдавая его за отказ шага.
        report = {
            "status": "checkpoint_error",
            "phase": "restore",
            "error": str(exc),
            "error_class": classify_model_error(exc),
            "source_sha256": _source_fingerprint(),
            "input_sha256": input_manifest(payload)["sha256"],
            "start_time_s": payload["time_s"],
            "completed_time_s": plant.time_s,
            "requested_step_s": args.step_s,
            "method": args.method,
            "rtol": args.rtol,
            "atol": args.atol,
            "jacobian_relative_step": args.jacobian_relative_step,
            "first_step_s": args.first_step_s,
            "profile": [],
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "phase": "restore"}))
        return
    rhs = plant.derivatives
    calls = 0
    started = time.monotonic()

    def counted(t, y):
        nonlocal calls
        calls += 1
        if time.monotonic() - started > args.wall_limit_s:
            raise SolverError("profile wall limit")
        return rhs(t, y)

    plant.derivatives = counted
    profile = cProfile.Profile()
    profile.enable()
    status = "completed"
    error = None
    try:
        plant.advance(
            args.step_s,
            rtol=args.rtol,
            atol=args.atol,
            method=args.method,
            jacobian_relative_step=args.jacobian_relative_step,
            first_step_s=args.first_step_s,
        )
    except (DomainError, SolverError) as exc:
        status = classify_model_error(exc) + "_error"
        error = str(exc)
    finally:
        profile.disable()
    elapsed = time.monotonic() - started
    rows = []
    for (file, line, name), (primitive, total, own, cumulative, _callers) in pstats.Stats(
        profile
    ).stats.items():
        rows.append(
            {
                "function": name,
                "file": file,
                "line": line,
                "calls": total,
                "own_s": own,
                "cumulative_s": cumulative,
            }
        )
    rows.sort(key=lambda row: row["cumulative_s"], reverse=True)
    report = {
        "status": status,
        "error": error,
        "source_sha256": _source_fingerprint(),
        "input_sha256": input_manifest(payload)["sha256"],
        "start_time_s": payload["time_s"],
        "completed_time_s": plant.time_s,
        "requested_step_s": args.step_s,
        "rtol": args.rtol,
        "atol": args.atol,
        "method": args.method,
        "jacobian_relative_step": args.jacobian_relative_step,
        "first_step_s": args.first_step_s,
        "wall_time_s": elapsed,
        "rhs_calls_including_jacobian_and_ledger": calls,
        "profile": rows[:25],
        "snapshot": export_snapshot(plant),
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "wall_time_s", "rhs_calls_including_jacobian_and_ledger")
            }
        )
    )


if __name__ == "__main__":
    main()
