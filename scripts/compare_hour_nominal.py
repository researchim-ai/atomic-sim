"""Compare complete runs on common output times against fixed budgets.

Usage: python -m scripts.compare_hour_nominal RUN REFERENCE --output result.json
Does not interpolate missing reference points or treat truncated runs as passes.
Command timelines require --allow-commands and must match exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FIELDS = {
    "thermal_power_relative": ("thermal_power_w", True),
    "fuel_temperature_absolute_k": ("fuel_temperature_k", False),
    "primary_pressure_absolute_pa": ("primary_pressure_pa", False),
    "sg_pressure_absolute_pa": ("sg_pressure_pa", False),
    "mass_flow_absolute_kg_s": ("mass_flow_kg_s", False),
}


def load_run(folder, horizon, *, allow_commands=False, extra_columns=()):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("report.json", "trajectory.csv", "input.json"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest["artifacts"][name]:
            raise ValueError("Checksum mismatch: %s" % (folder / name))
    report = json.loads((folder / "report.json").read_text())
    spec = json.loads((folder / "input.json").read_text())
    if spec["timeline"] and not allow_commands:
        raise ValueError("Nominal comparison requires an empty command timeline")
    if (
        report["status"] != "completed"
        or not report["numerical_acceptance"]
        or report["completed_time_s"] != horizon
        or spec["duration_s"] != horizon
    ):
        raise ValueError("Run has not passed the complete requested horizon")
    with (folder / "trajectory.csv").open() as stream:
        columns = {"time_s", *(field for field, _ in FIELDS.values()), *extra_columns}
        rows = [{key: float(row[key]) for key in columns} for row in csv.DictReader(stream)]
    times = [row["time_s"] for row in rows]
    if (
        not rows
        or times[0] != 0
        or times[-1] != horizon
        or any(b <= a for a, b in zip(times, times[1:]))
        or any(not math.isfinite(v) for row in rows for v in row.values())
    ):
        raise ValueError("Incomplete, non-monotone or non-finite trajectory")
    return report, {row["time_s"]: row for row in rows}


def compare(run, reference, budget_path, *, allow_commands=False):
    budget = json.loads(Path(budget_path).read_text())
    horizon = budget["horizon_s"]
    state_bounds = budget.get("state_bounds", {})
    report, rows = load_run(run, horizon, allow_commands=allow_commands, extra_columns=state_bounds)
    ref_report, ref_rows = load_run(
        reference, horizon, allow_commands=allow_commands, extra_columns=state_bounds
    )
    inputs = [json.loads((Path(folder) / "input.json").read_text()) for folder in (run, reference)]
    if inputs[0]["timeline"] != inputs[1]["timeline"]:
        raise ValueError("Comparison requires identical command timelines")
    if set(rows) - set(ref_rows):
        raise ValueError("Reference must contain every requested comparison time")
    measured = {}
    for key, (field, relative) in FIELDS.items():
        measured[key] = max(
            abs(row[field] - ref_rows[t][field])
            / (max(abs(ref_rows[t][field]), 1.0) if relative else 1.0)
            for t, row in rows.items()
        )
    checks = {key: measured[key] <= value for key, value in budget["observable_budgets"].items()}
    for field, (low, high) in state_bounds.items():
        checks["bounds_" + field] = all(
            low <= row[field] <= high for dataset in (rows, ref_rows) for row in dataset.values()
        )
    for index, expected in enumerate(budget.get("event_expectations", [])):
        matching = [
            [event for event in item.get("events", []) if event["kind"] == expected["kind"]]
            for item in (report, ref_report)
        ]
        checks["event_%d_%s" % (index, expected["kind"])] = all(
            len(events) == 1
            and abs(events[0]["t"] - expected["t"]) <= expected["time_tolerance_s"]
            and all(
                events[0].get(key) == value for key, value in expected.get("values", {}).items()
            )
            for events in matching
        )
    return {
        "schema_version": 1,
        "kind": "numerical_verification",
        "validated": False,
        "plant_referenced": False,
        "budget_sha256": hashlib.sha256(Path(budget_path).read_bytes()).hexdigest(),
        "runs": {str(run): report, str(reference): ref_report},
        "comparison_times": len(rows),
        "horizon_s": horizon,
        "measured": measured,
        "checks": checks,
        "numerical_hour_accepted": horizon == 3600
        and not inputs[0]["timeline"]
        and bool(checks)
        and all(checks.values()),
        "numerical_comparison_accepted": bool(checks) and all(checks.values()),
        "allow_commands": allow_commands,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--budgets", type=Path, default=Path("validation/hour_nominal_budget.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-commands", action="store_true")
    args = parser.parse_args()
    result = compare(args.run, args.reference, args.budgets, allow_commands=args.allow_commands)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"numerical_comparison_accepted": result["numerical_comparison_accepted"]}))
    if not result["numerical_comparison_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
