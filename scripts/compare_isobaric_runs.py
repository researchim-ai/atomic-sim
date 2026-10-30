"""Сопоставление полных траекторий одного открытого канала на общей сетке."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.isobaric import IsobaricChannel
from physics.hydraulics.wall_channel import WallChannel


def load(folder, horizon):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name in ("input.json", "report.json", "trajectory.csv", "checkpoint.json"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest["artifacts"][name]:
            raise ValueError("Checksum mismatch: " + str(folder / name))
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if (
        spec["engine"]
        not in ("isobaric-channel", "isobaric-wall-channel", "pressure-driven-wall-channel")
        or spec["duration_s"] != horizon
        or report["status"] != "completed"
        or report["requested_duration_s"] != horizon
        or not report["numerical_acceptance"]
        or report["completed_time_s"] != horizon
    ):
        raise ValueError("Channel run has not passed the requested horizon")
    with (folder / "trajectory.csv").open() as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    times = [row["time_s"] for row in rows]
    if (
        not rows
        or times[0] != 0.0
        or times[-1] != horizon
        or any(b <= a for a, b in zip(times, times[1:]))
        or any(not math.isfinite(value) for row in rows for value in row.values())
    ):
        raise ValueError("Invalid channel trajectory")
    payload = json.loads((folder / "checkpoint.json").read_text())
    model = {
        "isobaric-channel": IsobaricChannel,
        "isobaric-wall-channel": WallChannel,
        "pressure-driven-wall-channel": DynamicWallChannel,
    }[spec["engine"]]
    if payload["time_s"] != horizon or model.from_snapshot(payload).snapshot() != payload:
        raise ValueError("Channel checkpoint does not restore exactly")
    recomputed = {
        "mass_relative": max(abs(row["mass_residual_kg"]) for row in rows)
        / max(abs(rows[0]["stored_mass_kg"]), 1e-9),
        "energy_relative": max(abs(row["energy_residual_j"]) for row in rows)
        / max(abs(rows[0]["stored_energy_j"]), 1.0),
    }
    if not all(
        math.isclose(value, report["balance"][key], rel_tol=1e-12, abs_tol=1e-15)
        for key, value in recomputed.items()
    ):
        raise ValueError("CSV and declared balance disagree")
    return spec, report, {row["time_s"]: row for row in rows}, recomputed, manifest


def compare(run, reference, budgets):
    budget_path = Path(budgets)
    budget = json.loads(budget_path.read_text())
    a = load(run, budget["horizon_s"])
    b = load(reference, budget["horizon_s"])
    if (
        a[0]["engine"] != b[0]["engine"]
        or a[0]["channel"] != b[0]["channel"]
        or a[0].get("wall") != b[0].get("wall")
    ):
        raise ValueError("Comparison requires identical channel geometry and prescribed drives")
    if set(a[2]) - set(b[2]):
        raise ValueError("Reference must contain every comparison time")
    fields = {
        "outlet_enthalpy_relative": ("outlet_enthalpy_j_kg", True),
        "outlet_quality_absolute": ("outlet_quality", False),
        "stored_mass_relative": ("stored_mass_kg", True),
    }
    if a[0]["engine"] == "isobaric-wall-channel":
        fields["wall_temperature_absolute_k"] = ("max_wall_temperature_k", False)
    measured = {
        key: max(
            abs(row[field] - b[2][t][field])
            / (max(abs(b[2][t][field]), 1e-12) if relative else 1.0)
            for t, row in a[2].items()
        )
        for key, (field, relative) in fields.items()
    }
    checks = {key: value <= budget[key] for key, value in measured.items()}
    for key in ("mass_relative", "energy_relative"):
        checks[key] = a[3][key] <= budget[key] and b[3][key] <= budget[key]
    return {
        "schema_version": 1,
        "kind": "numerical_verification",
        "validated": False,
        "plant_referenced": False,
        "horizon_s": budget["horizon_s"],
        "comparison_times": len(a[2]),
        "budget_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        "runs": {str(run): a[1], str(reference): b[1]},
        "source_hashes": [a[4]["source_sha256"], b[4]["source_sha256"]],
        "measured": measured,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "Open prescribed-pressure channel, not VVER coastdown acceptance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--budgets", type=Path, default=Path("validation/isobaric_comparison_budget.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run, args.reference, args.budgets)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "measured": result["measured"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
