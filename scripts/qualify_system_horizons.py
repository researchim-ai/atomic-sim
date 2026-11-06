"""Reproducible legacy-plant horizon campaign preserving failed full requests.

Numerical comparison only. No task/physical promotion or edits to the plant.
"""

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from atomic_model_hem_throughflow.__main__ import publish
from atomic_model_workspace.common import digest
from sdk.experiment import _source_fingerprint, run_experiment, validate_experiment

CASES = [
    ("nominal_coarse", "vver_nominal_hour_central.yaml", {}),
    ("nominal_tight", "vver_nominal_hour_central_tight.yaml", {}),
    ("nominal_radau", "vver_nominal_hour_central_tight.yaml", {"method": "Radau"}),
    ("scram_coarse", "vver_scram_60s.yaml", {}),
    ("scram_tight", "vver_scram_60s_tight.yaml", {}),
    ("scram_quarter", "vver_scram_60s_tight.yaml", {"step_s": 0.25}),
    ("protected_coastdown_1800", "vver_protected_coastdown_120s.yaml", {"duration_s": 1800.0}),
    ("coastdown_refined_1800", "vver_protected_coastdown_120s_tight.yaml", {"duration_s": 1800.0}),
    (
        "load_rejection",
        "vver_scram_60s_tight.yaml",
        {"duration_s": 600.0, "timeline": [{"t": 10.0, "throttle": 0.0, "scram": True}]},
    ),
    (
        "feed_change",
        "vver_scram_60s_tight.yaml",
        {"duration_s": 120.0, "timeline": [{"t": 10.0, "feed_bias_kg_s": -50.0}]},
    ),
    (
        "sink_change",
        "vver_scram_60s_tight.yaml",
        {"duration_s": 120.0, "timeline": [{"t": 10.0, "sink_temperature_k": 298.15}]},
    ),
]
SCALES = dict(
    fission_power_w=3.1e9,
    thermal_power_w=3.1e9,
    fuel_temperature_k=1000.0,
    primary_pressure_pa=16e6,
    sg_pressure_pa=6.4e6,
    mass_flow_kg_s=18000.0,
    rod_insertion=1.0,
    reactivity=0.01,
)
CRITERIA = dict(max_scaled_common_time_difference=1e-4, exact_rod_stop_difference_s=1e-8)
FILES = {
    "input.json",
    "checkpoint.json",
    "manifest.json",
    "report.json",
    "report.md",
    "trajectory.csv",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    return dict(
        plant_source_sha256=_source_fingerprint(),
        driver_sha256=sha(__file__),
        scenarios={file: sha(Path("scenarios") / file) for _, file, _ in CASES},
    )


def requests():
    result = {}
    for name, file, changes in CASES:
        spec = yaml.safe_load((Path("scenarios") / file).read_text())
        spec["name"] = name
        changes = deepcopy(changes)
        method = changes.pop("method", None)
        if method:
            spec["solver"]["method"] = method
        spec.update(changes)
        if name == "scram_quarter":
            spec["solver"]["first_step_s"] = 0.25
        result[name] = validate_experiment(spec)
    return result


def common_time_difference(left, right):
    a = {float(row["time_s"]): row for row in left}
    b = {float(row["time_s"]): row for row in right}
    times = sorted(set(a) & set(b))
    if not times or times[0] != 0 or times[-1] != max(a) or times[-1] != max(b):
        raise ValueError("Comparison lacks a common full horizon")
    values = {
        field: max(abs(float(a[t][field]) - float(b[t][field])) / scale for t in times)
        for field, scale in SCALES.items()
    }
    return dict(
        common_times=len(times),
        max_scaled_differences=values,
        numerical_refinement_passed=max(values.values())
        <= CRITERIA["max_scaled_common_time_difference"],
    )


def inspect_case(folder, spec):
    folder = Path(folder)
    if (
        folder.is_symlink()
        or {p.name for p in folder.iterdir()} != FILES
        or any(p.is_symlink() for p in folder.iterdir())
    ):
        raise ValueError("Case inventory mismatch")
    report = json.loads((folder / "report.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        json.loads((folder / "input.json").read_text()) != spec
        or manifest["source_sha256"] != _source_fingerprint()
    ):
        raise ValueError("Case input/plant source mismatch")
    rows = list(csv.DictReader((folder / "trajectory.csv").open(newline="")))
    if not rows or any(
        not np.isfinite(float(row[field])) for row in rows for field in ["time_s", *SCALES]
    ):
        raise ValueError("Nonfinite or empty trajectory")
    stamps = [float(row["time_s"]) for row in rows]
    if (
        stamps[0] != 0
        or any(a >= b for a, b in zip(stamps, stamps[1:]))
        or stamps[-1] != report["completed_time_s"]
    ):
        raise ValueError("Trajectory clock mismatch")
    if report["requested_duration_s"] != spec["duration_s"]:
        raise ValueError("Requested horizon changed")
    complete = report["status"] == "completed" and report["completed_time_s"] == spec["duration_s"]
    actual_checks = {k: report["balance"][k] <= v for k, v in spec["budgets"].items()}
    if actual_checks != report["budget_checks"] or report["numerical_acceptance"] != (
        complete and all(actual_checks.values())
    ):
        raise ValueError("Numerical acceptance inconsistent with full horizon/balances")
    record = dict(
        name=spec["name"],
        status=report["status"],
        requested_duration_s=spec["duration_s"],
        completed_time_s=report["completed_time_s"],
        full_horizon_completed=complete,
        numerical_acceptance=report["numerical_acceptance"],
        balance=report["balance"],
        budget_checks=actual_checks,
        error=report["error"],
        events=report["events"],
        files={p.name: sha(p) for p in sorted(folder.iterdir())},
    )
    return record, rows


def assess(folder):
    rows = {}
    cases = {}
    for name, spec in requests().items():
        cases[name], rows[name] = inspect_case(Path(folder) / name, spec)
    pairs = {}
    for a, b in [
        ("nominal_coarse", "nominal_tight"),
        ("nominal_tight", "nominal_radau"),
        ("scram_coarse", "scram_tight"),
        ("scram_tight", "scram_quarter"),
    ]:
        pairs[a + "/" + b] = common_time_difference(rows[a], rows[b])
    scram_stops = {
        name: [e["t"] for e in cases[name]["events"] if e["kind"] == "rod_stop"]
        for name in ["scram_coarse", "scram_tight", "scram_quarter"]
    }
    stops = list(scram_stops.values())
    stop_agreement = (
        all(len(s) == 1 for s in stops)
        and max(s[0] for s in stops) - min(s[0] for s in stops)
        <= CRITERIA["exact_rod_stop_difference_s"]
    )
    # Existing ledger intentionally leaves its historical acceptance flag false.
    # Campaign completion is a separate explicit field, never rewritten in-place.
    return dict(
        schema_version=1,
        source=identity(),
        criteria=CRITERIA,
        normalization_scales=SCALES,
        assessment_scope="Numerical legacy-passport horizon completion/refinement only; no extension of phase/equipment scope",
        comparative_criteria_status="New numerical comparison protocol, not prospective experimental validation",
        cases=cases,
        comparisons=pairs,
        scram_rod_stop_times=scram_stops,
        scram_rod_stop_agreement=stop_agreement,
        completed_cases=sum(c["full_horizon_completed"] for c in cases.values()),
        failed_cases=[name for name, c in cases.items() if not c["full_horizon_completed"]],
        all_requested_blocks_closed=False,
        validated=False,
        plant_referenced=False,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ["run", "assess", "audit"]:
        sub.add_parser(cmd).add_argument("folder", type=Path)
    args = p.parse_args(argv)
    try:
        path = args.folder / "campaign.json"
        if args.command == "run":
            args.folder.mkdir(parents=True, exist_ok=False)
            for name, spec in requests().items():
                run_experiment(spec, args.folder / name, wall_limit_s=180)
        if args.command in ["run", "assess"]:
            if path.exists():
                raise FileExistsError(path)
            result = assess(args.folder)
            publish(path, dict(payload=result, sha256=digest(result)))
        else:
            saved = json.loads(path.read_text())
            if (
                set(saved) != {"payload", "sha256"}
                or digest(saved["payload"]) != saved["sha256"]
                or saved["payload"]["source"] != identity()
            ):
                raise ValueError("Campaign envelope/source mismatch")
            result = assess(args.folder)
            if digest(result) != saved["sha256"]:
                raise ValueError("Campaign recomputation differs")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    print(dict(completed=result["completed_cases"], failed=result["failed_cases"], validated=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
