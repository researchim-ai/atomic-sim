"""Replay the exact dynamic steam-cycle matrix and time-refinement evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.verify_steam_cycle import BUDGET, assess, matrix, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    hashes = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_steam_cycle.py"),
        reference_sha256=sha("scripts/steam_cycle_references.py"),
        helper_sha256=sha("scripts/verify_turbine.py"),
        budget_sha256=sha(BUDGET),
    )
    if (
        any(report[key] != value for key, value in hashes.items())
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Steam-cycle source/reference/budget mismatch")
    expected = {(name, policy): spec for name, policy, spec in matrix()}
    ids = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(expected)
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
        or report["all_cases_accepted"] is not True
    ):
        raise ValueError("Incomplete or duplicated steam-cycle matrix")
    paths, trajectories = set(), {}
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        identity = record["name"], record["policy"]
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external steam-cycle folder")
        paths.add(path)
        if json.loads((path / "input.json").read_text()) != expected[identity]:
            raise ValueError("Steam-cycle scenario substitution")
        fresh = assess(path, budget)
        if not fresh["accepted"] or any(
            fresh[key] != record[key] for key in ("accepted", "metrics", "termination")
        ):
            raise ValueError("Steam-cycle replay metrics/verdict mismatch")
        trajectories[identity] = json.loads((path / "trajectory.json").read_text())
    comparisons = []
    for name in sorted({name for name, _ in expected}):
        left, right = trajectories[name, "base"], trajectories[name, "tight"]
        if len(left) != len(right):
            raise ValueError("Different steam-cycle comparison lengths")
        error = 0.0
        for a, b in zip(left, right):
            if (
                a["phase"] != b["phase"]
                or abs(a["time_s"] - b["time_s"]) > budget["boundary_time_absolute_s"]
            ):
                raise ValueError("Different steam-cycle refinement grids or physical stops")
            for vessel in ("steam_generator", "condenser"):
                error = max(
                    error,
                    abs(a["vessels"][vessel]["pressure_pa"] - b["vessels"][vessel]["pressure_pa"]),
                )
        if not np.isfinite(error) or error > budget["temporal_pressure_absolute_pa"]:
            raise ValueError("Steam-cycle temporal refinement exceeded budget")
        comparisons.append(dict(name=name, pressure_absolute_pa=error))
    return dict(
        schema_version=1,
        passed=True,
        source_sha256=hashes["source_sha256"],
        verification_sha256=sha(folder / "verification.json"),
        audit_script_sha256=sha(__file__),
        integrations=len(ids),
        case_families=len(comparisons),
        temporal_comparisons=comparisons,
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    write(args.output, result)
    print(json.dumps(dict(passed=result["passed"], integrations=result["integrations"])))


if __name__ == "__main__":
    main()
