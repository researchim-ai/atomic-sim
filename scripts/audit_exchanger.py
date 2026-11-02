"""Recompute complete exchanger evidence, temporal subdivision and spatial checks."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.verify_exchanger import BUDGET, assess, matrix, sha, spatial_refinement
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    hashes = {
        "source_sha256": _source_fingerprint(),
        "script_sha256": sha("scripts/verify_exchanger.py"),
        "reference_sha256": sha("scripts/exchanger_references.py"),
        "budget_sha256": sha(BUDGET),
    }
    if (
        any(report[k] != value for k, value in hashes.items())
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Exchanger source/reference/budget mismatch")
    expected = {(name, policy): spec for name, policy, spec in matrix()}
    records = report["cases"]
    identities = [(r["name"], r["policy"]) for r in records]
    if (
        len(set(identities)) != len(identities)
        or set(identities) != set(expected)
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
        or not report["all_cases_accepted"]
    ):
        raise ValueError("Incomplete or duplicated exchanger acceptance matrix")
    paths, trajectories = set(), {}
    for record in records:
        path = Path(record["folder"]).resolve()
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external exchanger evidence folder")
        paths.add(path)
        identity = record["name"], record["policy"]
        if json.loads((path / "input.json").read_text()) != expected[identity]:
            raise ValueError("Exchanger scenario substitution")
        fresh = assess(path, budget)
        if (
            not fresh["accepted"]
            or fresh["metrics"] != record["metrics"]
            or record["accepted"] is not True
        ):
            raise ValueError("Exchanger recomputed metrics/verdict mismatch")
        trajectories[identity] = json.loads((path / "trajectory.json").read_text())
    comparisons = []
    for name in sorted({name for name, _ in expected}):
        fine = {(r["time_s"], r["phase"]): r for r in trajectories[name, "subdivided"]}
        difference = 0.0
        for row in trajectories[name, "base"]:
            other = fine[row["time_s"], row["phase"]]
            for side in ("a", "b", "wall"):
                difference = max(
                    difference,
                    float(
                        max(
                            abs(np.array(row["temperature_k"][side]) - other["temperature_k"][side])
                        )
                    ),
                )
        if not np.isfinite(difference) or difference > budget["temporal_temperature_absolute_k"]:
            raise ValueError("Exchanger temporal subdivision budget exceeded")
        comparisons.append(dict(name=name, temperature_absolute_k=difference))
    spatial = spatial_refinement(budget)
    if not spatial["passed"] or spatial != report["spatial_refinement"]:
        raise ValueError("Exchanger spatial reference mismatch")
    return dict(
        schema_version=1,
        passed=True,
        source_sha256=hashes["source_sha256"],
        verification_sha256=sha(folder / "verification.json"),
        audit_script_sha256=sha(__file__),
        integrations=len(records),
        case_families=len(comparisons),
        temporal_comparisons=comparisons,
        spatial_refinement=spatial,
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
