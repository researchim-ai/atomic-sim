"""Recompute the exact two-cycle acceptance matrix and solver-policy differences."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.verify_coupled_exchanger import BUDGET, assess, matrix, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    hashes = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_coupled_exchanger.py"),
        reference_sha256=sha("scripts/coupled_exchanger_references.py"),
        budget_sha256=sha(BUDGET),
    )
    if (
        any(report[key] != value for key, value in hashes.items())
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Coupled evidence source/reference/budget mismatch")
    expected = {(name, policy): spec for name, policy, spec in matrix()}
    identities = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(identities) != len(set(identities))
        or set(identities) != set(expected)
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
        or report["all_cases_accepted"] is not True
    ):
        raise ValueError("Incomplete or duplicated coupled acceptance matrix")
    paths, trajectories = set(), {}
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        identity = record["name"], record["policy"]
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external coupled evidence folder")
        paths.add(path)
        if json.loads((path / "input.json").read_text()) != expected[identity]:
            raise ValueError("Coupled acceptance scenario substitution")
        fresh = assess(path, budget)
        if (
            not fresh["accepted"]
            or fresh["metrics"] != record["metrics"]
            or record["accepted"] is not True
        ):
            raise ValueError("Coupled replay metrics/verdict mismatch")
        trajectories[identity] = json.loads((path / "trajectory.json").read_text())
    comparisons = []
    for name in sorted({name for name, _ in expected}):
        base, tight = trajectories[name, "base"], trajectories[name, "tight"]
        if [(r["time_s"], r["phase"]) for r in base] != [(r["time_s"], r["phase"]) for r in tight]:
            raise ValueError("Different coupled comparison grids")
        difference = 0.0
        for left, right in zip(base, tight):
            difference = max(
                difference,
                float(max(abs(np.array(left["wall_temperature_k"]) - right["wall_temperature_k"]))),
            )
            for side in ("a", "b"):
                for key in ("fluid_temperature_k", "wall_temperature_k"):
                    a, b = left["loops"][side]["loop"][key], right["loops"][side]["loop"][key]
                    difference = max(difference, max(abs(a[node] - b[node]) for node in a))
        if not np.isfinite(difference) or difference > budget["temporal_temperature_absolute_k"]:
            raise ValueError("Coupled temporal refinement budget exceeded")
        comparisons.append(dict(name=name, temperature_absolute_k=difference))
    return dict(
        schema_version=1,
        passed=True,
        source_sha256=hashes["source_sha256"],
        verification_sha256=sha(folder / "verification.json"),
        audit_script_sha256=sha(__file__),
        integrations=len(identities),
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
