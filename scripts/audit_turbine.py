"""Replay the complete steam-train matrix and output-step subdivision."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.verify_turbine import BUDGET, assess, matrix, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    hashes = dict(
        source_sha256=_source_fingerprint(),
        script_sha256=sha("scripts/verify_turbine.py"),
        reference_sha256=sha("scripts/turbine_references.py"),
        budget_sha256=sha(BUDGET),
    )
    if (
        any(report[key] != value for key, value in hashes.items())
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Turbine source/reference/budget mismatch")
    expected = {(name, policy): spec for name, policy, spec in matrix()}
    identities = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(identities) != len(set(identities))
        or set(identities) != set(expected)
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
        or report["all_cases_accepted"] is not True
    ):
        raise ValueError("Incomplete or duplicated turbine acceptance matrix")
    paths, trajectories = set(), {}
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        identity = record["name"], record["policy"]
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external turbine bundle folder")
        paths.add(path)
        if json.loads((path / "input.json").read_text()) != expected[identity]:
            raise ValueError("Turbine scenario substitution")
        fresh = assess(path, budget)
        if (
            not fresh["accepted"]
            or record["accepted"] is not True
            or fresh["metrics"] != record["metrics"]
            or fresh["termination"] != record["termination"]
        ):
            raise ValueError("Turbine replay metrics/termination mismatch")
        trajectories[identity] = json.loads((path / "trajectory.json").read_text())
    comparisons = []
    for name in sorted({name for name, _ in expected}):
        base, fine = trajectories[name, "base"], trajectories[name, "subdivided"]
        lookup = {(r["time_s"], r["phase"]): r for r in fine}
        error = 0.0
        for row in base:
            if row["phase"] == "domain_exit":
                other = fine[-1]
                if (
                    other["phase"] != "domain_exit"
                    or abs(other["time_s"] - row["time_s"]) > budget["boundary_time_absolute_s"]
                ):
                    raise ValueError("Turbine subdivision changes physical termination")
            else:
                other = lookup[row["time_s"], row["phase"]]
            error = max(error, abs(row["omega_rad_s"] - other["omega_rad_s"]))
        if not np.isfinite(error) or error > budget["temporal_omega_absolute_rad_s"]:
            raise ValueError("Turbine temporal subdivision exceeds budget")
        comparisons.append(dict(name=name, omega_absolute_rad_s=error))
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
