"""Recompute long-horizon, three-level steam-cycle qualification and real solver meshes."""

import argparse
import json
from pathlib import Path

from scripts.qualify_steam_cycle import BUDGET, DRIVERS, compare_levels, matrix, signature, trace
from scripts.verify_steam_cycle import assess
from scripts.verify_turbine import sha
from sdk.network_experiment import write

PHYSICAL_STOP_FAMILIES = {"speed_margin", "phase_margin", "pressure_margin"}


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "qualification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    if (
        report["signature"] != signature()
        or json.loads((folder / "budget.json").read_text()) != budget
        or sha(budget["base_budget_file"]) != budget["base_budget_sha256"]
        or json.loads(Path(budget["base_budget_file"]).read_text()) != budget["base_limits"]
    ):
        raise ValueError("Refinement runtime, criteria or driver mismatch")
    for path in DRIVERS:
        if sha(folder / "drivers" / Path(path).name) != report["signature"]["driver_sha256"][path]:
            raise ValueError("Frozen qualification driver mismatch")
    expected = {(name, level): spec for name, level, spec in matrix(budget)}
    identities = [(c["name"], c["policy"]) for c in report["cases"]]
    if (
        len(identities) != len(set(identities))
        or set(identities) != set(expected)
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
        or report["passed"] is not True
    ):
        raise ValueError("Incomplete/duplicated three-level qualification matrix")
    paths = set()
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        identity = record["name"], record["policy"]
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external qualification folder")
        paths.add(path)
        if "reused_from" in record and record["reused_from"]["manifest_sha256"] != sha(
            path / "manifest.json"
        ):
            raise ValueError("Reused packet provenance checksum mismatch")
        spec = json.loads((path / "input.json").read_text())
        if spec != expected[identity]:
            raise ValueError("Qualification scenario/policy substitution")
        actual = json.loads((path / "report.json").read_text())
        expected_status = "domain_exit" if record["name"] in PHYSICAL_STOP_FAMILIES else "completed"
        if actual["status"] != expected_status or (
            expected_status == "completed" and actual["completed_time_s"] != spec["duration_s"]
        ):
            raise ValueError("Requested operating horizon was not completed")
        fresh = assess(path, budget["base_limits"])
        if (
            fresh["accepted"] is not True
            or any(fresh[k] != record[k] for k in ("accepted", "metrics", "termination"))
            or trace(path) != record["solver_statistics"]
        ):
            raise ValueError("Recomputed metrics/mesh statistics mismatch")
    comparisons = compare_levels(report["cases"], budget)
    if comparisons != report["temporal_comparisons"]:
        raise ValueError("Temporal comparison verdict differs from replay")
    profiles = []
    for name in sorted({name for name, _ in expected}):
        levels = {c["policy"]: c for c in report["cases"] if c["name"] == name}
        profiles.append(
            dict(
                name=name,
                reference_errors={level: levels[level]["metrics"] for level in budget["levels"]},
                solver_statistics={
                    level: levels[level]["solver_statistics"] for level in budget["levels"]
                },
            )
        )
    return dict(
        schema_version=1,
        passed=True,
        signature=signature(),
        qualification_sha256=sha(folder / "qualification.json"),
        auditor_sha256=sha(__file__),
        integrations=len(identities),
        case_families=len(profiles),
        levels=budget["levels"],
        temporal_comparisons=comparisons,
        reference_error_profiles=profiles,
        completed_long_horizons_s=budget["long_horizons_s"],
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
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                integrations=result["integrations"],
                case_families=result["case_families"],
            )
        )
    )


if __name__ == "__main__":
    main()
