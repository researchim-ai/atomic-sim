"""Recompute complete primary/steam qualification, fixed scope and mesh comparisons."""

import argparse
import json
from pathlib import Path

from scripts.qualify_primary_steam import DRIVERS, PHYSICAL_STOPS, compare_levels, matrix, signature
from scripts.verify_primary_steam import BUDGET, assess
from scripts.verify_turbine import sha
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "qualification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    if (
        report["signature"] != signature()
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Primary/steam runtime, driver or budget mismatch")
    for path in DRIVERS:
        if sha(folder / "drivers" / Path(path).name) != report["signature"]["driver_sha256"][path]:
            raise ValueError("Frozen primary/steam driver mismatch")
    expected = {(name, level): spec for name, level, spec in matrix(budget)}
    identities = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(set(identities)) != len(identities)
        or set(identities) != set(expected)
        or report["passed"] is not True
        or report["integrations"] != len(expected)
        or report["case_families"] != len({name for name, _ in expected})
    ):
        raise ValueError("Incomplete or duplicated primary/steam matrix")
    paths = set()
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        identity = record["name"], record["policy"]
        if path in paths or not path.is_relative_to(folder.resolve()):
            raise ValueError("Shared or external primary/steam packet")
        paths.add(path)
        spec = json.loads((path / "input.json").read_text())
        if spec != expected[identity]:
            raise ValueError("Primary/steam scenario or policy substitution")
        if "reused_from" in record and record["reused_from"]["manifest_sha256"] != sha(
            path / "manifest.json"
        ):
            raise ValueError("Primary/steam reuse provenance mismatch")
        raw = json.loads((path / "report.json").read_text())
        status = "domain_exit" if record["name"] in PHYSICAL_STOPS else "completed"
        if (
            raw["status"] != status
            or status == "completed"
            and raw["completed_time_s"] != spec["duration_s"]
        ):
            raise ValueError("Requested primary/steam operating horizon not completed")
        fresh = assess(path, budget)
        if not fresh["accepted"] or any(fresh[key] != record[key] for key in fresh):
            raise ValueError("Primary/steam independent metric replay mismatch")
        print(
            json.dumps(dict(name=record["name"], level=record["policy"], audited=True)), flush=True
        )
    comparisons = compare_levels(report["cases"], budget)
    if comparisons != report["temporal_comparisons"]:
        raise ValueError("Primary/steam temporal replay mismatch")
    if signature() != report["signature"]:
        raise ValueError("Primary/steam signature changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=signature(),
        qualification_sha256=sha(folder / "qualification.json"),
        auditor_sha256=sha(__file__),
        integrations=len(expected),
        case_families=len({name for name, _ in expected}),
        completed_long_horizons_s=budget["long_horizons_s"],
        temporal_comparisons=comparisons,
        maximum_metrics={
            key: max(r["metrics"][key] for r in report["cases"]) for key in budget["limits"]
        },
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
    print(json.dumps({key: result[key] for key in ("passed", "integrations", "case_families")}))


if __name__ == "__main__":
    main()
