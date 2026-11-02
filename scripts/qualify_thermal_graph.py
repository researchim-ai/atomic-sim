"""Frozen, source-bound independent verification of conservative thermal graphs."""

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy

from scripts.solve_hydraulic_graph import BUDGET as HYDRAULIC_BUDGET
from scripts.solve_hydraulic_graph import sha
from scripts.solve_thermal_graph import BUDGET, solve_thermal_graph
from scripts.thermal_graph_references import cases, reference
from sdk.experiment import _source_fingerprint

DRIVERS = (
    "scripts/solve_thermal_graph.py",
    "scripts/solve_hydraulic_graph.py",
    "scripts/thermal_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/qualify_thermal_graph.py",
)


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={p: sha(p) for p in DRIVERS},
        budgets={str(p): sha(p) for p in (BUDGET, HYDRAULIC_BUDGET)},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assess(spec, solution, exact):
    oracle = reference(spec)
    budget = json.loads(BUDGET.read_text())
    metrics = {
        label: max(
            abs(solution["temperatures_k"][k] - t) for k, t in expected["temperatures_k"].items()
        )
        for label, expected in [("reference_error_k", oracle), ("manufactured_error_k", exact)]
    }
    if any(
        not np.isfinite(v) or v > budget["maximum_reference_temperature_error_k"]
        for v in metrics.values()
    ):
        raise ValueError("Thermal graph differs from independent reference/manufactured point")
    return metrics


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    (folder / "drivers").mkdir()
    (folder / "budgets").mkdir()
    for path in DRIVERS:
        shutil.copyfile(path, folder / "drivers" / Path(path).name)
    for path in (BUDGET, HYDRAULIC_BUDGET):
        shutil.copyfile(path, folder / "budgets" / path.name)
    records = []
    for name, spec, exact in cases():
        solved = solve_thermal_graph(spec)
        metrics = assess(spec, solved, exact)
        path = folder / name
        path.mkdir()
        for filename, value in [
            ("request.json", spec),
            ("expected.json", exact),
            ("solution.json", solved),
        ]:
            write(path / filename, value)
        records.append(
            dict(name=name, metrics=metrics, files={p.name: sha(p) for p in sorted(path.iterdir())})
        )
        print(json.dumps(dict(name=name, accepted=True)), flush=True)
    if signature() != captured:
        raise ValueError("Thermal signature changed during qualification")
    report = dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=records,
        validated=False,
        plant_referenced=False,
    )
    write(folder / "qualification.json", report)
    return report


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "qualification.json").read_text())
    captured = signature()
    if report.get("passed") is not True or report["signature"] != captured:
        raise ValueError("Thermal signature mismatch")
    for group in ("drivers", "budgets"):
        for path, digest in captured[group].items():
            if sha(folder / group / Path(path).name) != digest:
                raise ValueError("Frozen thermal " + group + " mismatch")
    expected = list(cases())
    if [c["name"] for c in report["cases"]] != [c[0] for c in expected]:
        raise ValueError("Incomplete thermal qualification matrix")
    for record, (name, spec, exact) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "expected.json", "solution.json"}:
            raise ValueError("Incomplete thermal packet")
        for filename, digest in record["files"].items():
            if sha(path / filename) != digest:
                raise ValueError("Thermal packet integrity mismatch")
        if (
            json.loads((path / "request.json").read_text()) != spec
            or json.loads((path / "expected.json").read_text()) != exact
        ):
            raise ValueError("Thermal fixture mismatch")
        solved = solve_thermal_graph(spec)
        if (
            solved != json.loads((path / "solution.json").read_text())
            or assess(spec, solved, exact) != record["metrics"]
        ):
            raise ValueError("Thermal numerical audit differs")
    if signature() != captured:
        raise ValueError("Thermal signature changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        maximum_nodes=max(len(s["hydraulic"]["nodes"]) for _, s, _ in expected),
        qualification_sha256=sha(folder / "qualification.json"),
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        write(args.output, audit(args.audit))
    else:
        qualify(args.output)


if __name__ == "__main__":
    main()
