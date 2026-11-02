"""Recompute steady graph balances and independent mixed-system references."""

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy

from scripts.hydraulic_graph_references import cases, reference
from scripts.solve_hydraulic_graph import BUDGET, sha, solve_hydraulic_graph
from sdk.experiment import _source_fingerprint

DRIVERS = (
    "scripts/solve_hydraulic_graph.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/qualify_hydraulic_graph.py",
)


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={p: sha(p) for p in DRIVERS},
        budget_sha256=sha(BUDGET),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assess(spec, solution, exact, budget):
    oracle = reference(spec)
    metrics = {}
    for key, unit, criterion in [
        ("pressures_pa", "pa", "maximum_reference_pressure_error_pa"),
        ("flows_kg_s", "kg_s", "maximum_reference_flow_error_kg_s"),
    ]:
        for name, expected in [("reference", oracle), ("manufactured", exact)]:
            error = max(abs(solution[key][k] - v) for k, v in expected[key].items())
            if error > budget[criterion]:
                raise ValueError("Graph disagrees with " + name + " " + key)
            metrics[name + "_" + unit] = error
    return metrics


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    budget = json.loads(BUDGET.read_text())
    write(folder / "budget.json", budget)
    (folder / "drivers").mkdir()
    for path in DRIVERS:
        shutil.copyfile(path, folder / "drivers" / Path(path).name)
    records = []
    for name, spec, exact in cases():
        solution = solve_hydraulic_graph(spec)
        metrics = assess(spec, solution, exact, budget)
        path = folder / name
        path.mkdir()
        for filename, value in [
            ("request.json", spec),
            ("solution.json", solution),
            ("expected.json", exact),
        ]:
            write(path / filename, value)
        records.append(
            dict(name=name, metrics=metrics, files={p.name: sha(p) for p in sorted(path.iterdir())})
        )
        print(json.dumps(dict(name=name, accepted=True)), flush=True)
    if signature() != captured:
        raise ValueError("Graph drivers changed during qualification")
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
    budget = json.loads(BUDGET.read_text())
    if (
        report.get("passed") is not True
        or report["signature"] != captured
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Graph signature or numerical budget mismatch")
    for path, digest in captured["drivers"].items():
        if sha(folder / "drivers" / Path(path).name) != digest:
            raise ValueError("Frozen graph driver mismatch")
    expected = list(cases())
    if [c["name"] for c in report["cases"]] != [c[0] for c in expected]:
        raise ValueError("Incomplete graph qualification matrix")
    for record, (name, spec, exact) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "solution.json", "expected.json"}:
            raise ValueError("Incomplete graph packet")
        for filename, digest in record["files"].items():
            if sha(path / filename) != digest:
                raise ValueError("Graph packet integrity mismatch")
        if (
            json.loads((path / "request.json").read_text()) != spec
            or json.loads((path / "expected.json").read_text()) != exact
        ):
            raise ValueError("Graph request/manufactured fixture mismatch")
        solved = solve_hydraulic_graph(spec)
        if (
            solved != json.loads((path / "solution.json").read_text())
            or assess(spec, solved, exact, budget) != record["metrics"]
        ):
            raise ValueError("Graph numerical audit differs")
    if signature() != captured:
        raise ValueError("Graph drivers changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        maximum_nodes=max(len(s["nodes"]) for _, s, _ in expected),
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
