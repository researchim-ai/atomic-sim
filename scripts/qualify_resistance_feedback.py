"""Frozen evidence for stationary temperature/resistance feedback."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.resistance_feedback_references import cases, reference
from scripts.solve_hydraulic_graph import sha
from scripts.solve_resistance_feedback import BUDGET, solve_resistance_feedback
from scripts.solve_resistance_feedback import signature as model_signature

EXTRA = (
    "scripts/resistance_feedback_references.py",
    "scripts/qualify_resistance_feedback.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/thermal_graph_references.py",
)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(spec, exact):
    solved = solve_resistance_feedback(spec)
    point = solved["solution"]
    oracle = reference(spec)
    budget = json.loads(BUDGET.read_text())
    metrics = {}
    for label, expected in [("reference", oracle), ("manufactured", exact)]:
        for key, tolerance in [
            ("temperatures_k", "maximum_reference_temperature_error_k"),
            ("pressures_pa", "maximum_reference_pressure_error_pa"),
            ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
        ]:
            actual = point[key] if key == "temperatures_k" else point["hydraulic"][key]
            error = max(abs(actual[k] - v) for k, v in expected[key].items())
            metrics[label + "_" + key] = error
            if not np.isfinite(error) or error > budget[tolerance]:
                raise ValueError("Feedback disagrees with " + label + " " + key)
    return dict(solution=solved, reference=oracle, metrics=metrics)


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for path in captured[group]:
            shutil.copyfile(path, folder / group / Path(path).name)
    records = []
    for name, spec, exact in cases():
        result = evaluate(spec, exact)
        path = folder / name
        path.mkdir()
        for filename, value in [
            ("request.json", spec),
            ("expected.json", exact),
            ("result.json", result),
        ]:
            write(path / filename, value)
        records.append(
            dict(
                name=name,
                metrics=result["metrics"],
                files={p.name: sha(p) for p in sorted(path.iterdir())},
            )
        )
        print(json.dumps(dict(name=name, accepted=True)), flush=True)
    if signature() != captured:
        raise ValueError("Feedback signature changed during campaign")
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
        raise ValueError("Feedback signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, digest in captured[group].items():
            if sha(folder / group / Path(path).name) != digest:
                raise ValueError("Feedback frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [r[0] for r in expected]:
        raise ValueError("Incomplete feedback matrix")
    for record, (name, spec, exact) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "expected.json", "result.json"}:
            raise ValueError("Incomplete feedback packet")
        for filename, digest in record["files"].items():
            if sha(path / filename) != digest:
                raise ValueError("Feedback packet integrity mismatch")
        if (
            json.loads((path / "request.json").read_text()) != spec
            or json.loads((path / "expected.json").read_text()) != exact
        ):
            raise ValueError("Feedback fixture mismatch")
        fresh = evaluate(spec, exact)
        if (
            fresh != json.loads((path / "result.json").read_text())
            or fresh["metrics"] != record["metrics"]
        ):
            raise ValueError("Feedback numerical replay differs")
    if signature() != captured:
        raise ValueError("Feedback sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        maximum_nodes=max(len(s["initial_temperatures_k"]) for _, s, _ in expected),
        qualification_sha256=sha(folder / "qualification.json"),
        global_uniqueness_proven=False,
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
