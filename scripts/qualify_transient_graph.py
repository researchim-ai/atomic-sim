"""Source-bound transient graph campaign, independent dual IVPs and restart."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.solve_hydraulic_graph import sha
from scripts.transient_graph_references import cases, reference
from scripts.transient_thermal_graph import BUDGET, ThermalGraph
from scripts.transient_thermal_graph import signature as model_signature

EXTRA = (
    "scripts/transient_graph_references.py",
    "scripts/qualify_transient_graph.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/thermal_graph_references.py",
)


def signature():
    result = model_signature()
    result["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return result


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(spec):
    budget = json.loads(BUDGET.read_text())
    model = ThermalGraph(spec)
    states = [model.snapshot()]
    for duration in (30, 30, 60):
        states.append(model.advance(duration))
    refs = [reference(spec, m) for m in ("Radau", "DOP853")]
    temperatures = np.array([[s["temperatures_k"][n] for n in model.names] for s in states])
    energies = np.array([s["integrated_external_energy_j"] for s in states])
    metrics = dict(
        reference_temperature_error_k=max(
            float(np.max(np.abs(temperatures - np.array(r["temperatures_k"])))) for r in refs
        ),
        reference_energy_error_j=max(
            float(np.max(np.abs(energies - np.array(r["integrated_external_energy_j"]))))
            for r in refs
        ),
        reference_method_energy_difference_j=float(
            np.max(
                np.abs(
                    np.array(refs[0]["integrated_external_energy_j"])
                    - np.array(refs[1]["integrated_external_energy_j"])
                )
            )
        ),
        maximum_energy_residual_j=max(abs(s["energy_residual_j"]) for s in states),
    )
    split = ThermalGraph(spec)
    split.advance(37.5)
    checkpoint = split.checkpoint()
    restored = ThermalGraph(spec)
    restored.restore(json.loads(json.dumps(checkpoint)))
    last = restored.advance(82.5)
    metrics["restart_temperature_error_k"] = max(
        abs(last["temperatures_k"][k] - states[-1]["temperatures_k"][k]) for k in model.names
    )
    metrics["restart_energy_error_j"] = abs(last["integrated_external_energy_j"] - energies[-1])
    criteria = dict(
        reference_temperature_error_k="maximum_reference_temperature_error_k",
        reference_energy_error_j="maximum_reference_energy_error_j",
        reference_method_energy_difference_j="maximum_reference_method_energy_difference_j",
        maximum_energy_residual_j="maximum_energy_error_j",
        restart_temperature_error_k="maximum_restart_temperature_error_k",
        restart_energy_error_j="maximum_energy_error_j",
    )
    if any(not np.isfinite(v) or v > budget[criteria[k]] for k, v in metrics.items()):
        raise ValueError("Dynamic graph qualification exceeds budget: " + str(metrics))
    return dict(
        states=states, references=refs, checkpoint=checkpoint, restarted=last, metrics=metrics
    )


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for path in captured[group]:
            shutil.copyfile(path, folder / group / Path(path).name)
    records = []
    for name, spec in cases():
        result = evaluate(spec)
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        write(path / "result.json", result)
        records.append(
            dict(
                name=name,
                metrics=result["metrics"],
                files={p.name: sha(p) for p in sorted(path.iterdir())},
            )
        )
        print(json.dumps(dict(name=name, accepted=True)), flush=True)
    if signature() != captured:
        raise ValueError("Dynamic signature changed during campaign")
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
        raise ValueError("Dynamic source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, digest in captured[group].items():
            if sha(folder / group / Path(path).name) != digest:
                raise ValueError("Dynamic frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [c[0] for c in expected]:
        raise ValueError("Incomplete dynamic matrix")
    for record, (name, spec) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "result.json"}:
            raise ValueError("Incomplete dynamic packet")
        for filename, digest in record["files"].items():
            if sha(path / filename) != digest:
                raise ValueError("Dynamic packet integrity mismatch")
        if json.loads((path / "request.json").read_text()) != spec:
            raise ValueError("Dynamic fixture mismatch")
        fresh = evaluate(spec)
        if (
            fresh != json.loads((path / "result.json").read_text())
            or fresh["metrics"] != record["metrics"]
        ):
            raise ValueError("Dynamic numerical replay differs")
    if signature() != captured:
        raise ValueError("Dynamic signature changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        independent_ivps=2 * len(expected),
        horizon_s=120,
        maximum_nodes=max(len(s["volumes_m3"]) for _, s in expected),
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
