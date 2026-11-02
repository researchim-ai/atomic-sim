"""Three measured time meshes, dual independent IVPs and exact adaptive restart."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.nonlinear_graph_references import cases, reference
from scripts.nonlinear_thermal_graph import BUDGET, NonlinearThermalGraph
from scripts.nonlinear_thermal_graph import signature as model_signature
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/nonlinear_graph_references.py",
    "scripts/qualify_nonlinear_graph.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 30.0, 60.0, 90.0, 120.0)
MESH_STEPS = (0.5, 0.25, 0.125)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(spec):
    budget = json.loads(BUDGET.read_text())
    references = [reference(spec, m, TIMES) for m in ("Radau", "DOP853")]
    meshes = []
    for cap in MESH_STEPS:
        request = json.loads(json.dumps(spec))
        request["max_step_s"] = cap
        model = NonlinearThermalGraph(request)
        states = [model.snapshot()]
        for a, b in zip(TIMES, TIMES[1:]):
            states.append(model.advance(b - a))
        temperatures = np.array([[s["temperatures_k"][k] for k in model.names] for s in states])
        energies = np.array([s["integrated_external_energy_j"] for s in states])
        flows = np.array([list(s["flows_kg_s"].values()) for s in states])
        metrics = dict(
            reference_temperature_error_k=max(
                float(np.max(np.abs(temperatures - np.asarray(r["temperatures_k"]))))
                for r in references
            ),
            reference_energy_error_j=max(
                float(np.max(np.abs(energies - np.asarray(r["integrated_external_energy_j"]))))
                for r in references
            ),
            reference_flow_error_kg_s=max(
                float(np.max(np.abs(flows - np.array([list(v.values()) for v in r["flows_kg_s"]]))))
                for r in references
            ),
            maximum_energy_error_j=max(abs(s["energy_residual_j"]) for s in states),
        )
        limits = dict(
            reference_temperature_error_k="maximum_reference_temperature_error_k",
            reference_energy_error_j="maximum_reference_energy_error_j",
            reference_flow_error_kg_s="maximum_reference_flow_error_kg_s",
            maximum_energy_error_j="maximum_energy_error_j",
        )
        if any(not np.isfinite(v) or v > budget[limits[k]] for k, v in metrics.items()):
            raise ValueError("Nonlinear reference/energy budget exceeded: " + str(metrics))
        maximum = max(s["maximum_accepted_step_s"] for s in states[1:])
        minimum = min(s["minimum_accepted_step_s"] for s in states[1:])
        meshes.append(
            dict(
                max_step_s=cap,
                maximum_measured_step_s=maximum,
                minimum_measured_step_s=minimum,
                states=states,
                metrics=metrics,
            )
        )
    measured = [m["maximum_measured_step_s"] for m in meshes]
    if not all(a > b for a, b in zip(measured, measured[1:])):
        raise ValueError("Three distinct measured time meshes were not obtained: " + str(measured))
    fine = meshes[-1]
    fine_temperature = np.array([list(s["temperatures_k"].values()) for s in fine["states"]])
    mesh_error = max(
        float(
            np.max(
                np.abs(
                    np.array([list(s["temperatures_k"].values()) for s in m["states"]])
                    - fine_temperature
                )
            )
        )
        for m in meshes[:-1]
    )
    if mesh_error > budget["maximum_mesh_temperature_difference_k"]:
        raise ValueError("Nonlinear time mesh budget exceeded")
    request = json.loads(json.dumps(spec))
    request["max_step_s"] = MESH_STEPS[-1]
    memory = NonlinearThermalGraph(request)
    memory.advance(TIMES[-1] * 0.3125)
    checkpoint = memory.checkpoint()
    restored = NonlinearThermalGraph(request)
    restored.restore(json.loads(json.dumps(checkpoint)))
    tail = TIMES[-1] * (1 - 0.3125)
    a = memory.advance(tail)
    b = restored.advance(tail)
    if a != b:
        raise ValueError("Adaptive checkpoint continuation differs from in-memory continuation")
    final = fine["states"][-1]
    split_error = max(
        abs(a["temperatures_k"][k] - final["temperatures_k"][k]) for k in final["temperatures_k"]
    )
    energy_error = abs(a["integrated_external_energy_j"] - final["integrated_external_energy_j"])
    if (
        split_error > budget["maximum_split_temperature_difference_k"]
        or energy_error > budget["maximum_split_energy_difference_j"]
    ):
        raise ValueError("Nonlinear split path exceeds numerical budget")
    return dict(
        meshes=meshes,
        references=references,
        checkpoint=checkpoint,
        restarted=a,
        metrics=dict(
            maximum_mesh_temperature_difference_k=mesh_error,
            maximum_split_temperature_difference_k=split_error,
            maximum_split_energy_difference_j=energy_error,
            adaptive_restart_identical=True,
        ),
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
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(spec)
        write(path / "result.json", result)
        if (
            name == "bridge_reversal"
            and not result["meshes"][-1]["states"][0]["flows_kg_s"]["bridge"]
            * result["meshes"][-1]["states"][-1]["flows_kg_s"]["bridge"]
            < 0
        ):
            raise ValueError("Bridge reversal was not observed")
        records.append(
            dict(
                name=name,
                metrics=result["metrics"],
                files={p.name: sha(p) for p in sorted(path.iterdir())},
            )
        )
        print(
            json.dumps(
                dict(
                    name=name,
                    accepted=True,
                    measured_steps_s=[m["maximum_measured_step_s"] for m in result["meshes"]],
                )
            ),
            flush=True,
        )
    if signature() != captured:
        raise ValueError("Nonlinear sources changed during qualification")
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
        raise ValueError("Nonlinear source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, digest in captured[group].items():
            if sha(folder / group / Path(path).name) != digest:
                raise ValueError("Nonlinear frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [c[0] for c in expected]:
        raise ValueError("Incomplete nonlinear matrix")
    for record, (name, spec) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "result.json"}:
            raise ValueError("Incomplete nonlinear packet")
        for filename, digest in record["files"].items():
            if sha(path / filename) != digest:
                raise ValueError("Nonlinear packet integrity mismatch")
        if json.loads((path / "request.json").read_text()) != spec:
            raise ValueError("Nonlinear fixture mismatch")
        result = evaluate(spec)
        if (
            result != json.loads((path / "result.json").read_text())
            or result["metrics"] != record["metrics"]
        ):
            raise ValueError("Nonlinear numerical replay differs")
    if signature() != captured:
        raise ValueError("Nonlinear sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * len(expected),
        horizon_s=TIMES[-1],
        maximum_nodes=max(len(s["volumes_m3"]) for _, s in expected),
        qualification_sha256=sha(folder / "qualification.json"),
        adaptive_restart_identical=True,
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
