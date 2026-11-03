"""Independent transient verification and bidirectional residual-engine handoff."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.nonlinear_graph_references import cases as legacy_cases
from scripts.nonlinear_graph_references import reference
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph
from scripts.reduced_residual_engine import ReducedResidualEngine
from scripts.registered_graph_system import BUDGET, graph_request
from scripts.registered_graph_system import signature as model_signature
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/qualify_reduced_engine.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 30.0, 37.5, 60.0, 120.0)
MESH_STEPS = (2.0, 1.0, 0.5)


def cases():
    selected = {
        "pipe_positive",
        "pipe_reverse",
        "idle_heating",
        "cycle_cooling",
        "8_mixed_negative",
        "bridge_reversal",
    }
    return ((name, spec) for name, spec in legacy_cases() if name in selected)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(a, b, budget):
    error = max(
        abs(a["temperatures_k"][name] - b["temperatures_k"][name]) for name in a["temperatures_k"]
    )
    energy = abs(a["integrated_external_energy_j"] - b["integrated_external_energy_j"])
    if (
        error > budget["maximum_split_temperature_error_k"]
        or energy > budget["maximum_split_energy_error_j"]
    ):
        raise ValueError("Reduced split/handoff budget exceeded")
    return dict(temperature_k=error, energy_j=energy)


def evaluate(spec):
    budget = json.loads(BUDGET.read_text())
    references = [reference(spec, method, TIMES) for method in ("Radau", "DOP853")]
    meshes = []
    for cap in MESH_STEPS:
        model = ReducedResidualEngine(graph_request(spec), max_macro_step_s=cap)
        states = [model.snapshot()]
        reports = []
        for a, b in zip(TIMES, TIMES[1:]):
            reports.append(model.advance(b - a))
            states.append(model.snapshot())
        errors = {}
        errors["temperature_k"] = max(
            abs(s["state"]["temperatures_k"][name] - r["temperatures_k"][i][j])
            for r in references
            for i, s in enumerate(states)
            for j, name in enumerate(s["state"]["temperatures_k"])
        )
        errors["flow_kg_s"] = max(
            abs(s["state"]["flows_kg_s"][name] - r["flows_kg_s"][i][name])
            for r in references
            for i, s in enumerate(states)
            for name in s["state"]["flows_kg_s"]
        )
        errors["energy_j"] = max(
            abs(s["state"]["integrated_external_energy_j"] - r["integrated_external_energy_j"][i])
            for r in references
            for i, s in enumerate(states)
        )
        errors["energy_residual_j"] = max(abs(s["state"]["energy_residual_j"]) for s in states)
        for key, limit in [
            ("temperature_k", "maximum_reference_temperature_error_k"),
            ("flow_kg_s", "maximum_reference_flow_error_kg_s"),
            ("energy_j", "maximum_reference_energy_error_j"),
            ("energy_residual_j", "maximum_energy_error_j"),
        ]:
            if not np.isfinite(errors[key]) or errors[key] > budget[limit]:
                raise ValueError("Reduced reference budget exceeded: " + key)
        measured = max(
            segment["maximum_internal_step_s"]
            for report in reports
            for segment in report["segments"]
        )
        meshes.append(
            dict(
                max_macro_step_s=cap,
                maximum_measured_internal_step_s=measured,
                states=states,
                reports=reports,
                errors=errors,
            )
        )
    if not all(
        a["maximum_measured_internal_step_s"] > b["maximum_measured_internal_step_s"]
        for a, b in zip(meshes, meshes[1:])
    ):
        raise ValueError("Reduced measured meshes are not distinct")
    cap = MESH_STEPS[-1]
    memory = ReducedResidualEngine(graph_request(spec), max_macro_step_s=cap)
    memory.advance(37.5)
    checkpoint = memory.checkpoint()
    restored = ReducedResidualEngine(graph_request(spec), max_macro_step_s=cap)
    restored.restore(json.loads(json.dumps(checkpoint)))
    a = memory.advance(TIMES[-1] - 37.5)
    b = restored.advance(TIMES[-1] - 37.5)
    if a != b:
        raise ValueError("Reduced checkpoint continuation differs")
    split = compare(a["state"], meshes[-1]["states"][-1]["state"], budget)
    old = NonlinearThermalGraph(spec)
    old.advance(37.5)
    incoming = ReducedResidualEngine(graph_request(spec), max_macro_step_s=cap)
    incoming.import_legacy_checkpoint(old.checkpoint())
    incoming.advance(TIMES[-1] - 37.5)
    imported = compare(incoming.snapshot()["state"], meshes[-1]["states"][-1]["state"], budget)
    outgoing = ReducedResidualEngine(graph_request(spec), max_macro_step_s=cap)
    outgoing.advance(37.5)
    handoff = outgoing.export_legacy_checkpoint()
    receiver = NonlinearThermalGraph(spec)
    receiver.restore(handoff["checkpoint"])
    receiver.advance(TIMES[-1] - 37.5)
    exported = compare(receiver.snapshot(), meshes[-1]["states"][-1]["state"], budget)
    return dict(
        meshes=meshes,
        references=references,
        checkpoint=checkpoint,
        restarted=a,
        legacy_incoming_final=incoming.snapshot(),
        legacy_outgoing_final=receiver.snapshot(),
        handoff=handoff,
        metrics=dict(
            adaptive_restart_values_identical=True,
            split=split,
            imported=imported,
            exported=exported,
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
        records.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(
            json.dumps(
                dict(
                    name=name,
                    passed=True,
                    measured_steps_s=[
                        m["maximum_measured_internal_step_s"] for m in result["meshes"]
                    ],
                )
            ),
            flush=True,
        )
    if signature() != captured:
        raise ValueError("Reduced sources changed during campaign")
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
        raise ValueError("Reduced source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, h in captured[group].items():
            if sha(folder / group / Path(path).name) != h:
                raise ValueError("Reduced frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [name for name, _ in expected]:
        raise ValueError("Incomplete reduced matrix")
    for record, (name, spec) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "result.json"}:
            raise ValueError("Incomplete reduced packet")
        for p, h in record["files"].items():
            if sha(path / p) != h:
                raise ValueError("Reduced packet integrity mismatch")
        if json.loads((path / "request.json").read_text()) != spec:
            raise ValueError("Reduced fixture mismatch")
        if evaluate(spec) != json.loads((path / "result.json").read_text()):
            raise ValueError("Reduced numerical replay differs")
    if signature() != captured:
        raise ValueError("Reduced sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * len(expected),
        bidirectional_handoffs=2 * len(expected),
        horizon_s=TIMES[-1],
        maximum_nodes=max(len(s["volumes_m3"]) for _, s in expected),
        qualification_sha256=sha(folder / "qualification.json"),
        adaptive_restart_values_identical=True,
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
