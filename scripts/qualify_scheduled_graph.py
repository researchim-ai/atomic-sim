"""Source-bound schedule validation: time cuts, dual IVPs, meshes and rollback."""

import argparse
import bisect
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.nonlinear_thermal_graph import digest
from scripts.scheduled_graph_references import cases, negative_cases, reference
from scripts.scheduled_thermal_graph import BUDGET, ScheduledGraphError, ScheduledThermalGraph
from scripts.scheduled_thermal_graph import signature as model_signature
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/scheduled_graph_references.py",
    "scripts/qualify_scheduled_graph.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 10.0, 30.0, 37.5, 60.0, 90.0, 120.0)
MESH_STEPS = (0.5, 0.25, 0.125)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_cuts(spec, reports):
    times = [e["time_s"] for e in spec["timeline"]]
    traces = [s for report in reports for s in report["switches"]]
    expected = [(i, t) for i, t in enumerate(times) if 0 < t <= TIMES[-1]]
    if [(s["receipt"]["event_index"], s["receipt"]["time_s"]) for s in traces] != expected:
        raise ValueError("Scheduled event was skipped, duplicated or reordered")
    continuity = (
        "temperatures_k",
        "temperature_increments_k",
        "energy_integral_scaled_k",
        "integrated_external_energy_j",
        "stored_energy_j",
        "energy_residual_j",
    )
    for trace in traces:
        time = trace["receipt"]["time_s"]
        if (
            trace["left"]["time_s"] != time
            or trace["right"]["time_s"] != time
            or any(trace["left"][k] != trace["right"][k] for k in continuity)
        ):
            raise ValueError("Scheduled switch introduced a stock/energy/time impulse")
    for report in reports:
        for segment in report["segments"]:
            a, b = segment["start_s"], segment["end_s"]
            if (
                a >= b
                or any(a < t < b for t in times)
                or segment["active_event_count"] != bisect.bisect_right(times, a)
            ):
                raise ValueError("Integrator crossed a command discontinuity")
    return len(traces)


def evaluate(spec):
    budget = json.loads(BUDGET.read_text())
    refs = [reference(spec, method, TIMES) for method in ("Radau", "DOP853")]
    meshes = []
    for cap in MESH_STEPS:
        request = json.loads(json.dumps(spec))
        request["model_input"]["max_step_s"] = cap
        model = ScheduledThermalGraph(request)
        states = [model.snapshot()]
        reports = []
        for a, b in zip(TIMES, TIMES[1:]):
            reports.append(model.advance(b - a))
            states.append(model.snapshot())
        switches = verify_cuts(request, reports)
        expected_count = bisect.bisect_right([e["time_s"] for e in request["timeline"]], TIMES[-1])
        if states[-1]["active_event_count"] != expected_count:
            raise ValueError("Scheduled final command cursor mismatch")
        metrics = {}
        for key, limit in [
            ("temperatures_k", "maximum_reference_temperature_error_k"),
            ("pressures_pa", "maximum_reference_pressure_error_pa"),
            ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
        ]:
            error = max(
                abs(s["state"][key][name] - r["states"][i][key][name])
                for r in refs
                for i, s in enumerate(states)
                for name in s["state"][key]
            )
            metrics["reference_" + key] = error
            if not np.isfinite(error) or error > budget[limit]:
                raise ValueError("Scheduled reference " + key + " budget exceeded: " + str(error))
        energy_error = max(
            abs(
                s["state"]["integrated_external_energy_j"]
                - r["states"][i]["integrated_external_energy_j"]
            )
            for r in refs
            for i, s in enumerate(states)
        )
        conservation = max(abs(s["state"]["energy_residual_j"]) for s in states)
        if (
            energy_error > budget["maximum_reference_energy_error_j"]
            or conservation > budget["maximum_energy_error_j"]
        ):
            raise ValueError("Scheduled energy budget exceeded")
        metrics.update(reference_energy_error_j=energy_error, maximum_energy_error_j=conservation)
        segments = [segment["result"] for report in reports for segment in report["segments"]]
        meshes.append(
            dict(
                max_step_s=cap,
                maximum_measured_step_s=max(s["maximum_accepted_step_s"] for s in segments),
                minimum_measured_step_s=min(s["minimum_accepted_step_s"] for s in segments),
                states=states,
                reports=reports,
                metrics=metrics,
                verified_nonzero_event_switches=switches,
            )
        )
    actual = [m["maximum_measured_step_s"] for m in meshes]
    if not all(a > b for a, b in zip(actual, actual[1:])):
        raise ValueError("Three distinct measured schedule time meshes were not obtained")
    fine = meshes[-1]
    mesh_error = max(
        abs(s["state"]["temperatures_k"][k] - fine["states"][i]["state"]["temperatures_k"][k])
        for mesh in meshes[:-1]
        for i, s in enumerate(mesh["states"])
        for k in s["state"]["temperatures_k"]
    )
    if mesh_error > budget["maximum_mesh_temperature_difference_k"]:
        raise ValueError("Scheduled mesh temperature budget exceeded")
    request = json.loads(json.dumps(spec))
    request["model_input"]["max_step_s"] = MESH_STEPS[-1]
    memory = ScheduledThermalGraph(request)
    memory.advance(37.5)
    checkpoint = memory.checkpoint()
    restored = ScheduledThermalGraph(request)
    restored.restore(json.loads(json.dumps(checkpoint)))
    a = memory.advance(TIMES[-1] - 37.5)
    b = restored.advance(TIMES[-1] - 37.5)
    if a != b:
        raise ValueError("Scheduled adaptive restart differs from in-memory continuation")
    final = fine["states"][-1]
    split_error = max(
        abs(a["state"]["temperatures_k"][k] - final["state"]["temperatures_k"][k])
        for k in a["state"]["temperatures_k"]
    )
    split_energy = abs(
        a["state"]["integrated_external_energy_j"] - final["state"]["integrated_external_energy_j"]
    )
    if (
        split_error > budget["maximum_split_temperature_difference_k"]
        or split_energy > budget["maximum_split_energy_difference_j"]
    ):
        raise ValueError("Scheduled split path exceeds numerical budget")
    return dict(
        meshes=meshes,
        references=refs,
        checkpoint=checkpoint,
        restarted=a,
        metrics=dict(
            maximum_mesh_temperature_difference_k=mesh_error,
            maximum_split_temperature_difference_k=split_error,
            maximum_split_energy_difference_j=split_energy,
            adaptive_restart_values_identical=True,
        ),
    )


def rejection(spec, criterion):
    model = ScheduledThermalGraph(spec)
    before = model.checkpoint()
    try:
        model.advance(10)
    except ScheduledGraphError as error:
        if error.report["criterion"] != criterion or model.checkpoint() != before:
            raise ValueError(
                "Scheduled negative case has wrong diagnostic or partial state"
            ) from error
        return dict(
            diagnostics=error.report,
            before_checkpoint_sha256=digest(before),
            after_checkpoint_sha256=digest(model.checkpoint()),
            full_request_rolled_back=True,
        )
    raise ValueError("Scheduled negative case was falsely accepted")


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for path in captured[group]:
            shutil.copyfile(path, folder / group / Path(path).name)
    records = []
    negatives = []
    for name, spec in cases():
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(spec)
        write(path / "result.json", result)
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
    for name, spec, criterion in negative_cases():
        path = folder / "negative" / name
        path.mkdir(parents=True)
        write(path / "request.json", spec)
        result = rejection(spec, criterion)
        write(path / "result.json", result)
        negatives.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(json.dumps(dict(name=name, rejected_as_expected=True)), flush=True)
    if signature() != captured:
        raise ValueError("Scheduled sources changed during campaign")
    report = dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=records,
        negative_cases=negatives,
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
        raise ValueError("Scheduled source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, digest_value in captured[group].items():
            if sha(folder / group / Path(path).name) != digest_value:
                raise ValueError("Scheduled frozen companion mismatch")
    expected = list(cases())
    negative = list(negative_cases())
    if [r["name"] for r in report["cases"]] != [c[0] for c in expected] or [
        r["name"] for r in report["negative_cases"]
    ] != [c[0] for c in negative]:
        raise ValueError("Incomplete scheduled matrix")
    for records, items, prefix in [
        (report["cases"], expected, folder),
        (report["negative_cases"], negative, folder / "negative"),
    ]:
        for record, item in zip(records, items):
            name, spec = item[:2]
            path = prefix / name
            if set(record["files"]) != {"request.json", "result.json"}:
                raise ValueError("Incomplete scheduled packet")
            for filename, digest_value in record["files"].items():
                if sha(path / filename) != digest_value:
                    raise ValueError("Scheduled packet integrity mismatch")
            if json.loads((path / "request.json").read_text()) != spec:
                raise ValueError("Scheduled fixture mismatch")
            fresh = evaluate(spec) if len(item) == 2 else rejection(spec, item[2])
            if fresh != json.loads((path / "result.json").read_text()) or (
                len(item) == 2 and fresh["metrics"] != record["metrics"]
            ):
                raise ValueError("Scheduled numerical replay differs")
    if signature() != captured:
        raise ValueError("Scheduled sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * len(expected),
        negative_checks=len(negative),
        horizon_s=TIMES[-1],
        maximum_nodes=max(len(s["model_input"]["volumes_m3"]) for _, s in expected),
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
