"""Independent scheduled/feedback IVPs and full command-transaction replay."""

import argparse
import bisect
import json
import shutil
from pathlib import Path

import numpy as np

from core.network import NetworkError
from scripts.controlled_reduced_engine import BUDGET, OnlineReducedGraph, ScheduledReducedGraph
from scripts.controlled_reduced_engine import signature as model_signature
from scripts.nonlinear_graph_references import bridge_spec, pipe_spec
from scripts.nonlinear_thermal_graph import digest
from scripts.scheduled_graph_references import cases as scheduled_cases
from scripts.scheduled_graph_references import event, negative_cases, reference, wrap
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/qualify_controlled_radau.py",
    "scripts/scheduled_graph_references.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 10.0, 30.0, 37.5, 60.0, 90.0, 120.0)
DURATIONS = (10.0, 20.0, 7.5, 22.5, 30.0, 30.0)
MESH_STEPS = (2.0, 1.0, 0.5)
WARM_MESH_STEPS = (0.5, 0.25, 0.125)
MINIMUM_MESH_RATIO = 1.2


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def cases():
    for name, spec in scheduled_cases():
        yield "scheduled_" + name, "scheduled", spec
    warm = bridge_spec()
    warm["steady_input"]["hydraulic"]["nodes"][1]["max_pressure_pa"] = 196500.0
    yield (
        "scheduled_warm_reinitialization",
        "scheduled",
        wrap(warm, [event(60.0, hydraulic_nodes=dict(supply=dict(pressure_pa=201000.0)))]),
    )
    yield "online_thermal_feedback", "online", pipe_spec()
    yield "online_inlet_feedback", "online", bridge_spec()


def policy(name, observation):
    state = observation["model"]["engine"]["state"]
    if name == "online_inlet_feedback":
        return dict(
            hydraulic_nodes=dict(
                a=dict(injection_kg_s=0.25 if observation["model"]["time_s"] < 37.5 else -0.25)
            ),
            thermal_nodes=dict(a=dict(supply_temperature_k=350.0)),
        )
    power = max(-100000.0, min(100000.0, 10000.0 * (330.0 - state["temperatures_k"]["a"])))
    return dict(thermal_nodes=dict(a=dict(heat_power_w=power)))


def errors(states, refs):
    budget = json.loads(BUDGET.read_text())
    result = {}
    for field, limit in [
        ("temperatures_k", "maximum_reference_temperature_error_k"),
        ("pressures_pa", "maximum_reference_pressure_error_pa"),
        ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
    ]:
        value = max(
            abs(state[field][name] - ref["states"][i][field][name])
            for ref in refs
            for i, state in enumerate(states)
            for name in state[field]
        )
        if not np.isfinite(value) or value > budget[limit]:
            raise ValueError("Controlled reference budget exceeded: " + field)
        result[field] = value
    value = max(
        abs(
            state["integrated_external_energy_j"] - ref["states"][i]["integrated_external_energy_j"]
        )
        for ref in refs
        for i, state in enumerate(states)
    )
    conservation = max(abs(state["energy_residual_j"]) for state in states)
    if (
        not np.isfinite(value)
        or value > budget["maximum_reference_energy_error_j"]
        or conservation > budget["maximum_energy_error_j"]
    ):
        raise ValueError("Controlled energy budget exceeded")
    result.update(reference_energy_j=value, energy_residual_j=conservation)
    return result


def measured(reports):
    return max(
        segment["maximum_internal_step_s"]
        for report in reports
        for outer in report["segments"]
        for segment in outer["result"]["segments"]
    )


def verify_switches(spec, reports):
    times = [e["time_s"] for e in spec["timeline"]]
    expected = [(i, t) for i, t in enumerate(times) if 0 < t <= TIMES[-1]]
    traces = [trace for report in reports for trace in report["switches"]]
    if [(t["receipt"]["event_index"], t["receipt"]["time_s"]) for t in traces] != expected:
        raise ValueError("Controlled command skipped or replayed")
    for trace in traces:
        left, right = trace["left"], trace["right"]
        time = trace["receipt"]["time_s"]
        if (
            left["time_s"] != time
            or right["time_s"] != time
            or left["increments"] != right["increments"]
        ):
            raise ValueError("Controlled switch changed stock/time frame")
        for key in (
            "temperatures_k",
            "temperature_increments_k",
            "integrated_external_energy_j",
            "energy_residual_j",
        ):
            if left["state"][key] != right["state"][key]:
                raise ValueError("Controlled switch introduced an energy impulse")
    for report in reports:
        for segment in report["segments"]:
            a, b = segment["start_s"], segment["end_s"]
            if (
                a >= b
                or any(a < t < b for t in times)
                or segment["active_event_count"] != bisect.bisect_right(times, a)
            ):
                raise ValueError("Radau crossed a command discontinuity")
    return len(traces)


def compare(a, b):
    budget = json.loads(BUDGET.read_text())
    temperature = max(
        abs(a["temperatures_k"][n] - b["temperatures_k"][n]) for n in a["temperatures_k"]
    )
    energy = abs(a["integrated_external_energy_j"] - b["integrated_external_energy_j"])
    if (
        temperature > budget["maximum_split_temperature_error_k"]
        or energy > budget["maximum_split_energy_error_j"]
    ):
        raise ValueError("Controlled split path exceeds budget")
    return dict(temperature_k=temperature, energy_j=energy)


def evaluate_scheduled(spec, caps=None):
    caps = MESH_STEPS if caps is None else caps
    refs = [reference(spec, method, TIMES) for method in ("Radau", "DOP853")]
    meshes = []
    for cap in caps:
        model = ScheduledReducedGraph(spec, max_macro_step_s=cap)
        states = [model.snapshot()]
        reports = []
        for a, b in zip(TIMES, TIMES[1:]):
            reports.append(model.advance(b - a))
            states.append(model.snapshot())
        count = verify_switches(spec, reports)
        meshes.append(
            dict(
                max_macro_step_s=cap,
                maximum_measured_internal_step_s=measured(reports),
                states=states,
                reports=reports,
                errors=errors([s["engine"]["state"] for s in states], refs),
                verified_nonzero_switches=count,
            )
        )
    model = ScheduledReducedGraph(spec, max_macro_step_s=caps[-1])
    model.advance(37.5)
    checkpoint = model.checkpoint()
    restored = ScheduledReducedGraph(spec, max_macro_step_s=caps[-1])
    restored.restore(json.loads(json.dumps(checkpoint)))
    a = model.advance(TIMES[-1] - 37.5)
    b = restored.advance(TIMES[-1] - 37.5)
    if a != b:
        raise ValueError("Scheduled Radau restart continuation differs")
    split = compare(a["engine"]["state"], meshes[-1]["states"][-1]["engine"]["state"])
    return dict(
        meshes=meshes,
        references=refs,
        checkpoint=checkpoint,
        restarted=a,
        split_errors=split,
        restart_values_identical=True,
    )


def evaluate_online(name, spec):
    meshes = []
    for cap in MESH_STEPS:
        model = OnlineReducedGraph(spec, max_macro_step_s=cap)
        states = [model.observation()]
        reports = []
        times = [0.0]
        for duration in DURATIONS:
            observation = model.observation()
            report = model.step(
                policy(name, observation), duration, expected_revision=observation["revision"]
            )
            reports.append(report)
            states.append(model.observation())
            times.append(model.observation()["model"]["time_s"])
        history = model.checkpoint()["payload"]["history"]
        refs = [reference(wrap(spec, history), method, times) for method in ("Radau", "DOP853")]
        # Pre-command observations have old p/q. Compare new command's right
        # state at interior sampling clocks against right-continuous IVPs.
        comparable = [report["command_switch"]["right"]["state"] for report in reports] + [
            states[-1]["model"]["engine"]["state"]
        ]
        model_copy = OnlineReducedGraph(spec, max_macro_step_s=cap)
        for duration in DURATIONS[:3]:
            observation = model_copy.observation()
            model_copy.step(
                policy(name, observation), duration, expected_revision=observation["revision"]
            )
        checkpoint = model_copy.checkpoint()
        restored = OnlineReducedGraph(spec, max_macro_step_s=cap)
        restored.restore(json.loads(json.dumps(checkpoint)))
        tail = []
        for duration in DURATIONS[3:]:
            observation = model_copy.observation()
            changes = policy(name, observation)
            a = model_copy.step(changes, duration, expected_revision=observation["revision"])
            b = restored.step(changes, duration, expected_revision=observation["revision"])
            if a != b:
                raise ValueError("Online Radau restart continuation differs")
            tail.append(b)
        if restored.observation() != model.observation():
            raise ValueError("Online Radau restarted episode differs")
        meshes.append(
            dict(
                max_macro_step_s=cap,
                maximum_measured_internal_step_s=measured([r["integration"] for r in reports]),
                states=states,
                reports=reports,
                references=refs,
                history=history,
                errors=errors(comparable, refs),
                checkpoint=checkpoint,
                continuations=tail,
            )
        )
    return dict(meshes=meshes, restart_values_identical=True)


def evaluate(name, mode, spec):
    result = (
        evaluate_scheduled(
            spec, WARM_MESH_STEPS if name == "scheduled_warm_reinitialization" else MESH_STEPS
        )
        if mode == "scheduled"
        else evaluate_online(name, spec)
    )
    actual = [m["maximum_measured_internal_step_s"] for m in result["meshes"]]
    require_mesh_separation(actual)
    return result


def require_mesh_separation(actual):
    if len(actual) != 3 or not all(
        a >= MINIMUM_MESH_RATIO * b > 0 for a, b in zip(actual, actual[1:])
    ):
        raise ValueError("Controlled actual time meshes are not sufficiently distinct")


def rejection(spec):
    model = ScheduledReducedGraph(spec)
    before = model.checkpoint()
    try:
        model.advance(10.0)
    except NetworkError as error:
        if model.checkpoint() != before:
            raise ValueError("Controlled negative scenario partially committed") from error
        return dict(
            error_type=type(error).__name__,
            message=str(error),
            before_checkpoint_sha256=digest(before),
            after_checkpoint_sha256=digest(model.checkpoint()),
            full_request_rolled_back=True,
        )
    raise ValueError("Controlled invalid scenario falsely accepted")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
    for name, mode, spec in cases():
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(name, mode, spec)
        write(path / "result.json", result)
        records.append(
            dict(name=name, mode=mode, files={p.name: sha(p) for p in sorted(path.iterdir())})
        )
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
    for name, spec, _ in negative_cases():
        path = folder / "negative" / name
        path.mkdir(parents=True)
        write(path / "request.json", spec)
        write(path / "result.json", rejection(spec))
        negatives.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(json.dumps(dict(name=name, rejected_as_expected=True)), flush=True)
    if signature() != captured:
        raise ValueError("Controlled sources changed during campaign")
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
        raise ValueError("Controlled source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, h in captured[group].items():
            if sha(folder / group / Path(path).name) != h:
                raise ValueError("Controlled frozen companion mismatch")
    expected = list(cases())
    negative = list(negative_cases())
    if [(r["name"], r["mode"]) for r in report["cases"]] != [(n, m) for n, m, _ in expected] or [
        r["name"] for r in report["negative_cases"]
    ] != [n for n, _, _ in negative]:
        raise ValueError("Incomplete controlled matrix")
    for records, items, prefix in [
        (report["cases"], expected, folder),
        (report["negative_cases"], negative, folder / "negative"),
    ]:
        for record, item in zip(records, items):
            name = item[0]
            spec = item[2] if prefix == folder else item[1]
            path = prefix / name
            if set(record["files"]) != {"request.json", "result.json"}:
                raise ValueError("Incomplete controlled packet")
            for file, h in record["files"].items():
                if sha(path / file) != h:
                    raise ValueError("Controlled packet integrity mismatch")
            if json.loads((path / "request.json").read_text()) != spec:
                raise ValueError("Controlled fixture mismatch")
            fresh = evaluate(*item) if prefix == folder else rejection(spec)
            if fresh != json.loads((path / "result.json").read_text()):
                raise ValueError("Controlled numerical replay differs")
    if signature() != captured:
        raise ValueError("Controlled sources changed during audit")
    scheduled = sum(mode == "scheduled" for _, mode, _ in expected)
    online = len(expected) - scheduled
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        scheduled_cases=scheduled,
        online_cases=online,
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * scheduled + 2 * online * len(MESH_STEPS),
        negative_checks=len(negative),
        horizon_s=TIMES[-1],
        maximum_nodes=max(
            len((s["model_input"] if m == "scheduled" else s)["volumes_m3"]) for _, m, s in expected
        ),
        qualification_sha256=sha(folder / "qualification.json"),
        restart_values_identical=True,
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
