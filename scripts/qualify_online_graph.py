"""Recompute online control episodes against independent segmented IVPs."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from scripts.nonlinear_graph_references import bridge_spec, pipe_spec
from scripts.online_thermal_graph import (
    BUDGET,
    OnlineThermalGraph,
    wrap,
)
from scripts.online_thermal_graph import (
    signature as model_signature,
)
from scripts.scheduled_graph_references import cases as scheduled_cases
from scripts.scheduled_graph_references import reference
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/qualify_online_graph.py",
    "scripts/scheduled_graph_references.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
DURATIONS = (10.0, 20.0, 7.5, 22.5, 30.0, 30.0)
MESH_STEPS = (0.5, 0.25, 0.125)


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def cases():
    yield "thermal_feedback", pipe_spec()
    yield "pressure_reversal", pipe_spec()
    yield "inlet_feedback", bridge_spec()
    yield "eight_node_feedback", dict(scheduled_cases())["endpoint_checkpoint"]["model_input"]


def policy(name, observation):
    """Demonstration bounded proportional controls; not tuned plant regulators."""
    state = observation["model"]["state"]
    time = state["time_s"]
    temperatures = state["temperatures_k"]
    node = next(iter(temperatures))
    if name == "pressure_reversal":
        return dict(hydraulic_nodes=dict(a=dict(pressure_pa=198000.0 if time < 37.5 else 201000.0)))
    if name == "inlet_feedback":
        return dict(
            hydraulic_nodes=dict(a=dict(injection_kg_s=0.25 if time < 37.5 else -0.25)),
            thermal_nodes=dict(a=dict(supply_temperature_k=350.0)),
        )
    power = max(-100000.0, min(100000.0, 10000.0 * (330.0 - temperatures[node])))
    return dict(thermal_nodes={node: dict(heat_power_w=power)})


def evaluate(name, spec):
    budget = json.loads(BUDGET.read_text())
    meshes = []
    for cap in MESH_STEPS:
        request = deepcopy(spec)
        request["max_step_s"] = cap
        model = OnlineThermalGraph(request)
        observations = [model.observation()]
        reports = []
        times = [0.0]
        for duration in DURATIONS:
            before = model.observation()
            changes = policy(name, before)
            report = model.step(changes, duration, expected_revision=before["revision"])
            reports.append(report)
            observations.append(model.observation())
            times.append(model.observation()["model"]["time_s"])
        history = model.checkpoint()["payload"]["history"]
        refs = [reference(wrap(request, history), method, times) for method in ("Radau", "DOP853")]
        errors = {}
        for key, limit in [
            ("temperatures_k", "maximum_reference_temperature_error_k"),
            ("pressures_pa", "maximum_reference_pressure_error_pa"),
            ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
        ]:
            error = max(
                abs(o["model"]["state"][key][k] - r["states"][i][key][k])
                for r in refs
                for i, o in enumerate(observations)
                for k in o["model"]["state"][key]
            )
            # Sample observations precede a new command at the same clock. The
            # independent trajectory is right-continuous, so compare p/q against
            # the next command's right state, while temperature stocks agree.
            if key != "temperatures_k":
                error = max(
                    abs(s[key][k] - r["states"][i][key][k])
                    for r in refs
                    for i, s in enumerate(
                        [p["command_switch"]["right"] for p in reports]
                        + [observations[-1]["model"]["state"]]
                    )
                    for k in s[key]
                )
            errors[key] = error
            if error > budget[limit]:
                raise ValueError("Online reference budget exceeded: " + key)
        energy_error = max(
            abs(
                o["model"]["state"]["integrated_external_energy_j"]
                - r["states"][i]["integrated_external_energy_j"]
            )
            for r in refs
            for i, o in enumerate(observations)
        )
        if energy_error > budget["maximum_reference_energy_error_j"]:
            raise ValueError("Online reference energy budget exceeded")
        errors["energy_j"] = energy_error
        steps = [
            s["result"]["maximum_accepted_step_s"]
            for report in reports
            for s in report["integration"]["segments"]
        ]
        # Restart midway with original input, then recompute future policy actions.
        original = OnlineThermalGraph(request)
        for duration in DURATIONS[:3]:
            o = original.observation()
            original.step(policy(name, o), duration, expected_revision=o["revision"])
        checkpoint = original.checkpoint()
        restored = OnlineThermalGraph(request)
        restored.restore(json.loads(json.dumps(checkpoint)))
        continuations = []
        for duration in DURATIONS[3:]:
            o = original.observation()
            command = policy(name, o)
            a = original.step(command, duration, expected_revision=o["revision"])
            b = restored.step(command, duration, expected_revision=o["revision"])
            if a != b:
                raise ValueError("Online adaptive restart differs")
            continuations.append(b)
        if restored.observation() != model.observation():
            raise ValueError("Online restarted episode differs")
        meshes.append(
            dict(
                max_step_s=cap,
                maximum_measured_step_s=max(steps),
                observations=observations,
                reports=reports,
                history=history,
                references=refs,
                errors=errors,
                checkpoint=checkpoint,
                continuations=continuations,
            )
        )
    if not all(
        a["maximum_measured_step_s"] > b["maximum_measured_step_s"]
        for a, b in zip(meshes, meshes[1:])
    ):
        raise ValueError("Online meshes are not distinct")
    return dict(meshes=meshes, adaptive_restart_values_identical=True)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for p in captured[group]:
            shutil.copyfile(p, folder / group / Path(p).name)
    records = []
    for name, spec in cases():
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(name, spec)
        write(path / "result.json", result)
        records.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(json.dumps(dict(name=name, passed=True)), flush=True)
    if signature() != captured:
        raise ValueError("Online sources changed during campaign")
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
        raise ValueError("Online source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for p, h in captured[group].items():
            if sha(folder / group / Path(p).name) != h:
                raise ValueError("Online frozen companion mismatch")
    expected = list(cases())
    if [r["name"] for r in report["cases"]] != [name for name, _ in expected]:
        raise ValueError("Incomplete online matrix")
    for record, (name, spec) in zip(report["cases"], expected):
        path = folder / name
        if set(record["files"]) != {"request.json", "result.json"}:
            raise ValueError("Incomplete online packet")
        for p, h in record["files"].items():
            if sha(path / p) != h:
                raise ValueError("Online packet integrity mismatch")
        if json.loads((path / "request.json").read_text()) != spec:
            raise ValueError("Online fixture mismatch")
        if evaluate(name, spec) != json.loads((path / "result.json").read_text()):
            raise ValueError("Online numerical replay differs")
    if signature() != captured:
        raise ValueError("Online sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * len(expected) * len(MESH_STEPS),
        horizon_s=sum(DURATIONS),
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
