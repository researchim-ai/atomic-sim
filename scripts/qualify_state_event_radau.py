"""Source-bound temperature event qualification against independent event IVPs."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from core.network import NetworkError
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import sha
from scripts.state_event_radau import BUDGET, StateEventGraph
from scripts.state_event_radau import signature as model_signature
from scripts.state_event_references import cases, negative_cases, reference

EXTRA = (
    "scripts/qualify_state_event_radau.py",
    "scripts/state_event_references.py",
    "scripts/scheduled_graph_references.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 3.0, 7.5, 30.0, 60.0, 120.0)
MESH_STEPS = (2.0, 1.0, 0.5)
CASE_MESH_STEPS = {"no_crossing": (1.0, 0.5, 0.25)}
MINIMUM_MESH_RATIO = 1.2


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_states(states, references):
    budget = json.loads(BUDGET.read_text())
    errors = {}
    for key, limit in [
        ("temperatures_k", "maximum_reference_temperature_error_k"),
        ("pressures_pa", "maximum_reference_pressure_error_pa"),
        ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
    ]:
        value = max(
            abs(s[key][n] - r["states"][i][key][n])
            for r in references
            for i, s in enumerate(states)
            for n in s[key]
        )
        errors[key] = value
        if not np.isfinite(value) or value > budget[limit]:
            raise ValueError("State-event reference budget exceeded: " + key)
    energy = max(
        abs(s["integrated_external_energy_j"] - r["states"][i]["integrated_external_energy_j"])
        for r in references
        for i, s in enumerate(states)
    )
    balance = max(abs(s["energy_residual_j"]) for s in states)
    if (
        not np.isfinite(energy)
        or energy > budget["maximum_reference_energy_error_j"]
        or balance > budget["maximum_energy_error_j"]
    ):
        raise ValueError("State-event energy budget exceeded")
    errors.update(reference_energy_j=energy, energy_residual_j=balance)
    return errors


def verify_roots(model, reports, references):
    budget = json.loads(BUDGET.read_text())
    fired = model.snapshot()["fired"]
    if len({r["guard_id"] for r in fired}) != len(fired):
        raise ValueError("State guard fired more than once")
    errors = []
    for reference_result in references:
        if [r["guard_id"] for r in fired] != [r["guard_id"] for r in reference_result["fired"]]:
            raise ValueError("Independent state-event sequence differs")
        errors.extend(
            abs(a["time_s"] - b["time_s"]) for a, b in zip(fired, reference_result["fired"])
        )
    error = max(errors, default=0.0)
    if error > budget["maximum_reference_event_time_error_s"]:
        raise ValueError("State-event root-time budget exceeded")
    for report in reports:
        for trace in report["switches"]:
            root = trace["root"]
            a, b = root["bracket_s"]
            gl, gr = root["bracket_values_k"]
            direction = root["direction"]
            left, right = trace["left"], trace["right"]
            if (
                a > b
                or b != root["time_s"]
                or b - a > budget["root_time_atol_s"] + 2 * np.spacing(b)
                or not direction * gl < 0 <= direction * gr
                or abs(root["root_value_k"]) > budget["root_value_atol_k"]
            ):
                raise ValueError("State-event root bracket certificate invalid")
            if (
                left["time_s"] != root["time_s"]
                or right["time_s"] != root["time_s"]
                or left["increments"] != right["increments"]
            ):
                raise ValueError("State-event switch changed time/stock frame")
            for key in (
                "temperatures_k",
                "temperature_increments_k",
                "integrated_external_energy_j",
                "energy_residual_j",
            ):
                if left["state"][key] != right["state"][key]:
                    raise ValueError("State-event switch introduced a stock/energy impulse")
    return error


def evaluate(spec, *, caps=None):
    caps = MESH_STEPS if caps is None else caps
    references = [reference(spec, method, TIMES) for method in ("Radau", "DOP853")]
    meshes = []
    for cap in caps:
        model = StateEventGraph(spec, max_macro_step_s=cap)
        states = [model.snapshot()]
        reports = []
        for a, b in zip(TIMES, TIMES[1:]):
            reports.append(model.advance(b - a))
            states.append(model.snapshot())
        errors = compare_states([s["model"]["engine"]["state"] for s in states], references)
        errors["event_time_s"] = verify_roots(model, reports, references)
        measured = max(
            segment["maximum_trial_internal_step_s"]
            for report in reports
            for segment in report["segments"]
        )
        meshes.append(
            dict(
                max_macro_step_s=cap,
                maximum_measured_trial_step_s=measured,
                states=states,
                reports=reports,
                errors=errors,
            )
        )
    actual = [m["maximum_measured_trial_step_s"] for m in meshes]
    if not all(a >= MINIMUM_MESH_RATIO * b > 0 for a, b in zip(actual, actual[1:])):
        raise ValueError("State-event actual time meshes are not sufficiently distinct")
    cap = caps[-1]
    a = StateEventGraph(spec, max_macro_step_s=cap)
    a.advance(7.5)
    checkpoint = a.checkpoint()
    b = StateEventGraph(spec, max_macro_step_s=cap)
    b.restore(json.loads(json.dumps(checkpoint)))
    tail = a.advance(TIMES[-1] - 7.5)
    restored = b.advance(TIMES[-1] - 7.5)
    if tail != restored:
        raise ValueError("State-event checkpoint continuation differs")
    # Checkpoint directly on a root, with right-hand mode already applied.
    stop = StateEventGraph(spec, max_macro_step_s=cap)
    stopped = stop.advance(TIMES[-1], stop_at_event=True)
    root_checkpoint = stop.checkpoint()
    other = StateEventGraph(spec, max_macro_step_s=cap)
    other.restore(json.loads(json.dumps(root_checkpoint)))
    if stopped["time_s"] < TIMES[-1]:
        a_root = stop.advance(TIMES[-1] - stopped["time_s"])
        b_root = other.advance(TIMES[-1] - stopped["time_s"])
        if a_root != b_root:
            raise ValueError("Root checkpoint continuation differs")
    return dict(
        meshes=meshes,
        references=references,
        checkpoint=checkpoint,
        restarted=tail,
        root_checkpoint=root_checkpoint,
        stopped_at_root=stopped,
        restart_values_identical=True,
    )


def rejection(spec, criterion):
    model = StateEventGraph(spec)
    before = model.checkpoint()
    try:
        model.advance(20.0)
    except NetworkError as error:
        if model.checkpoint() != before:
            raise ValueError("Failed state event partially committed") from error
        report = getattr(error, "report", dict(message=str(error)))
        if criterion is not None and report.get("criterion") != criterion:
            raise ValueError("State-event rejection has wrong diagnostic") from error
        return dict(
            diagnostics=report,
            before_checkpoint_sha256=digest(before),
            after_checkpoint_sha256=digest(model.checkpoint()),
            full_request_rolled_back=True,
        )
    raise ValueError("Invalid state event falsely accepted")


def qualify(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    captured = signature()
    for group in ("drivers", "budgets", "qualification_drivers"):
        (folder / group).mkdir()
        for path in captured[group]:
            shutil.copyfile(path, folder / group / Path(path).name)
    records = []
    negative = []
    for name, spec in cases():
        path = folder / name
        path.mkdir()
        write(path / "request.json", spec)
        result = evaluate(spec, caps=CASE_MESH_STEPS.get(name, MESH_STEPS))
        write(path / "result.json", result)
        records.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(
            json.dumps(
                dict(
                    name=name,
                    passed=True,
                    measured_steps_s=[m["maximum_measured_trial_step_s"] for m in result["meshes"]],
                )
            ),
            flush=True,
        )
    for name, spec, criterion in negative_cases():
        path = folder / "negative" / name
        path.mkdir(parents=True)
        write(path / "request.json", spec)
        write(path / "result.json", rejection(spec, criterion))
        negative.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(json.dumps(dict(name=name, rejected_as_expected=True)), flush=True)
    if signature() != captured:
        raise ValueError("State-event sources changed during campaign")
    report = dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=records,
        negative_cases=negative,
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
        raise ValueError("State-event source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, h in captured[group].items():
            if sha(folder / group / Path(path).name) != h:
                raise ValueError("State-event frozen companion mismatch")
    expected = list(cases())
    negative = list(negative_cases())
    if [r["name"] for r in report["cases"]] != [n for n, _ in expected] or [
        r["name"] for r in report["negative_cases"]
    ] != [n for n, _, _ in negative]:
        raise ValueError("Incomplete state-event matrix")
    for records, items, prefix in [
        (report["cases"], expected, folder),
        (report["negative_cases"], negative, folder / "negative"),
    ]:
        for record, item in zip(records, items):
            name, spec = item[:2]
            path = prefix / name
            if set(record["files"]) != {"request.json", "result.json"}:
                raise ValueError("Incomplete state-event packet")
            for file, h in record["files"].items():
                if sha(path / file) != h:
                    raise ValueError("State-event packet integrity mismatch")
            if json.loads((path / "request.json").read_text()) != spec:
                raise ValueError("State-event fixture mismatch")
            fresh = (
                evaluate(spec, caps=CASE_MESH_STEPS.get(name, MESH_STEPS))
                if len(item) == 2
                else rejection(spec, item[2])
            )
            if fresh != json.loads((path / "result.json").read_text()):
                raise ValueError("State-event numerical replay differs")
    if signature() != captured:
        raise ValueError("State-event sources changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=captured,
        cases=len(expected),
        production_mesh_trajectories=len(expected) * len(MESH_STEPS),
        independent_ivps=2 * len(expected),
        negative_checks=len(negative),
        maximum_nodes=max(len(s["model_input"]["volumes_m3"]) for _, s in expected),
        horizon_s=TIMES[-1],
        qualification_sha256=sha(folder / "qualification.json"),
        restart_values_identical=True,
        root_checkpoint_restore_verified=True,
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
