"""Independent verification of repeatable finite-mode thermal control."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from core.network import NetworkError
from scripts.hysteresis_references import cases, negative_cases, reference
from scripts.hysteretic_radau import BUDGET, HystereticRadauGraph
from scripts.hysteretic_radau import signature as model_signature
from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import sha

EXTRA = (
    "scripts/qualify_hysteretic_radau.py",
    "scripts/hysteresis_references.py",
    "scripts/state_event_references.py",
    "scripts/scheduled_graph_references.py",
    "scripts/nonlinear_graph_references.py",
    "scripts/hydraulic_graph_references.py",
    "scripts/resistance_feedback_references.py",
    "scripts/thermal_graph_references.py",
)
TIMES = (0.0, 2.0, 11.0, 37.5, 61.0, 120.0)
MESH_STEPS = (2.0, 1.0, 0.5)
CASE_MESH_STEPS = {"standby": (1.0, 0.5, 0.25), "bridge_relay": (0.5, 0.25, 0.125)}
MINIMUM_MESH_RATIO = 1.2


def signature():
    value = model_signature()
    value["qualification_drivers"] = {p: sha(p) for p in EXTRA}
    return value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(name, spec):
    budget = json.loads(BUDGET.read_text())
    caps = CASE_MESH_STEPS.get(name, MESH_STEPS)
    references = [reference(spec, method, TIMES) for method in ("Radau", "DOP853")]
    meshes = []
    for cap in caps:
        model = HystereticRadauGraph(spec, max_macro_step_s=cap)
        states = [model.snapshot()]
        reports = []
        for a, b in zip(TIMES, TIMES[1:]):
            reports.append(model.advance(b - a))
            states.append(model.snapshot())
        errors = {}
        for key, limit in [
            ("temperatures_k", "maximum_reference_temperature_error_k"),
            ("pressures_pa", "maximum_reference_pressure_error_pa"),
            ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
        ]:
            value = max(
                abs(s["engine"]["state"][key][node] - r["states"][i][key][node])
                for r in references
                for i, s in enumerate(states)
                for node in s["engine"]["state"][key]
            )
            errors[key] = value
            if not np.isfinite(value) or value > budget[limit]:
                raise ValueError("Hysteretic reference budget exceeded: " + key)
        energy = max(
            abs(
                s["engine"]["state"]["integrated_external_energy_j"]
                - r["states"][i]["integrated_external_energy_j"]
            )
            for r in references
            for i, s in enumerate(states)
        )
        balance = max(abs(s["engine"]["state"]["energy_residual_j"]) for s in states)
        if (
            not np.isfinite(energy)
            or energy > budget["maximum_reference_energy_error_j"]
            or balance > budget["maximum_energy_error_j"]
        ):
            raise ValueError("Hysteretic reference energy budget exceeded")
        history = model.snapshot()["history"]
        root_errors = []
        for r in references:
            if [(x["guard_id"], x["source_mode"], x["target_mode"]) for x in history] != [
                (x["guard_id"], x["source_mode"], x["target_mode"]) for x in r["history"]
            ]:
                raise ValueError("Independent hysteretic transition path differs")
            if [(s["mode"], s["transition_count"]) for s in states] != [
                (s["mode"], s["transition_count"]) for s in r["states"]
            ]:
                raise ValueError("Independent sampled mode state differs")
            root_errors.extend(
                abs(a["time_s"] - b["time_s"]) for a, b in zip(history, r["history"])
            )
        root_error = max(root_errors, default=0.0)
        if root_error > budget["maximum_reference_event_time_error_s"]:
            raise ValueError("Hysteretic event-time budget exceeded")
        previous = 0.0
        for record in history:
            if record["bracket_s"][0] - previous < spec["minimum_residence_s"]:
                raise ValueError("Hysteretic minimum residence not certified")
            previous = record["time_s"]
        for report in reports:
            for switch in report["switches"]:
                a, b = switch["left"], switch["right"]
                if (
                    a["time_s"] != b["time_s"]
                    or a["increments"] != b["increments"]
                    or any(
                        a["state"][key] != b["state"][key]
                        for key in (
                            "temperatures_k",
                            "temperature_increments_k",
                            "integrated_external_energy_j",
                            "energy_residual_j",
                        )
                    )
                ):
                    raise ValueError("Mode switch introduced a stock/energy impulse")
        errors.update(reference_energy_j=energy, energy_residual_j=balance, event_time_s=root_error)
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
                transition_count=len(history),
            )
        )
    actual = [m["maximum_measured_trial_step_s"] for m in meshes]
    if not all(a >= MINIMUM_MESH_RATIO * b > 0 for a, b in zip(actual, actual[1:])):
        raise ValueError("Hysteretic trial meshes are not sufficiently distinct")
    cap = caps[-1]
    a = HystereticRadauGraph(spec, max_macro_step_s=cap)
    a.advance(37.5)
    checkpoint = a.checkpoint()
    b = HystereticRadauGraph(spec, max_macro_step_s=cap)
    b.restore(json.loads(json.dumps(checkpoint)))
    x = a.advance(TIMES[-1] - 37.5)
    y = b.advance(TIMES[-1] - 37.5)
    if x != y:
        raise ValueError("Hysteretic checkpoint continuation differs")
    stopped = HystereticRadauGraph(spec, max_macro_step_s=cap)
    first = stopped.advance(TIMES[-1], stop_at_transition=True)
    root_checkpoint = stopped.checkpoint()
    other = HystereticRadauGraph(spec, max_macro_step_s=cap)
    other.restore(json.loads(json.dumps(root_checkpoint)))
    if first["time_s"] < TIMES[-1]:
        if stopped.advance(TIMES[-1] - first["time_s"]) != other.advance(
            TIMES[-1] - first["time_s"]
        ):
            raise ValueError("Hysteretic root-checkpoint continuation differs")
    return dict(
        meshes=meshes,
        references=references,
        checkpoint=checkpoint,
        restarted=x,
        root_checkpoint=root_checkpoint,
        stopped_at_transition=first,
        restart_values_identical=True,
    )


def rejection(spec, criterion):
    model = HystereticRadauGraph(spec)
    before = model.checkpoint()
    try:
        model.advance(20.0)
    except NetworkError as error:
        if model.checkpoint() != before:
            raise ValueError("Failed hysteretic transition partially committed") from error
        report = getattr(error, "report", dict(message=str(error)))
        if criterion is not None and report.get("criterion") != criterion:
            raise ValueError("Unexpected hysteretic refusal diagnostic") from error
        return dict(
            diagnostics=report,
            before_checkpoint_sha256=digest(before),
            after_checkpoint_sha256=digest(model.checkpoint()),
            full_request_rolled_back=True,
        )
    raise ValueError("Invalid hysteretic program falsely accepted")


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
        result = evaluate(name, spec)
        write(path / "result.json", result)
        records.append(dict(name=name, files={p.name: sha(p) for p in sorted(path.iterdir())}))
        print(
            json.dumps(
                dict(
                    name=name,
                    passed=True,
                    transition_count=result["meshes"][-1]["transition_count"],
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
        raise ValueError("Hysteretic sources changed during campaign")
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
        raise ValueError("Hysteretic source signature mismatch")
    for group in ("drivers", "budgets", "qualification_drivers"):
        for path, h in captured[group].items():
            if sha(folder / group / Path(path).name) != h:
                raise ValueError("Hysteretic frozen companion mismatch")
    expected = list(cases())
    negative = list(negative_cases())
    if [r["name"] for r in report["cases"]] != [n for n, _ in expected] or [
        r["name"] for r in report["negative_cases"]
    ] != [n for n, _, _ in negative]:
        raise ValueError("Incomplete hysteretic matrix")
    for records, items, prefix in [
        (report["cases"], expected, folder),
        (report["negative_cases"], negative, folder / "negative"),
    ]:
        for record, item in zip(records, items):
            name, spec = item[:2]
            path = prefix / name
            if set(record["files"]) != {"request.json", "result.json"}:
                raise ValueError("Incomplete hysteretic packet")
            for file, h in record["files"].items():
                if sha(path / file) != h:
                    raise ValueError("Hysteretic packet integrity mismatch")
            if json.loads((path / "request.json").read_text()) != spec:
                raise ValueError("Hysteretic fixture mismatch")
            fresh = evaluate(name, spec) if len(item) == 2 else rejection(spec, item[2])
            if fresh != json.loads((path / "result.json").read_text()):
                raise ValueError("Hysteretic numerical replay differs")
    if signature() != captured:
        raise ValueError("Hysteretic sources changed during audit")
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
        transition_checkpoint_restore_verified=True,
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
