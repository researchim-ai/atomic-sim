"""Source-bound supervisory campaign; audit recomputes all results independently."""

import argparse
import json
from pathlib import Path

from scripts.nonlinear_thermal_graph import digest
from scripts.solve_hydraulic_graph import sha
from scripts.supervised_radau import BUDGET, SupervisedRadauGraph, signature
from scripts.supervision_references import cases, reference


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/supervision_references.py",
                "scripts/qualify_supervised_radau.py",
                "scripts/hysteresis_references.py",
                "scripts/state_event_references.py",
                "scripts/scheduled_graph_references.py",
                "scripts/nonlinear_graph_references.py",
                "scripts/hydraulic_graph_references.py",
                "scripts/resistance_feedback_references.py",
                "scripts/thermal_graph_references.py",
            )
        },
    )


def campaign():
    captured = capture()
    budget = json.loads(BUDGET.read_text())
    results = []
    cached = {}
    independent_ivps = 0
    for name, request, operations in cases():
        meshes = (0.5, 0.25, 0.125) if name == "bridge_pause" else (2.0, 1.0, 0.5)
        for mesh in meshes:
            model = SupervisedRadauGraph(request, max_macro_step_s=mesh)
            traces = []
            for index, (when, target, duration) in enumerate(operations):
                if when > model.snapshot()["time_s"]:
                    traces.append(model.advance(when - model.snapshot()["time_s"]))
                traces.append(
                    model.step(
                        target,
                        duration,
                        expected_revision=model.revision,
                        command_id=f"command-{index}",
                    )
                )
            traces.append(model.advance(120 - model.snapshot()["time_s"]))
            state = model.snapshot()
            restored = SupervisedRadauGraph(request, max_macro_step_s=mesh)
            if not restored.restore(model.checkpoint()) == state:
                raise ValueError("Supervisory campaign acceptance failed")
            if not model.advance(0.25) == restored.advance(0.25):
                raise ValueError("Supervisory campaign acceptance failed")
            online = [r for r in state["history"] if r["kind"] == "online"]
            errors = []
            for method in ("DOP853", "Radau"):
                key = (name, method)
                if operations or key not in cached:
                    cached[key] = reference(request, 120, online, method)
                    independent_ivps += 1
                ref = cached[key]
                if not state["mode"] == ref["mode"]:
                    raise ValueError("Supervisory campaign acceptance failed")
                if not len(state["history"]) == len(ref["history"]):
                    raise ValueError("Supervisory campaign acceptance failed")
                event_error = 0.0
                for actual, expected in zip(state["history"], ref["history"]):
                    for field in ("kind", "source_mode", "target_mode", "changed"):
                        if not actual[field] == expected[field]:
                            raise ValueError("Supervisory campaign acceptance failed")
                    field = {
                        "state": "guard_id",
                        "calendar": "calendar_id",
                        "online": "command_id",
                    }[actual["kind"]]
                    if not actual[field] == expected["id"]:
                        raise ValueError("Supervisory campaign acceptance failed")
                    event_error = max(event_error, abs(actual["time_s"] - expected["time_s"]))
                if not event_error <= budget["maximum_reference_event_time_error_s"]:
                    raise ValueError("Supervisory campaign acceptance failed")
                row = dict(method=method, event_time_error_s=event_error)
                for field, limit in [
                    ("temperatures_k", "maximum_reference_temperature_error_k"),
                    ("pressures_pa", "maximum_reference_pressure_error_pa"),
                    ("flows_kg_s", "maximum_reference_flow_error_kg_s"),
                ]:
                    actual = state["engine"]["state"][field]
                    if not set(actual) == set(ref[field]):
                        raise ValueError("Supervisory campaign acceptance failed")
                    error = max((abs(actual[k] - ref[field][k]) for k in actual))
                    if not error <= budget[limit]:
                        raise ValueError((name, mesh, field, error))
                    row[field + "_maximum_error"] = error
                row["energy_error_j"] = abs(
                    state["engine"]["state"]["integrated_external_energy_j"]
                    - ref["integrated_external_energy_j"]
                )
                if not row["energy_error_j"] <= budget["maximum_reference_energy_error_j"]:
                    raise ValueError("Supervisory campaign acceptance failed")
                errors.append(row)
            results.append(
                dict(
                    case=name,
                    mesh_s=mesh,
                    maximum_trial_step_s=max(
                        (
                            segment["maximum_trial_internal_step_s"]
                            for trace in traces
                            for segment in trace["segments"]
                        )
                    ),
                    request_sha256=digest(request),
                    checkpoint_sha256=digest(model.checkpoint()),
                    history=state["history"],
                    reference_errors=errors,
                )
            )
    for name, _, _ in cases():
        steps = [r["maximum_trial_step_s"] for r in results if r["case"] == name]
        if not all((a / b >= 1.2 for a, b in zip(steps, steps[1:]))):
            raise ValueError((name, steps))
    if not capture() == captured:
        raise ValueError("Sources changed during campaign")
    return dict(
        schema_version=1,
        accepted=True,
        scope="local fixed-property supervisory software qualification",
        signature=captured,
        case_count=9,
        trajectory_count=len(results),
        independent_ivp_count=independent_ivps,
        validated=False,
        plant_referenced=False,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "audit"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    actual = campaign()
    if args.action == "audit":
        if json.loads(args.path.read_text()) != actual:
            raise ValueError("Campaign differs from independent recomputation")
        print(json.dumps(dict(accepted=True, report_sha256=sha(args.path), signature=capture())))
    else:
        args.path.write_text(json.dumps(actual, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    accepted=True,
                    trajectory_count=actual["trajectory_count"],
                    report_sha256=sha(args.path),
                )
            )
        )


if __name__ == "__main__":
    main()
