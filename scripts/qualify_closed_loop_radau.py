"""Source-bound closed-loop matrix with independent sampled-policy IVPs."""

import argparse
import json
from pathlib import Path

from scripts.closed_loop_radau import ClosedLoopRadau, signature
from scripts.closed_loop_references import cases, reference
from scripts.nonlinear_thermal_graph import digest
from scripts.qualify_supervised_radau import capture as supervision_capture
from scripts.solve_hydraulic_graph import sha


def capture():
    parents = supervision_capture()["qualification"]
    return dict(
        runtime=signature(),
        qualification={
            **parents,
            **{
                p: sha(p)
                for p in (
                    "scripts/qualify_supervised_radau.py",
                    "scripts/supervision_references.py",
                    "scripts/closed_loop_references.py",
                    "scripts/qualify_closed_loop_radau.py",
                )
            },
        },
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def campaign():
    captured = capture()
    rows = []
    matrix = list(cases())
    require(
        len(matrix) == 9 and len({name for name, _ in matrix}) == 9, "Incomplete closed-loop matrix"
    )
    for name, request in matrix:
        references = [reference(request, 40, method) for method in ("DOP853", "Radau")]
        caps = (0.125, 0.0625, 0.03125) if name == "eight_nodes" else (0.5, 0.25, 0.125)
        for cap in caps:
            a = ClosedLoopRadau(request, max_macro_step_s=cap)
            first = a.run(17, expected_revision=0)
            b = ClosedLoopRadau(request, max_macro_step_s=cap)
            require(
                b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot(),
                "Joint checkpoint mismatch",
            )
            second = a.run(23, expected_revision=1)
            require(second == b.run(23, expected_revision=1), "Closed-loop continuation differs")
            state = a.snapshot()
            actual = state["plant"]["engine"]["state"]
            errors = []
            for ref in references:
                require(state["plant"]["mode"] == ref["mode"], "Final mode differs")
                require(len(state["history"]) == len(ref["samples"]), "Sample count differs")
                sample_error = 0.0
                for record, sample in zip(state["history"], ref["samples"]):
                    require(
                        record["decision"] == sample["decision"],
                        "Independent policy decision differs",
                    )
                    require(record["uniform_draw"] == sample["uniform_draw"], "RNG draw differs")
                    sample_error = max(
                        sample_error,
                        abs(record["true_temperature_k"] - sample["true_temperature_k"]),
                        abs(record["measured_temperature_k"] - sample["measured_temperature_k"]),
                    )
                require(sample_error <= 1e-6, "Sample temperature error")
                history = state["plant"]["history"]
                require(
                    len(history) == len(ref["events"]), f"Plant event count differs: {name} {cap}"
                )
                event_error = 0.0
                for record, event in zip(history, ref["events"]):
                    for field in ("kind", "source_mode", "target_mode"):
                        require(record[field] == event[field], "Independent event sequence differs")
                    field = {
                        "online": "command_id",
                        "calendar": "calendar_id",
                        "state": "guard_id",
                    }[record["kind"]]
                    require(record[field] == event["id"], "Event ID differs")
                    event_error = max(event_error, abs(record["time_s"] - event["time_s"]))
                require(event_error <= 1e-5, "Event time error")
                error = dict(sample_temperature_k=sample_error, event_time_s=event_error)
                for field, tol in [
                    ("temperatures_k", 1e-6),
                    ("pressures_pa", 1e-5),
                    ("flows_kg_s", 1e-7),
                ]:
                    require(set(actual[field]) == set(ref[field]), "Observation fields differ")
                    error[field] = max(
                        abs(actual[field][key] - ref[field][key]) for key in actual[field]
                    )
                    require(error[field] <= tol, "Final observation error: " + field)
                error["energy_j"] = abs(
                    actual["integrated_external_energy_j"] - ref["integrated_external_energy_j"]
                )
                require(error["energy_j"] <= 0.1, "Final energy error")
                errors.append(error)
            rows.append(
                dict(
                    case=name,
                    cap_s=cap,
                    horizon_s=state["plant"]["time_s"],
                    ticks=40,
                    maximum_trial_step_s=max(
                        first["work"]["maximum_trial_internal_step_s"],
                        second["work"]["maximum_trial_internal_step_s"],
                    ),
                    request_sha256=digest(request),
                    checkpoint_sha256=digest(a.checkpoint()),
                    policy_history=state["history"],
                    plant_history=state["plant"]["history"],
                    errors=errors,
                )
            )
        steps = [r["maximum_trial_step_s"] for r in rows if r["case"] == name]
        require(
            all(a / b >= 1.2 for a, b in zip(steps, steps[1:])),
            ("Trial meshes not separated", name, steps),
        )
    require(capture() == captured, "Source drift during qualification")
    return dict(
        schema_version=1,
        accepted=True,
        signature=captured,
        case_count=9,
        trajectory_count=len(rows),
        independent_closed_loop_ivps=18,
        samples_per_trajectory=40,
        results=rows,
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "audit"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = campaign()
    if args.action == "run":
        args.path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    else:
        require(
            json.loads(args.path.read_text()) == result,
            "Report differs from complete recomputation",
        )
    print(json.dumps(dict(accepted=True, report_sha256=sha(args.path), signature=capture())))


if __name__ == "__main__":
    main()
