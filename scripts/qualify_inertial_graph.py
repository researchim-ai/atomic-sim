"""Source-bound inertial graph matrix with independent bordered IVPs."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from scripts.inertial_graph import BUDGET, InertialGraph, InertialGraphError, digest, signature
from scripts.inertial_graph_references import IndependentInertial, cases, junction
from scripts.solve_hydraulic_graph import sha

TIMES = (2.0, 10.0, 10.5, 20.0, 25.0, 25.5, 40.0)
CAPS = (0.2, 0.1, 0.05)


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in ("scripts/inertial_graph_references.py", "scripts/qualify_inertial_graph.py")
        },
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def negative_checks():
    matrix = {name: spec for name, spec, _ in cases()}
    results = []
    bad = deepcopy(matrix["series_junction"])
    bad["initial_flows_kg_s"]["aj"] = 1.0
    unanchored = deepcopy(matrix["single_pipe"])
    unanchored["nodes"] = [junction(n["id"]) for n in unanchored["nodes"]]
    for name, spec in [("inconsistent_initial_mass", bad), ("missing_pressure_anchor", unanchored)]:
        try:
            InertialGraph(spec)
        except ValueError as error:
            results.append(dict(case=name, accepted=False, error=str(error)))
        else:
            raise ValueError("Invalid inertial input accepted")
    limited = deepcopy(matrix["single_pipe"])
    limited["max_abs_flow_kg_s"] = 3
    a = InertialGraph(limited)
    before = a.checkpoint()
    try:
        a.step(dict(a=1000000.0, b=200000.0), 40.0, expected_revision=0, command_id="overload")
    except InertialGraphError as error:
        require(a.checkpoint() == before, "Negative command partially committed")
        results.append(
            dict(case="trial_flow_overload", accepted=False, criterion=error.report["criterion"])
        )
    else:
        raise ValueError("Out-of-domain transient accepted")
    return results


def trajectory(spec, schedule, cap):
    a = InertialGraph(spec, max_step_s=cap)
    b = None
    rows = []
    work = []
    boundaries = dict(schedule)
    for end in TIMES:
        current = a.snapshot()["state"]["time_s"]
        if current < end:
            out = a.advance(end - current)
            if b is not None:
                require(out == b.advance(end - current), "Restart differs")
            work.append(out["work"])
        if end in boundaries:
            values = boundaries[end]
            out = a.step(
                values, 0.5, expected_revision=a.revision, command_id="boundary-" + str(end)
            )
            if b is not None:
                require(
                    out
                    == b.step(
                        values, 0.5, expected_revision=b.revision, command_id="boundary-" + str(end)
                    ),
                    "Restart command differs",
                )
            transition = out["boundary_transition"]
            left, right = transition["left"], transition["right"]
            for field in (
                "flows_kg_s",
                "kinetic_energy_j",
                "pressure_work_j",
                "gravity_work_j",
                "exported_heat_j",
            ):
                require(left[field] == right[field], "Boundary command introduced a stock impulse")
            rows.append(right)
            work.append(out["work"])
        else:
            rows.append(a.snapshot()["state"])
        if end == 20:
            b = InertialGraph(spec, max_step_s=cap)
            require(
                b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot(),
                "Checkpoint differs",
            )
    return dict(
        observations=rows,
        history=a.snapshot()["history"],
        checkpoint_sha256=digest(a.checkpoint()),
        maximum_internal_step_s=max(r["maximum_internal_step_s"] for r in work),
        rhs=sum(r["rhs"] for r in work),
    )


def campaign():
    captured = capture()
    budget = json.loads(BUDGET.read_text())
    results = []
    matrix = list(cases())
    require(
        len(matrix) == 9 and len({n for n, _, _ in matrix}) == 9, "Incomplete inertial case matrix"
    )
    for name, spec, schedule in matrix:
        references = [
            IndependentInertial(spec).integrate(40.0, schedule, method, TIMES)
            for method in ("DOP853", "Radau")
        ]
        caps = (0.05, 0.025, 0.0125) if name == "boundary_reversal" else CAPS
        for cap in caps:
            output = trajectory(spec, schedule, cap)
            errors = []
            for reference in references:
                require(len(reference) == len(output["observations"]), "Observation count differs")
                error = dict(flow_kg_s=0.0, pressure_pa=0.0, energy_j=0.0)
                for actual, expected in zip(output["observations"], reference):
                    require(actual["time_s"] == expected["time_s"], "Observation clock differs")
                    for field, key, limit in [
                        ("flows_kg_s", "flow_kg_s", "reference_flow_atol_kg_s"),
                        ("pressures_pa", "pressure_pa", "reference_pressure_atol_pa"),
                    ]:
                        require(
                            set(actual[field]) == set(expected[field]),
                            "Observation channels differ",
                        )
                        value = max(
                            abs(actual[field][k] - expected[field][k]) for k in actual[field]
                        )
                        require(value <= budget[limit], (name, cap, field, value))
                        error[key] = max(error[key], value)
                    energy = max(
                        abs(actual[k] - expected[k])
                        for k in ("pressure_work_j", "gravity_work_j", "exported_heat_j")
                    )
                    require(
                        energy <= budget["reference_energy_atol_j"], (name, cap, "energy", energy)
                    )
                    error["energy_j"] = max(error["energy_j"], energy)
                errors.append(error)
            results.append(
                dict(
                    case=name,
                    cap_s=cap,
                    spec_sha256=digest(spec),
                    schedule_sha256=digest(schedule),
                    errors=errors,
                    **output,
                )
            )
        steps = [r["maximum_internal_step_s"] for r in results if r["case"] == name]
        require(
            all(a / b >= 1.2 for a, b in zip(steps, steps[1:])),
            ("Trial meshes not separated", name, steps),
        )
    negative = negative_checks()
    require(capture() == captured, "Source drift during qualification")
    return dict(
        schema_version=1,
        accepted=True,
        signature=captured,
        case_count=9,
        trajectory_count=27,
        independent_ivp_count=18,
        observations_per_trajectory=len(TIMES),
        horizon_s=40.0,
        negative_checks=negative,
        results=results,
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
            "Inertial report differs from complete recomputation",
        )
    print(json.dumps(dict(accepted=True, report_sha256=sha(args.path), signature=capture())))


if __name__ == "__main__":
    main()
