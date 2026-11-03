"""Source-bound closed buoyant loops, independent transients and stationary branches."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from scripts.buoyant_graph_references import IndependentBuoyant, cases
from scripts.buoyant_inertial_graph import (
    BUDGET,
    LEDGERS,
    InertialGraphError,
    digest,
    signature,
)
from scripts.buoyant_inertial_graph import (
    BuoyantInertialGraph as InertialGraph,
)
from scripts.solve_hydraulic_graph import sha

TIMES = (2.0, 10.0, 10.5, 20.0, 25.0, 25.5, 40.0)
CAPS = (0.1, 0.05, 0.025)


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/thermal_inertial_references.py",
                "scripts/inertial_graph_references.py",
                "scripts/qualify_buoyant_graph.py",
                "scripts/buoyant_graph_references.py",
            )
        },
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def negative_checks():
    matrix = {n: r for n, r, _ in cases()}
    results = []
    bad = deepcopy(matrix["heated_startup"])
    bad["buoyancy"]["maximum_relative_density_change"] = 0.1
    try:
        InertialGraph(bad)
    except ValueError as e:
        results.append(dict(case="invalid_contrast_limit", accepted=False, error=str(e)))
    else:
        raise ValueError("Invalid buoyancy contract accepted")
    for name in ("density", "temperature"):
        r = deepcopy(matrix["uniform_heating"])
        if name == "density":
            r["buoyancy"]["maximum_relative_density_change"] = 0.001
        else:
            r["buoyancy"]["expansion_per_k"] = 0
        a = InertialGraph(r)
        before = a.checkpoint()
        try:
            a.advance(100)
        except InertialGraphError as e:
            require(a.checkpoint() == before, "Negative buoyant interval partially committed")
            results.append(dict(case=name, accepted=False, criterion=e.report["criterion"]))
        else:
            raise ValueError("Out-of-domain buoyant transient accepted")
    return results


def stationary_checks(matrix):
    rows = []
    for name, r, _ in matrix:
        if name not in ("heated_startup", "mirror_startup", "eight_nodes"):
            continue
        steady = IndependentBuoyant(r).stationary(-0.13 if name == "mirror_startup" else 0.13)
        seeded = deepcopy(r)
        seeded["base"]["hydraulic"]["initial_flows_kg_s"] = steady["flows_kg_s"]
        for node, T in steady["temperatures_k"].items():
            seeded["base"]["thermal"]["nodes"][node]["initial_temperature_k"] = T
        a = InertialGraph(seeded, max_step_s=0.2)
        initial = a.snapshot()["state"]
        final = a.advance(40)["state"]
        errors = {}
        for field, limit in [
            ("flows_kg_s", 1e-7),
            ("temperatures_k", 1e-6),
            ("pressures_pa", 1e-5),
        ]:
            error = max(abs(final[field][k] - steady[field][k]) for k in steady[field])
            require(error <= limit, ("Stationary branch drift", name, field, error))
            errors[field] = error
        require(abs(final["thermal_energy_change_j"]) < 1e-3, "Stationary thermal inventory drift")
        rows.append(dict(case=name, reference=steady, initial=initial, final=final, errors=errors))
    require(len(rows) == 3, "Incomplete nontrivial stationary matrix")
    return rows


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
                "temperatures_k",
                "kinetic_energy_j",
                "thermal_energy_change_j",
                *LEDGERS,
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
            IndependentBuoyant(spec).integrate(40.0, schedule, method, TIMES)
            for method in ("DOP853", "Radau")
        ]
        caps = CAPS
        for cap in caps:
            output = trajectory(spec, schedule, cap)
            errors = []
            for reference in references:
                require(len(reference) == len(output["observations"]), "Observation count differs")
                error = dict(flow_kg_s=0.0, pressure_pa=0.0, temperature_k=0.0, energy_j=0.0)
                for actual, expected in zip(output["observations"], reference):
                    require(actual["time_s"] == expected["time_s"], "Observation clock differs")
                    for field, key, limit in [
                        ("flows_kg_s", "flow_kg_s", "reference_flow_atol_kg_s"),
                        ("pressures_pa", "pressure_pa", "reference_pressure_atol_pa"),
                        ("temperatures_k", "temperature_k", "reference_temperature_atol_k"),
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
                        abs(actual[k] - value) for k, value in zip(LEDGERS, expected["ledgers"])
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
    stationary = stationary_checks(matrix)
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
        stationary_checks=stationary,
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
