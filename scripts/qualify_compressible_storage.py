"""Source-bound compressible storage, complete synthetic EOS and mass/energy/entropy budgets."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from scripts.compressible_storage_graph import (
    BUDGET,
    LEDGERS,
    InertialGraphError,
    StiffenedCaloricEOS,
    digest,
    signature,
)
from scripts.compressible_storage_graph import (
    CompressibleStorageGraph as InertialGraph,
)
from scripts.solve_hydraulic_graph import sha
from scripts.storage_graph_references import IndependentStorage, cases, fluid

TIMES = (1.0, 5.0, 5.25, 10.0, 12.0, 12.25, 20.0)
CAPS = (0.05, 0.025, 0.0125)


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/storage_graph_references.py",
                "scripts/qualify_compressible_storage.py",
            )
        },
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def negative_checks():
    matrix = {n: r for n, r, _ in cases()}
    results = []
    bad = deepcopy(matrix["sealed_heat"])
    bad["fluid"]["gamma"] = 1.0
    try:
        InertialGraph(bad)
    except ValueError as e:
        results.append(dict(case="invalid_eos", accepted=False, error=str(e)))
    else:
        raise ValueError("Invalid EOS accepted")
    for name in ("temperature", "volume", "density"):
        r = deepcopy(
            matrix["adiabatic_compression"]
            if name == "volume"
            else matrix["mass_energy_equalization"]
        )
        if name == "density":
            r["nodes"][0]["initial_temperature_k"] = 600.0
            f = r["fluid"]
            rho = (r["nodes"][0]["initial_pressure_pa"] + f["pressure_shift_pa"]) / (
                (f["gamma"] - 1) * f["cv_j_kg_k"] * 600
            )
            r["domain"]["min_density_kg_m3"] = rho * 0.9999
        a = InertialGraph(r)
        before = a.checkpoint()
        try:
            if name == "temperature":
                a.step(
                    {n: -1e6 for n in a.system.names},
                    10,
                    expected_revision=0,
                    command_id="cooldown",
                )
            else:
                a.advance(100 if name == "volume" else 10)
        except InertialGraphError as e:
            require(a.checkpoint() == before, "Invalid storage call partially committed")
            results.append(dict(case=name, accepted=False, criterion=e.report["criterion"]))
        else:
            raise ValueError("Out-of-domain storage transient accepted")
    return results


def eos_checks():
    rows = []
    for shift in (0.0, 1e5, 1e8):
        eos = StiffenedCaloricEOS(fluid(shift))
        for T in (250.0, 300.0, 400.0):
            point = eos.p_T(2e5, T)
            rho = point["density_kg_m3"]
            dr = rho * 1e-5
            inverse = eos.rho_u(rho, point["internal_energy_j_kg"])
            require(abs(inverse["pressure_pa"] - 2e5) < 0.001, "EOS inverse pressure failed")
            upper = eos.rho_T(rho + dr, T * ((rho + dr) / rho) ** (eos.gamma - 1))
            lower = eos.rho_T(rho - dr, T * ((rho - dr) / rho) ** (eos.gamma - 1))
            sound = (upper["pressure_pa"] - lower["pressure_pa"]) / (2 * dr)
            sound_error = abs(sound / point["sound_speed_squared_m2_s2"] - 1)
            require(sound_error < 1e-6, "Isentropic sound-speed derivative failed")
            up = eos.rho_T(rho + dr, T + 0.001)
            down = eos.rho_T(rho - dr, T - 0.001)
            du = (up["internal_energy_j_kg"] - down["internal_energy_j_kg"]) / 2
            ds = (up["entropy_j_kg_k"] - down["entropy_j_kg_k"]) / 2
            dv = (1 / (rho + dr) - 1 / (rho - dr)) / 2
            gibbs = abs(du - T * ds + point["pressure_pa"] * dv)
            require(gibbs < 1e-5, "Gibbs differential identity failed")
            rows.append(
                dict(
                    pressure_shift_pa=shift,
                    temperature_k=T,
                    point=point,
                    inverse=inverse,
                    sound_relative_error=sound_error,
                    gibbs_error_j_kg=gibbs,
                )
            )
    return rows


def equilibrium_check(matrix):
    r = deepcopy(next(r for name, r, _ in matrix if name == "mass_energy_equalization"))
    a = InertialGraph(r, max_step_s=0.2)
    s = a.system
    expectedT = sum(s.A0) / (s.eos.cv * sum(s.m0))
    expectedP = sum(s.p0 * s.V0) / sum(s.V0)
    expectedRho = sum(s.m0) / sum(s.V0)
    final = a.advance(200)["state"]
    require(
        max(abs(v - expectedT) for v in final["temperatures_k"].values()) < 1e-6,
        "Analytic equilibrium temperature failed",
    )
    require(
        max(abs(v - expectedP) for v in final["pressures_pa"].values()) < 0.001,
        "Analytic equilibrium pressure failed",
    )
    require(
        max(abs(v - expectedRho) for v in final["densities_kg_m3"].values()) < 1e-7,
        "Analytic equilibrium density failed",
    )
    require(final["generated_entropy_j_k"] > 0, "Equilibrium entropy production not positive")
    return dict(
        horizon_s=200,
        expected_temperature_k=float(expectedT),
        expected_pressure_pa=float(expectedP),
        expected_density_kg_m3=float(expectedRho),
        final=final,
    )


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
                values, 0.25, expected_revision=a.revision, command_id="boundary-" + str(end)
            )
            if b is not None:
                require(
                    out
                    == b.step(
                        values,
                        0.25,
                        expected_revision=b.revision,
                        command_id="boundary-" + str(end),
                    ),
                    "Restart command differs",
                )
            transition = out["heater_transition"]
            left, right = transition["left"], transition["right"]
            for field in (
                "masses_kg",
                "volumes_m3",
                "temperatures_k",
                "pressures_pa",
                "internal_energies_j",
                *LEDGERS,
            ):
                require(left[field] == right[field], "Boundary command introduced a stock impulse")
            rows.append(right)
            work.append(out["work"])
        else:
            rows.append(a.snapshot()["state"])
        if end == 10:
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
            IndependentStorage(spec).integrate(20.0, schedule, method, TIMES)
            for method in ("DOP853", "Radau")
        ]
        caps = CAPS
        for cap in caps:
            output = trajectory(spec, schedule, cap)
            errors = []
            for reference in references:
                require(len(reference) == len(output["observations"]), "Observation count differs")
                error = dict(
                    mass_kg=0.0, pressure_pa=0.0, temperature_k=0.0, energy_j=0.0, entropy_j_k=0.0
                )
                for actual, expected in zip(output["observations"], reference):
                    require(actual["time_s"] == expected["time_s"], "Observation clock differs")
                    for field, key, limit in [
                        ("masses_kg", "mass_kg", "reference_mass_atol_kg"),
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
                    for index, key in enumerate(LEDGERS):
                        value = abs(actual[key] - expected["ledgers"][index])
                        energy = index < 2
                        limit = budget[
                            "reference_energy_atol_j" if energy else "reference_entropy_atol_j_k"
                        ]
                        require(value <= limit, (name, cap, key, value))
                        field = "energy_j" if energy else "entropy_j_k"
                        error[field] = max(error[field], value)
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
    eos = eos_checks()
    equilibrium = equilibrium_check(matrix)
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
        horizon_s=20.0,
        negative_checks=negative,
        eos_checks=eos,
        equilibrium_check=equilibrium,
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
