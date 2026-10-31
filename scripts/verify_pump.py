"""Displacement-pump reference integrations and reproducible acceptance evidence.

The reference uses explicit DOP853 on independent absolute-temperature equations,
not the production RHS, Jacobian or relative-energy implementation.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from physics.hydraulics.pump_loop import PumpCirculationLoop, steady_pump_loop
from physics.hydraulics.valved_pump_loop import ValvedPumpLoop
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.pump_experiment import run_pump_experiment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valve_opening(spec, t):
    v = spec["valve"]
    epoch, opening, target = 0.0, v["initial_opening"], v["initial_opening"]
    for c in spec["timeline"]:
        if c["t"] > t:
            break
        if "valve_opening" in c:
            opening = target + (opening - target) * np.exp(-(c["t"] - epoch) / v["actuator_time_s"])
            epoch, target = c["t"], c["valve_opening"]
    return float(target + (opening - target) * np.exp(-(t - epoch) / v["actuator_time_s"]))


def resistance_at(spec, order, resistance, t):
    if "valve" not in spec:
        return resistance
    v = spec["valve"]
    a = valve_opening(spec, t)
    rho = spec["loop"]["model"]["fluid"]["density_kg_m3"]
    area = spec["loop"]["model"]["fluid"]["area_m2"]
    result = resistance.copy()
    result[order.index(v["segment"])] += (
        v["open_loss_coefficient"]
        + (v["closed_loss_coefficient"] - v["open_loss_coefficient"]) * (1 - a) ** 2
    ) / (2 * rho * area**2)
    return result


def reference(spec):
    model = spec["loop"]["model"]
    pump = spec["pump"]
    init = spec["loop"]["initialization"]
    nodes = {x["id"]: x["parameters"] for x in model["topology"]["nodes"]}
    links = {
        x["from"].split(".")[0]: x["to"].split(".")[0] for x in model["topology"]["connections"]
    }
    order = [model["reference"]["segment"]]
    while len(order) < len(nodes):
        order.append(links[order[-1]])
    rho = model["fluid"]["density_kg_m3"]
    cp = model["fluid"]["heat_capacity_j_kg_k"]
    area = model["fluid"]["area_m2"]
    length = np.array([nodes[x]["length_m"] for x in order])
    inertia = sum(length) / area
    resistance = np.array([nodes[x]["loss_coefficient"] for x in order]) / (2 * rho * area**2)
    capacity = rho * area * length * cp
    wall = np.array([nodes[x]["wall_capacity_j_k"] for x in order])
    ua = np.array([nodes[x]["fluid_wall_ua_w_k"] for x in order])
    ambient = np.array([nodes[x]["ambient_ua_w_k"] for x in order])
    heater = np.array([nodes[x]["wall_power_w"] for x in order])
    n = len(order)
    index = order.index(pump["segment"])
    initial = np.r_[
        init["mass_flow_kg_s"],
        spec["initial_omega_rad_s"],
        [init["fluid_temperature_k"][x] for x in order],
        [init["wall_temperature_k"][x] for x in order],
        np.zeros(4),
    ]

    def rhs(t, z, torque):
        flow, omega = z[:2]
        tf = z[2 : 2 + n]
        tw = z[2 + n : 2 + 2 * n]
        head = (pump["displacement_m3_rad"] * omega - flow / rho) / pump["leakage_m3_s_pa"]
        leak = pump["leakage_m3_s_pa"] * head**2
        qwall = ua * (tw - tf)
        r = resistance_at(spec, order, resistance, t)
        fluid_heat = (
            cp * abs(flow) * (np.roll(tf, 1 if flow >= 0 else -1) - tf)
            + r * abs(flow) ** 3 / rho
            + qwall
        )
        wall_heat = heater - qwall + ambient * (model["ambient_temperature_k"] - tw)
        wall_heat[index] += leak
        return np.r_[
            (head - sum(r) * flow * abs(flow)) / inertia,
            (torque - pump["displacement_m3_rad"] * head - pump["drag_nm_s_rad"] * omega)
            / pump["inertia_kg_m2"],
            fluid_heat / capacity,
            wall_heat / wall,
            torque * omega,
            pump["drag_nm_s_rad"] * omega**2,
            leak,
            head * flow / rho,
        ]

    pieces = []
    events = []
    state = initial.copy()
    start = 0.0
    torque = spec["motor_torque_nm"]
    for command in spec["timeline"] + [dict(t=spec["duration_s"], motor_torque_nm=0)]:
        end = command["t"]
        if end > start:
            result = solve_ivp(
                lambda t, y: rhs(t, y, torque),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-13,
                atol=1e-12,
                max_step=0.02,
                dense_output=True,
            )
            if not result.success:
                raise ValueError("Independent explicit reference failed")
            for index_event, kind in ((0, "flow_zero"), (1, "shaft_zero")):
                for i in range(len(result.t) - 1):
                    a, b = result.y[index_event, i : i + 2]
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        root = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: result.sol(t)[index_event],
                                result.t[i],
                                result.t[i + 1],
                                xtol=1e-12,
                            )
                        )
                        events.append(dict(kind=kind, t=root, direction=direction))
            pieces.append((start, end, result.sol))
            state = result.y[:, -1]
        start = end
        torque = command.get("motor_torque_nm", torque)

    def at(t):
        if t == 0:
            return initial.copy()
        return next(sol(t) for a, b, sol in pieces if a <= t <= b)

    return at, events, order, capacity, wall, length, resistance


def assess(folder, budget):
    folder = Path(folder)
    spec = json.loads((folder / "input.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    for name, h in manifest["artifacts"].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p) != h:
            raise ValueError("Bundle checksum/path mismatch")
    if manifest["source_sha256"] != _source_fingerprint():
        raise ValueError("Pump source mismatch")
    at, events, order, capacity, wall, length, resistance = reference(spec)
    model = spec["loop"]["model"]
    p = spec["pump"]
    rho = model["fluid"]["density_kg_m3"]
    area = model["fluid"]["area_m2"]
    n = len(order)
    index = order.index(p["segment"])
    initial = at(0)
    metrics = dict(
        mass_flow_abs_kg_s=0.0,
        omega_abs_rad_s=0.0,
        temperature_abs_k=0.0,
        integral_abs_j=0.0,
        energy_abs_j=0.0,
        power_transfer_abs_w=0.0,
        pressure_abs_pa=0.0,
        event_time_abs_s=0.0,
    )
    if "valve" in spec:
        metrics["opening_abs"] = 0.0
    for row in rows:
        if "valve" in spec:
            metrics["opening_abs"] = max(
                metrics["opening_abs"],
                abs(row["valve_opening"] - valve_opening(spec, row["time_s"])),
            )
        z = at(row["time_s"])
        o = row["loop"]
        flow = o["mass_flow_kg_s"]
        omega = row["omega_rad_s"]
        metrics["mass_flow_abs_kg_s"] = max(metrics["mass_flow_abs_kg_s"], abs(flow - z[0]))
        metrics["omega_abs_rad_s"] = max(metrics["omega_abs_rad_s"], abs(omega - z[1]))
        temperatures = np.r_[
            [o["fluid_temperature_k"][x] for x in order],
            [o["wall_temperature_k"][x] for x in order],
        ]
        metrics["temperature_abs_k"] = max(
            metrics["temperature_abs_k"], float(max(abs(temperatures - z[2 : 2 + 2 * n])))
        )
        observed = [
            row["motor_work_j"],
            row["bearing_heat_j"],
            row["leakage_heat_j"],
            o["external_head_work_j"],
        ]
        metrics["integral_abs_j"] = max(
            metrics["integral_abs_j"], float(max(abs(np.array(observed) - z[-4:])))
        )
        energy = float(
            np.r_[capacity, wall] @ (temperatures - initial[2 : 2 + 2 * n])
            + sum(length) / area * (flow**2 - initial[0] ** 2) / (2 * rho)
            + p["inertia_kg_m2"] * (omega**2 - initial[1] ** 2) / 2
            - row["motor_work_j"]
            + row["bearing_heat_j"]
            - row["external_ambient_heater_heat_j"]
        )
        metrics["energy_abs_j"] = max(
            metrics["energy_abs_j"],
            abs(energy),
            abs(row["total_energy_residual_j"]),
            abs(row["shaft_energy_residual_j"]),
        )
        transfer = (
            p["displacement_m3_rad"] * row["head_pa"] * omega
            - row["head_pa"] * flow / rho
            - p["leakage_m3_s_pa"] * row["head_pa"] ** 2
        )
        metrics["power_transfer_abs_w"] = max(metrics["power_transfer_abs_w"], abs(transfer))
        # Pressure reference is independently reconstructed from the exact reference flow.
        head = (p["displacement_m3_rad"] * z[1] - z[0] / rho) / p["leakage_m3_s_pa"]
        heads = np.zeros(n)
        heads[index] = head
        r = resistance_at(spec, order, resistance, row["time_s"])
        acceleration = (head - sum(r) * z[0] * abs(z[0])) / (sum(length) / area)
        dp = length / area * acceleration + r * z[0] * abs(z[0]) - heads
        faces = np.r_[
            model["reference"]["pressure_pa"],
            model["reference"]["pressure_pa"] - np.cumsum(dp)[:-1],
        ]
        for i, name in enumerate(order):
            a = o["ports"][name + ".right"]
            b = o["ports"][order[(i + 1) % n] + ".left"]
            if (
                a["mass_flow_kg_s"] != -b["mass_flow_kg_s"]
                or a["energy_flow_w"] != -b["energy_flow_w"]
                or a["pressure_pa"] != b["pressure_pa"]
            ):
                raise ValueError("Nonconservative shared port")
            metrics["pressure_abs_pa"] = max(
                metrics["pressure_abs_pa"],
                abs(o["ports"][name + ".left"]["pressure_pa"] - faces[i]),
            )
    actual = [e for e in report["events"] if e["kind"] in ("flow_zero", "shaft_zero")]
    events.sort(key=lambda e: (e["t"], e["kind"]))
    if len(actual) != len(events):
        raise ValueError("Missing or extra shaft/flow crossing")
    for a, b in zip(actual, events):
        if a["kind"] != b["kind"] or a["direction"] != b["direction"]:
            raise ValueError("Incorrect crossing kind/direction")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(a["t"] - b["t"]))
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    restored = (ValvedPumpLoop if "valve" in spec else PumpCirculationLoop).from_snapshot(
        checkpoint
    )
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Checkpoint final observation mismatch")
    restart = restored.advance(budget["restart_extension_s"], rtol=1e-10, atol_scale=0.01)
    accepted = (
        report["numerical_acceptance"]
        and report["completed_time_s"] == spec["duration_s"]
        and all(np.isfinite(v) and v <= budget[k] for k, v in metrics.items())
    )
    return dict(
        accepted=bool(accepted),
        metrics=metrics,
        restart_energy_abs_j=abs(restart["total_energy_residual_j"]),
        events=actual,
    )


def scenarios():
    base = json.loads(Path("scenarios/ai05_displacement_pump_trip.json").read_text())
    cases = []
    for label, flow, omega, torque, commands in [
        ("trip_positive", 0.5, 20, 0.2, [(5, 0)]),
        ("trip_negative", -0.5, -20, -0.2, [(5, 0)]),
        ("reverse_positive", 0.5, 20, 0.2, [(2, -0.2)]),
        ("reverse_negative", -0.5, -20, -0.2, [(2, 0.2)]),
        ("startup_positive", 0, 0, 0.2, []),
        ("startup_negative", 0, 0, -0.2, []),
        ("backdrive_positive", 2, 0, 0, []),
        ("backdrive_negative", -2, 0, 0, []),
    ]:
        s = deepcopy(base)
        s.update(
            name=label,
            initial_omega_rad_s=omega,
            motor_torque_nm=torque,
            timeline=[dict(t=t, motor_torque_nm=q) for t, q in commands],
        )
        s["loop"]["initialization"]["mass_flow_kg_s"] = flow
        cases.append((label, s))
    for torque, label in ((0.2, "steady_positive"), (-0.2, "steady_negative"), (0, "steady_zero")):
        s = deepcopy(base)
        for n in s["loop"]["model"]["topology"]["nodes"]:
            n["parameters"].update(fluid_wall_ua_w_k=100, ambient_ua_w_k=20)
        m = steady_pump_loop(s["loop"]["model"], s["pump"], torque)
        o = m.observation()
        s.update(
            name=label,
            duration_s=100,
            step_s=2,
            initial_omega_rad_s=o["omega_rad_s"],
            motor_torque_nm=torque,
            timeline=[],
        )
        s["loop"]["initialization"].update(
            mass_flow_kg_s=o["loop"]["mass_flow_kg_s"],
            fluid_temperature_k=o["loop"]["fluid_temperature_k"],
            wall_temperature_k=o["loop"]["wall_temperature_k"],
        )
        cases.append((label, s))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget_path = Path("validation/ai05_pump_budget_v1.json")
    budget = json.loads(budget_path.read_text())
    write(args.output / "budget.json", budget)
    results = []
    for name, spec in scenarios():
        for policy in ("base", "tight"):
            s = deepcopy(spec)
            if policy == "tight":
                s["solver"].update(rtol=1e-11, atol_scale=0.01, max_step_s=0.025)
            folder = args.output / (name + "_" + policy)
            run_pump_experiment(s, folder)
            result = assess(folder, budget)
            result.update(name=name, policy=policy, folder=str(folder))
            results.append(result)
        # Accuracy on the common output grid is tested against the same independent reference in both policies.
    for name, s in scenarios()[:2]:
        s["solver"].update(method="BDF", rtol=1e-11, atol_scale=0.01)
        folder = args.output / (name + "_bdf")
        run_pump_experiment(s, folder)
        result = assess(folder, budget)
        result.update(name=name, policy="bdf", folder=str(folder))
        results.append(result)
    report = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(budget_path),
        script_sha256=sha(__file__),
        all_cases_accepted=all(r["accepted"] for r in results),
        integrations=len(results),
        case_families=11,
        cases=results,
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "verification.json", report)
    print(
        json.dumps(dict(all_cases_accepted=report["all_cases_accepted"], integrations=len(results)))
    )
    return 0 if report["all_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
