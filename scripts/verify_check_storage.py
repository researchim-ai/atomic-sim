"""Independent fixed-scope acceptance for pump, shaft and inertial gas storage."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.hydraulics.check_pump_storage import CheckPumpStorage
from scripts.check_storage_references import reference
from sdk.check_storage_experiment import run_check_storage_experiment
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scenarios():
    base = json.loads(Path("scenarios/ai05_check_valve_restart.json").read_text())
    cases = []
    cases.append(("restart", deepcopy(base)))
    cases[-1][1]["name"] = "restart"
    for name, left, right, omega, torque, flow, crack, thermal, duration in [
        ("open_coast", 200000, 100000, 20, 0.2, 0, 100, False, 20),
        ("positive_initial_flow", 99000, 100000, 0, 0, 1, 100, False, 20),
        ("blocked_backpressure", 100000, 200000, 0, 0, 0, 100, False, 20),
        ("blocked_shaft_decay", 100000, 200000, 20, 0, 0, 100, False, 20),
        ("analytic_release", 99000, 100000, 0, 1, 0, 100, False, 20),
        ("braking_after_seat", 200000, 100000, 20, -0.2, 0.5, 100, False, 20),
        ("thermal_closed", 100000, 200000, 0, 0, 0, 100, True, 20),
        ("high_cracking", 100050, 100000, 0, 1, 0, 5000, False, 20),
        ("equilibrium_zero", 100000, 100000, 0, 0, 0, 0, True, 100),
        ("equilibrium_threshold", 100100, 100000, 0, 0, 0, 100, False, 100),
    ]:
        spec = deepcopy(base)
        spec.update(
            name=name, timeline=[], duration_s=duration, step_s=2 if duration == 100 else 0.5
        )
        spec["model"]["cracking_pressure_pa"] = crack
        plant = spec["model"]["plant_configuration"]
        plant.update(initial_omega_rad_s=omega, motor_torque_nm=torque)
        plant["hydraulic_configuration"]["initial_mass_flow_kg_s"] = flow
        vessels = plant["hydraulic_configuration"]["storage_configuration"]["vessels"]
        for vessel, pressure in zip(vessels, (left, right)):
            vessel.update(
                initial_gas_pressure_pa=pressure,
                gas_liquid_ua_w_k=10 if thermal else 0,
                gas_ambient_ua_w_k=2 if thermal else 0,
                liquid_heater_w=20 if name == "thermal_closed" else 0,
            )
        cases.append((name, spec))
    for name, opening in (("initial_release", True), ("initial_seat", False)):
        spec = deepcopy(base)
        spec.update(name=name, timeline=[], duration_s=1, step_s=0.1)
        plant = spec["model"]["plant_configuration"]
        if opening:
            spec["model"]["cracking_pressure_pa"] = 0
            plant.update(initial_omega_rad_s=1e-8, motor_torque_nm=1)
            for vessel in plant["hydraulic_configuration"]["storage_configuration"]["vessels"]:
                vessel["initial_gas_pressure_pa"] = 100000
        else:
            plant["hydraulic_configuration"]["initial_mass_flow_kg_s"] = 1e-15
            plant["motor_torque_nm"] = 0
        cases.append((name, spec))
    return cases


def assess(folder, budget):
    folder = Path(folder)
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["source_sha256"] != _source_fingerprint():
        raise ValueError("Inertial storage source mismatch")
    for name, h in manifest["artifacts"].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p) != h:
            raise ValueError("Inertial storage bundle checksum/path mismatch")
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    restored = CheckPumpStorage.from_snapshot(checkpoint)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Inertial storage checkpoint mismatch")
    restarted = restored.advance(
        budget["restart_extension_s"], rtol=1e-11, atol_scale=0.01, max_step_s=0.02
    )
    extended = deepcopy(spec)
    extended["duration_s"] += budget["restart_extension_s"]
    at, reference_events = reference(extended)
    plant = spec["model"]["plant_configuration"]
    v = plant["hydraulic_configuration"]["storage_configuration"]["vessels"]
    names = [x["id"] for x in v]
    fluid = plant["hydraulic_configuration"]["storage_configuration"]["fluid"]
    rho, cv = fluid["density_kg_m3"], fluid["liquid_cv_j_kg_k"]
    inertia = plant["hydraulic_configuration"]["line"]["inertance_m_inv"]
    m0 = np.array([x["initial_liquid_mass_kg"] for x in v])
    tl0 = np.array([x["initial_liquid_temperature_k"] for x in v])
    tg0 = np.array([x["initial_gas_temperature_k"] for x in v])
    cg = np.array(
        [
            x["initial_gas_pressure_pa"]
            * (x["total_volume_m3"] - x["initial_liquid_mass_kg"] / rho)
            / ((x["gas_gamma"] - 1) * x["initial_gas_temperature_k"])
            for x in v
        ]
    )
    flow0 = plant["hydraulic_configuration"]["initial_mass_flow_kg_s"]
    metrics = dict(
        transfer_mass_abs_kg=0.0,
        mass_flow_abs_kg_s=0.0,
        pressure_abs_pa=0.0,
        temperature_abs_k=0.0,
        boundary_work_abs_j=0.0,
        energy_abs_j=0.0,
        local_energy_abs_w=0.0,
        link_energy_abs_w=0.0,
        event_time_abs_s=0.0,
        omega_abs_rad_s=0.0,
        shaft_energy_abs_j=0.0,
        power_transfer_abs_w=0.0,
        mass_abs_kg=0.0,
        reverse_flow_kg_s=0.0,
        closed_flow_kg_s=0.0,
        constraint_residual_pa=0.0,
        root_flow_abs_kg_s=0.0,
        root_projection_j=0.0,
        root_pressure_abs_pa=0.0,
        phase_mismatches=0,
    )
    pump = plant["pump"]
    omega0 = plant["initial_omega_rad_s"]
    for row in rows + [restarted]:
        exact = at(row["time_s"])
        metrics["reverse_flow_kg_s"] = max(metrics["reverse_flow_kg_s"], -row["mass_flow_kg_s"])
        if row["check_mode"] == "closed":
            metrics["closed_flow_kg_s"] = max(
                metrics["closed_flow_kg_s"], abs(row["mass_flow_kg_s"])
            )
        metrics["constraint_residual_pa"] = max(
            metrics["constraint_residual_pa"], row["check_constraint_residual_pa"]
        )
        metrics["phase_mismatches"] += int(row["check_mode"] != exact["check_mode"])
        for key, metric in (
            ("transfer_mass_kg", "transfer_mass_abs_kg"),
            ("mass_flow_kg_s", "mass_flow_abs_kg_s"),
            ("omega_rad_s", "omega_abs_rad_s"),
        ):
            metrics[metric] = max(metrics[metric], abs(row[key] - exact[key]))
        for key, refkey, metric in (
            ("pressure_pa", "pressure", "pressure_abs_pa"),
            ("liquid_temperature_k", "liquid_temperature", "temperature_abs_k"),
            ("gas_temperature_k", "gas_temperature", "temperature_abs_k"),
            ("gas_boundary_work_j", "gas_boundary_work", "boundary_work_abs_j"),
        ):
            metrics[metric] = max(
                metrics[metric],
                float(max(abs(np.array([row[key][n] for n in names]) - exact[refkey]))),
            )
        mass = np.array([row["liquid_mass_kg"][n] for n in names])
        tl = np.array([row["liquid_temperature_k"][n] for n in names])
        tg = np.array([row["gas_temperature_k"][n] for n in names])
        flow = row["mass_flow_kg_s"]
        energy = float(
            sum(cv * ((mass - m0) * tl0 + mass * (tl - tl0)))
            + cg @ (tg - tg0)
            + inertia * (flow - flow0) * (flow + flow0) / (2 * rho)
            + 0.5
            * pump["inertia_kg_m2"]
            * (row["omega_rad_s"] - omega0)
            * (row["omega_rad_s"] + omega0)
            - row["motor_work_j"]
            + row["bearing_heat_j"]
            - row["external_heat_j"]
        )
        metrics["energy_abs_j"] = max(
            metrics["energy_abs_j"], abs(energy), abs(row["energy_residual_j"])
        )
        metrics["mass_abs_kg"] = max(metrics["mass_abs_kg"], abs(sum(mass) - sum(m0)))
        metrics["local_energy_abs_w"] = max(
            metrics["local_energy_abs_w"],
            max(abs(x) for x in row["local_energy_residual_w"].values()),
        )
        metrics["boundary_work_abs_j"] = max(
            metrics["boundary_work_abs_j"], abs(row["external_heat_j"] - exact["external_heat_j"])
        )
        for name, end in zip(names, ("left", "right")):
            a, b = row["ports"][name + ".transfer"], row["ports"]["connection." + end]
            if (
                a["pressure_pa"] != b["pressure_pa"]
                or a["mass_flow_kg_s"] != -b["mass_flow_kg_s"]
                or a["energy_flow_w"] != -b["energy_flow_w"]
            ):
                raise ValueError("Nonconservative inertial storage port")
        kinetic_rate = inertia * flow * row["flow_acceleration_kg_s2"] / rho
        link = (
            sum(row["ports"]["connection." + end]["energy_flow_w"] for end in ("left", "right"))
            + row["shaft_to_pump_power_w"]
            - row["leakage_power_w"]
            - kinetic_rate
        )
        metrics["link_energy_abs_w"] = max(
            metrics["link_energy_abs_w"], abs(link), abs(row["link_energy_residual_w"])
        )
        for key in ("motor_work_j", "bearing_heat_j", "leakage_heat_j", "head_work_j"):
            metrics["boundary_work_abs_j"] = max(
                metrics["boundary_work_abs_j"], abs(row[key] - exact[key])
            )
        shaft_change = (
            0.5
            * pump["inertia_kg_m2"]
            * (row["omega_rad_s"] - omega0)
            * (row["omega_rad_s"] + omega0)
        )
        shaft_defect = (
            shaft_change
            - row["motor_work_j"]
            + row["bearing_heat_j"]
            + row["leakage_heat_j"]
            + row["head_work_j"]
        )
        metrics["shaft_energy_abs_j"] = max(
            metrics["shaft_energy_abs_j"], abs(shaft_defect), abs(row["shaft_energy_residual_j"])
        )
        power = (
            pump["displacement_m3_rad"] * row["head_pa"] * row["omega_rad_s"]
            - row["head_pa"] * flow / rho
            - pump["leakage_m3_s_pa"] * row["head_pa"] ** 2
        )
        metrics["power_transfer_abs_w"] = max(metrics["power_transfer_abs_w"], abs(power))
        thermal_a = row["ports"]["pump.leak_heat"]
        thermal_b = row["ports"][pump["leak_heat_vessel"] + ".pump_heat"]
        mechanical_a = row["ports"]["shaft.to_hydraulics"]
        mechanical_b = row["ports"]["hydraulics.shaft"]
        if (
            thermal_a["temperature_k"] != thermal_b["temperature_k"]
            or thermal_a["heat_flow_w"] != -thermal_b["heat_flow_w"]
            or mechanical_a["omega_rad_s"] != mechanical_b["omega_rad_s"]
            or mechanical_a["torque_nm"] != -mechanical_b["torque_nm"]
        ):
            raise ValueError("Nonconservative thermal/mechanical pump port")
        if row["dissipation_w"] < 0:
            raise ValueError("Nonpassive inertial line")
    actual = [
        e for e in report["events"] if e["kind"] in ("check_opened", "check_closed", "shaft_zero")
    ]
    expected = [e for e in reference_events if e["t"] <= spec["duration_s"]]
    if len(actual) != len(expected):
        raise ValueError("Missing or extra inertial flow crossing")
    for a, b in zip(actual, expected):
        if a["kind"] != b["kind"] or (
            a["kind"] == "shaft_zero" and a["direction"] != b["direction"]
        ):
            raise ValueError("Wrong inertial crossing direction")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(a["t"] - b["t"]))
    for event in actual:
        if event["kind"] == "check_closed":
            metrics["root_flow_abs_kg_s"] = max(
                metrics["root_flow_abs_kg_s"], abs(event["flow_residual_kg_s"])
            )
            metrics["root_projection_j"] = max(
                metrics["root_projection_j"], event["kinetic_projection_j"]
            )
        if event["kind"] == "check_opened":
            metrics["root_pressure_abs_pa"] = max(
                metrics["root_pressure_abs_pa"],
                abs(event["drive_pa"] - budget["opening_resolution_pa"]),
            )
    if spec["name"] == "analytic_release":
        D = pump["displacement_m3_rad"]
        C = pump["leakage_m3_s_pa"]
        B = pump["drag_nm_s_rad"]
        J = pump["inertia_kg_m2"]
        decay = (D * D / C + B) / J
        equilibrium = plant["motor_torque_nm"] / (D * D / C + B)
        threshold = (
            (
                spec["model"]["cracking_pressure_pa"]
                + v[1]["initial_gas_pressure_pa"]
                - v[0]["initial_gas_pressure_pa"]
                + budget["opening_resolution_pa"]
            )
            * C
            / D
        )
        analytic = -np.log((threshold - equilibrium) / (omega0 - equilibrium)) / decay
        opening = next(e for e in actual if e["kind"] == "check_opened")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(opening["t"] - analytic))
    if spec["name"] == "blocked_shaft_decay":
        D = pump["displacement_m3_rad"]
        C = pump["leakage_m3_s_pa"]
        B = pump["drag_nm_s_rad"]
        J = pump["inertia_kg_m2"]
        decay = (D * D / C + B) / J
        for row in rows:
            factor = omega0**2 * (1 - np.exp(-2 * decay * row["time_s"])) / (2 * decay)
            metrics["omega_abs_rad_s"] = max(
                metrics["omega_abs_rad_s"],
                abs(row["omega_rad_s"] - omega0 * np.exp(-decay * row["time_s"])),
            )
            metrics["boundary_work_abs_j"] = max(
                metrics["boundary_work_abs_j"],
                abs(row["bearing_heat_j"] - B * factor),
                abs(row["leakage_heat_j"] - D * D / C * factor),
            )
    accepted = (
        report["numerical_acceptance"]
        and report["completed_time_s"] == spec["duration_s"]
        and all(np.isfinite(x) and x <= budget[k] for k, x in metrics.items())
    )
    return dict(
        accepted=bool(accepted),
        metrics=metrics,
        events=actual,
        restart_energy_abs_j=abs(restarted["energy_residual_j"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path("validation/ai05_check_storage_budget_v1.json")
    budget = json.loads(path.read_text())
    write(args.output / "budget.json", budget)
    records = []
    for name, spec in scenarios():
        for policy in ("base", "tight"):
            s = deepcopy(spec)
            if policy == "tight":
                s["solver"].update(rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
            folder = args.output / (name + "_" + policy)
            run_check_storage_experiment(s, folder)
            record = assess(folder, budget)
            record.update(name=name, policy=policy, folder=str(folder))
            records.append(record)
    for name, s in scenarios()[:2]:
        s["solver"].update(method="BDF", rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
        folder = args.output / (name + "_bdf")
        run_check_storage_experiment(s, folder)
        record = assess(folder, budget)
        record.update(name=name, policy="bdf", folder=str(folder))
        records.append(record)
    result = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(path),
        script_sha256=sha(__file__),
        reference_sha256=sha("scripts/check_storage_references.py"),
        coefficient_reference_sha256=sha("scripts/storage_references.py"),
        integrations=len(records),
        case_families=13,
        all_cases_accepted=all(r["accepted"] for r in records),
        cases=records,
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "verification.json", result)
    print(
        json.dumps(dict(all_cases_accepted=result["all_cases_accepted"], integrations=len(records)))
    )
    return 0 if result["all_cases_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
