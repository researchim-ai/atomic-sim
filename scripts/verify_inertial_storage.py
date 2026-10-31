"""Independent fixed-scope acceptance for gas stores and kinetic hydraulic inertance."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.hydraulics.inertial_storage_pair import InertialStoragePair
from scripts.inertial_storage_references import lossless_half_period, reference
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.storage_experiment import run_storage_experiment


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scenarios():
    base = json.loads(Path("scenarios/ai05_inertial_storage_damping.json").read_text())
    cases = []
    for name, reverse, resistance, flow, thermal in [
        ("lossless_forward", False, 0, 0, False),
        ("lossless_reverse", True, 0, 0, False),
        ("damped_forward", False, 1000, 0, False),
        ("damped_reverse", True, 1000, 0, False),
        ("opposed_forward", False, 1000, -1, False),
        ("opposed_reverse", True, 1000, 1, False),
        ("thermal_forward", False, 1000, 0, True),
        ("thermal_reverse", True, 1000, 0, True),
    ]:
        s = deepcopy(base)
        s.update(name=name, timeline=[])
        s["model"].update(initial_mass_flow_kg_s=flow)
        s["model"]["line"]["resistance_pa_s2_kg2"] = resistance
        v = s["model"]["storage_configuration"]["vessels"]
        if reverse:
            v[0]["initial_gas_pressure_pa"], v[1]["initial_gas_pressure_pa"] = 100000, 200000
        if thermal:
            for a in v:
                a.update(gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2, liquid_heater_w=20)
        cases.append((name, s))
    s = deepcopy(base)
    s["name"] = "resistance_switch"
    cases.append((s["name"], s))
    s = deepcopy(base)
    s.update(name="equilibrium", timeline=[], duration_s=100, step_s=2)
    for v in s["model"]["storage_configuration"]["vessels"]:
        v.update(initial_gas_pressure_pa=100000, gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2)
    cases.append((s["name"], s))
    s = deepcopy(base)
    s.update(name="small_signal", timeline=[])
    s["model"]["line"]["resistance_pa_s2_kg2"] = 0
    for v, p in zip(s["model"]["storage_configuration"]["vessels"], (100000.5, 99999.5)):
        v["initial_gas_pressure_pa"] = p
    cases.append((s["name"], s))
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
    restored = InertialStoragePair.from_snapshot(checkpoint)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Inertial storage checkpoint mismatch")
    restarted = restored.advance(
        budget["restart_extension_s"], rtol=1e-11, atol_scale=0.01, max_step_s=0.02
    )
    extended = deepcopy(spec)
    extended["duration_s"] += budget["restart_extension_s"]
    at, reference_events = reference(extended)
    v = spec["model"]["storage_configuration"]["vessels"]
    names = [x["id"] for x in v]
    fluid = spec["model"]["storage_configuration"]["fluid"]
    rho, cv = fluid["density_kg_m3"], fluid["liquid_cv_j_kg_k"]
    inertia = spec["model"]["line"]["inertance_m_inv"]
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
    flow0 = spec["model"]["initial_mass_flow_kg_s"]
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
        nonlinear_period_abs_s=0.0,
        mass_abs_kg=0.0,
    )
    for row in rows + [restarted]:
        exact = at(row["time_s"])
        for key, metric in (
            ("transfer_mass_kg", "transfer_mass_abs_kg"),
            ("mass_flow_kg_s", "mass_flow_abs_kg_s"),
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
            - kinetic_rate
        )
        metrics["link_energy_abs_w"] = max(
            metrics["link_energy_abs_w"], abs(link), abs(row["link_energy_residual_w"])
        )
        if row["dissipation_w"] < 0:
            raise ValueError("Nonpassive inertial line")
    actual = [e for e in report["events"] if e["kind"] == "flow_zero"]
    expected = [e for e in reference_events if e["t"] <= spec["duration_s"]]
    if len(actual) != len(expected):
        raise ValueError("Missing or extra inertial flow crossing")
    for a, b in zip(actual, expected):
        if a["direction"] != b["direction"]:
            raise ValueError("Wrong inertial crossing direction")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(a["t"] - b["t"]))
    half_period = None
    if spec["name"] in ("lossless_forward", "lossless_reverse"):
        half_period = lossless_half_period(spec)
        for i, e in enumerate(actual):
            metrics["nonlinear_period_abs_s"] = max(
                metrics["nonlinear_period_abs_s"], abs(e["t"] - (i + 1) * half_period)
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
        lossless_half_period_s=half_period,
        restart_energy_abs_j=abs(restarted["energy_residual_j"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path("validation/ai05_inertial_storage_budget_v1.json")
    budget = json.loads(path.read_text())
    write(args.output / "budget.json", budget)
    records = []
    for name, spec in scenarios():
        for policy in ("base", "tight"):
            s = deepcopy(spec)
            if policy == "tight":
                s["solver"].update(rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
            folder = args.output / (name + "_" + policy)
            run_storage_experiment(s, folder)
            record = assess(folder, budget)
            record.update(name=name, policy=policy, folder=str(folder))
            records.append(record)
    for name, s in scenarios()[2:4]:
        s["solver"].update(method="BDF", rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
        folder = args.output / (name + "_bdf")
        run_storage_experiment(s, folder)
        record = assess(folder, budget)
        record.update(name=name, policy="bdf", folder=str(folder))
        records.append(record)
    result = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(path),
        script_sha256=sha(__file__),
        reference_sha256=sha("scripts/inertial_storage_references.py"),
        coefficient_reference_sha256=sha("scripts/storage_references.py"),
        integrations=len(records),
        case_families=11,
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
