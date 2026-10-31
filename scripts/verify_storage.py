"""Gas-storage acceptance from independent temperature equations and quadrature."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from physics.hydraulics.gas_storage_pair import GasStoragePair
from scripts.storage_references import adiabatic_at, thermal_reference
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.storage_experiment import run_storage_experiment


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scenarios():
    base = json.loads(Path("scenarios/ai05_storage_isolation.json").read_text())
    cases = []
    for name, reverse, kind, timeline, thermal in [
        ("adiabatic_forward", False, "bidirectional", False, False),
        ("adiabatic_reverse", True, "bidirectional", False, False),
        ("isolation", False, "bidirectional", True, False),
        ("check_forward", False, "check", False, False),
        ("check_reverse_blocked", True, "check", False, False),
        ("thermal_forward", False, "bidirectional", False, True),
        ("thermal_reverse", True, "bidirectional", False, True),
    ]:
        s = deepcopy(base)
        s["name"] = name
        if not timeline:
            s["timeline"] = []
        if reverse:
            (
                s["model"]["vessels"][0]["initial_gas_pressure_pa"],
                s["model"]["vessels"][1]["initial_gas_pressure_pa"],
            ) = 100000, 200000
        s["model"]["connection"].update(
            kind=kind, cracking_pressure_pa=100 if kind == "check" else 0
        )
        if thermal:
            for v in s["model"]["vessels"]:
                v.update(gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2, liquid_heater_w=20)
        cases.append((name, s))
    s = deepcopy(base)
    s.update(name="heated_check_opening", timeline=[])
    s["model"]["connection"].update(kind="check")
    for i, v in enumerate(s["model"]["vessels"]):
        v.update(
            initial_gas_pressure_pa=100000 + 5 * i,
            gas_liquid_ua_w_k=100 if i == 0 else 0,
            liquid_heater_w=100 if i == 0 else 0,
        )
    cases.append((s["name"], s))
    s = deepcopy(base)
    s.update(name="cooling_check_closure", timeline=[])
    s["model"]["connection"].update(kind="check", cracking_pressure_pa=5)
    for i, v in enumerate(s["model"]["vessels"]):
        v.update(
            initial_gas_pressure_pa=100010 if i == 0 else 100000,
            initial_gas_temperature_k=310 if i == 0 else 300,
            gas_liquid_ua_w_k=10 if i == 0 else 0,
        )
    cases.append((s["name"], s))
    s = deepcopy(base)
    s.update(name="equilibrium", timeline=[], duration_s=100, step_s=2)
    for v in s["model"]["vessels"]:
        v.update(initial_gas_pressure_pa=100000, gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2)
    cases.append((s["name"], s))
    s = deepcopy(base)
    s.update(name="isolated_heating", timeline=[], duration_s=100, step_s=2)
    s["model"]["connection"]["conductance_m3_s_pa"] = 0
    for v in s["model"]["vessels"]:
        v.update(gas_liquid_ua_w_k=10, gas_ambient_ua_w_k=2, liquid_heater_w=100)
    cases.append((s["name"], s))
    return cases


def assess(folder, budget):
    folder = Path(folder)
    spec = json.loads((folder / "input.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if manifest["source_sha256"] != _source_fingerprint():
        raise ValueError("Storage source mismatch")
    for name, h in manifest["artifacts"].items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p) != h:
            raise ValueError("Storage checksum/path mismatch")
    at, events = thermal_reference(spec)
    v = spec["model"]["vessels"]
    names = [a["id"] for a in v]
    rho = spec["model"]["fluid"]["density_kg_m3"]
    cv = spec["model"]["fluid"]["liquid_cv_j_kg_k"]
    initial_mass = np.array([a["initial_liquid_mass_kg"] for a in v])
    initial_t = np.array([a["initial_liquid_temperature_k"] for a in v])
    initial_tg = np.array([a["initial_gas_temperature_k"] for a in v])
    cg = np.array(
        [
            a["initial_gas_pressure_pa"]
            * (a["total_volume_m3"] - a["initial_liquid_mass_kg"] / rho)
            / ((a["gas_gamma"] - 1) * a["initial_gas_temperature_k"])
            for a in v
        ]
    )
    metrics = dict(
        transfer_mass_abs_kg=0.0,
        pressure_abs_pa=0.0,
        temperature_abs_k=0.0,
        boundary_work_abs_j=0.0,
        energy_abs_j=0.0,
        local_energy_abs_w=0.0,
        event_time_abs_s=0.0,
        mass_abs_kg=0.0,
    )
    for row in rows:
        ref = at(row["time_s"])
        metrics["transfer_mass_abs_kg"] = max(
            metrics["transfer_mass_abs_kg"], abs(row["transfer_mass_kg"] - ref["transfer_mass_kg"])
        )
        metrics["pressure_abs_pa"] = max(
            metrics["pressure_abs_pa"],
            float(max(abs(np.array([row["pressure_pa"][n] for n in names]) - ref["pressure"]))),
        )
        for key, rkey in (
            ("liquid_temperature_k", "liquid_temperature"),
            ("gas_temperature_k", "gas_temperature"),
        ):
            metrics["temperature_abs_k"] = max(
                metrics["temperature_abs_k"],
                float(max(abs(np.array([row[key][n] for n in names]) - ref[rkey]))),
            )
        metrics["boundary_work_abs_j"] = max(
            metrics["boundary_work_abs_j"],
            float(
                max(
                    abs(
                        np.array([row["gas_boundary_work_j"][n] for n in names])
                        - ref["gas_boundary_work"]
                    )
                )
            ),
            abs(row["external_heat_j"] - ref["external_heat_j"]),
        )
        mass = np.array([row["liquid_mass_kg"][n] for n in names])
        tl = np.array([row["liquid_temperature_k"][n] for n in names])
        tg = np.array([row["gas_temperature_k"][n] for n in names])
        # Independent total-energy reconstruction uses saved absolute temperatures.
        energy = float(
            np.sum(cv * ((mass - initial_mass) * initial_t + mass * (tl - initial_t)))
            + cg @ (tg - initial_tg)
            - row["external_heat_j"]
        )
        metrics["energy_abs_j"] = max(
            metrics["energy_abs_j"], abs(energy), abs(row["energy_residual_j"])
        )
        metrics["local_energy_abs_w"] = max(
            metrics["local_energy_abs_w"],
            max(abs(x) for x in row["local_energy_residual_w"].values()),
        )
        metrics["mass_abs_kg"] = max(metrics["mass_abs_kg"], abs(sum(mass) - sum(initial_mass)))
        if row["dissipation_w"] < 0:
            raise ValueError("Nonpassive connection")
        for name, end in zip(names, ("left", "right")):
            a, b = row["ports"][name + ".transfer"], row["ports"]["connection." + end]
            if (
                a["pressure_pa"] != b["pressure_pa"]
                or a["mass_flow_kg_s"] != -b["mass_flow_kg_s"]
                or a["energy_flow_w"] != -b["energy_flow_w"]
            ):
                raise ValueError("Nonconservative storage port")
    actual = [e for e in report["events"] if e["kind"] == "check_threshold"]
    if len(actual) != len(events):
        raise ValueError("Missing or extra check event")
    for a, b in zip(actual, events):
        if a["direction"] != b["direction"]:
            raise ValueError("Wrong check event direction")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(a["t"] - b["t"]))
    if spec["name"] == "heated_check_opening":
        cl = initial_mass[0] * cv
        capacity = cg[0]
        ua = v[0]["gas_liquid_ua_w_k"]
        tau = cl * capacity / (ua * (cl + capacity))
        p0 = v[0]["initial_gas_pressure_pa"]
        p1 = v[1]["initial_gas_pressure_pa"]
        exact = brentq(
            lambda t: (
                p0
                * (
                    1
                    + v[0]["liquid_heater_w"]
                    / (cl + capacity)
                    * (t - tau * (1 - np.exp(-t / tau)))
                    / initial_tg[0]
                )
                - p1
            ),
            0,
            spec["duration_s"],
        )
        if len(actual) != 1:
            raise ValueError("Expected one analytic opening")
        metrics["event_time_abs_s"] = max(metrics["event_time_abs_s"], abs(actual[0]["t"] - exact))
    quadrature = []
    if not any(
        a[k] for a in v for k in ("gas_liquid_ua_w_k", "gas_ambient_ua_w_k", "liquid_heater_w")
    ):
        for t in (0.1, 1, 3):
            q = adiabatic_at(spec, t)
            ref = at(t)
            if (
                abs(q["transfer_mass_kg"] - ref["transfer_mass_kg"])
                > budget["transfer_mass_abs_kg"]
                or max(abs(q["pressure"] - ref["pressure"])) > budget["pressure_abs_pa"]
            ):
                raise ValueError("Quadrature and temperature reference disagree")
            quadrature.append(
                dict(t=t, transfer_mass_abs_kg=abs(q["transfer_mass_kg"] - ref["transfer_mass_kg"]))
            )
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    restored = GasStoragePair.from_snapshot(checkpoint)
    if restored.observation() != {k: value for k, value in rows[-1].items() if k != "phase"}:
        raise ValueError("Storage checkpoint final observation mismatch")
    restart = restored.advance(budget["restart_extension_s"], rtol=1e-11, atol_scale=0.01)
    accepted = (
        report["numerical_acceptance"]
        and report["completed_time_s"] == spec["duration_s"]
        and all(np.isfinite(x) and x <= budget[k] for k, x in metrics.items())
    )
    return dict(
        accepted=bool(accepted),
        metrics=metrics,
        events=actual,
        quadrature=quadrature,
        restart_energy_abs_j=abs(restart["energy_residual_j"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = Path("validation/ai05_storage_budget_v1.json")
    budget = json.loads(path.read_text())
    write(args.output / "budget.json", budget)
    records = []
    for name, spec in scenarios():
        for policy in ("base", "tight"):
            s = deepcopy(spec)
            if policy == "tight":
                s["solver"].update(rtol=1e-11, atol_scale=0.01, max_step_s=0.025)
            folder = args.output / (name + "_" + policy)
            run_storage_experiment(s, folder)
            result = assess(folder, budget)
            result.update(name=name, policy=policy, folder=str(folder))
            records.append(result)
    for name, s in scenarios()[:2]:
        s["solver"].update(method="BDF", rtol=1e-12, max_step_s=0.01, atol_scale=0.001)
        folder = args.output / (name + "_bdf")
        run_storage_experiment(s, folder)
        result = assess(folder, budget)
        result.update(name=name, policy="bdf", folder=str(folder))
        records.append(result)
    result = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(path),
        script_sha256=sha(__file__),
        reference_sha256=sha("scripts/storage_references.py"),
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
