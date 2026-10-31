"""Fixed finite-leakage valve cases with independent reference and time refinement."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from physics.hydraulics.pump_loop import steady_pump_loop
from scripts.verify_pump import assess, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.pump_experiment import run_pump_experiment


def scenarios():
    base = json.loads(Path("scenarios/ai05_valve_close_reopen.json").read_text())
    cases = []
    for label, sign, tau, reverse in [
        ("cycle_positive", 1, 0.5, False),
        ("cycle_negative", -1, 0.5, False),
        ("fast_actuator", 1, 0.02, False),
        ("slow_actuator", 1, 5, False),
        ("reverse_positive", 1, 0.5, True),
        ("reverse_negative", -1, 0.5, True),
    ]:
        s = deepcopy(base)
        s["name"] = label
        s["valve"]["actuator_time_s"] = tau
        s["initial_omega_rad_s"] *= sign
        s["motor_torque_nm"] *= sign
        s["loop"]["initialization"]["mass_flow_kg_s"] *= sign
        if reverse:
            s["timeline"][-1]["motor_torque_nm"] = -0.2 * sign
        cases.append((label, s))
    for opening, label, torque in [
        (0, "closed_leaking_positive", 0.2),
        (0, "closed_leaking_negative", -0.2),
        (1, "open_positive", 0.2),
    ]:
        s = deepcopy(base)
        s.update(name=label, duration_s=100, step_s=2, timeline=[], motor_torque_nm=torque)
        s["valve"]["initial_opening"] = opening
        for n in s["loop"]["model"]["topology"]["nodes"]:
            n["parameters"].update(fluid_wall_ua_w_k=100, ambient_ua_w_k=20)
        effective = deepcopy(s["loop"]["model"])
        for n in effective["topology"]["nodes"]:
            if n["id"] == s["valve"]["segment"]:
                n["parameters"]["loss_coefficient"] += (
                    s["valve"]["open_loss_coefficient"]
                    + (s["valve"]["closed_loss_coefficient"] - s["valve"]["open_loss_coefficient"])
                    * (1 - opening) ** 2
                )
        m = steady_pump_loop(effective, s["pump"], torque)
        o = m.observation()
        s["initial_omega_rad_s"] = o["omega_rad_s"]
        s["loop"]["initialization"].update(
            mass_flow_kg_s=o["loop"]["mass_flow_kg_s"],
            fluid_temperature_k=o["loop"]["fluid_temperature_k"],
            wall_temperature_k=o["loop"]["wall_temperature_k"],
        )
        cases.append((label, s))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget_path = Path("validation/ai05_valve_budget_v1.json")
    budget = json.loads(budget_path.read_text())
    write(args.output / "budget.json", budget)
    records = []
    for name, s in scenarios():
        for policy in ("base", "tight"):
            spec = deepcopy(s)
            if policy == "tight":
                spec["solver"].update(rtol=1e-11, atol_scale=0.01, max_step_s=0.025)
            folder = args.output / (name + "_" + policy)
            run_pump_experiment(spec, folder)
            record = assess(folder, budget)
            record.update(name=name, policy=policy, folder=str(folder))
            records.append(record)
    for name, s in scenarios()[:2]:
        s["solver"].update(method="BDF", rtol=1e-11, atol_scale=0.01)
        folder = args.output / (name + "_bdf")
        run_pump_experiment(s, folder)
        record = assess(folder, budget)
        record.update(name=name, policy="bdf", folder=str(folder))
        records.append(record)
    result = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(budget_path),
        script_sha256=sha(__file__),
        reference_sha256=sha("scripts/verify_pump.py"),
        integrations=len(records),
        case_families=9,
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
