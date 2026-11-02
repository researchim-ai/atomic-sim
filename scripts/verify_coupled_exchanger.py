"""Fixed-scope independent acceptance of the monolithic two-cycle exchanger."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.thermal.coupled_pump_exchanger import CoupledPumpExchanger
from scripts.coupled_exchanger_references import reference
from sdk.coupled_exchanger_experiment import prepare, run_coupled_exchanger_experiment
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write

BUDGET = Path("validation/ai05_coupled_exchanger_budget_v1.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError("Nonfinite coupled evidence value")


def scenarios():
    base = json.loads(Path("scenarios/ai05_coupled_pump_exchanger.json").read_text())
    result = []
    for name in (
        "base",
        "trip_reverse",
        "steady",
        "reverse_a",
        "reverse_b",
        "both_reverse",
        "both_stalled",
        "one_stalled",
        "zero_exchange",
        "shared_ambient",
        "heterogeneous",
        "cocurrent_mapping",
        "initial_command",
    ):
        spec = deepcopy(base)
        spec.update(name=name, timeline=[])
        if name == "trip_reverse":
            spec["timeline"] = deepcopy(base["timeline"])
        elif name == "steady":
            spec["initialization"] = {"mode": "steady"}
        elif name in ("reverse_a", "reverse_b", "both_reverse"):
            for side in ("a", "b"):
                if name == f"reverse_{side}" or name == "both_reverse":
                    raw = spec["loops"][side]
                    raw["loop"]["initialization"]["mass_flow_kg_s"] *= -1
                    raw["initial_omega_rad_s"] *= -1
                    raw["motor_torque_nm"] *= -1
        elif name in ("both_stalled", "one_stalled"):
            for side in ("a", "b") if name == "both_stalled" else ("a",):
                raw = spec["loops"][side]
                raw.update(initial_omega_rad_s=0, motor_torque_nm=0)
                raw["loop"]["initialization"]["mass_flow_kg_s"] = 0
        elif name == "zero_exchange":
            for cell in spec["exchanger"]["cells"]:
                cell.update(a_ua_w_k=0, b_ua_w_k=0)
        elif name == "shared_ambient":
            for cell in spec["exchanger"]["cells"]:
                cell.update(ambient_ua_w_k=10, wall_power_w=100)
        elif name == "heterogeneous":
            for index, cell in enumerate(spec["exchanger"]["cells"]):
                cell.update(
                    wall_capacity_j_k=2000 * (index + 1),
                    a_ua_w_k=100 * (index + 1),
                    b_ua_w_k=200 * (index + 1),
                )
        elif name == "cocurrent_mapping":
            for index, cell in enumerate(spec["exchanger"]["cells"]):
                cell["b_segment"] = f"s{index}"
        elif name == "initial_command":
            spec["timeline"] = [
                dict(t=0, controls=dict(a_motor_torque_nm=0, b_motor_torque_nm=0.2))
            ]
        result.append((name, spec))
    return result


def matrix():
    for index, (name, spec) in enumerate(scenarios()):
        for policy in ("base", "tight", "bdf") if index < 2 else ("base", "tight"):
            candidate = deepcopy(spec)
            if policy != "base":
                candidate["solver"].update(rtol=1e-11, max_step_s=0.025, atol_scale=0.01)
            if policy == "bdf":
                candidate["solver"]["method"] = "BDF"
            yield name, policy, prepare(candidate)[0]


def assess(folder, budget):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    required = {
        "input.json",
        "initialization.json",
        "trajectory.json",
        "checkpoint.json",
        "report.json",
    }
    if manifest["source_sha256"] != _source_fingerprint() or not required.issubset(
        manifest["artifacts"]
    ):
        raise ValueError("Coupled evidence source/artifact mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError("Coupled artifact checksum/path mismatch")
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    require_finite(rows)
    require_finite(report)
    initial_controls = {
        f"{side}_motor_torque_nm": spec["loops"][side]["motor_torque_nm"] for side in ("a", "b")
    }
    expected_rows = [(0.0, "initial", initial_controls)]
    expected_events = []
    command_index, active = 0, initial_controls
    for index in range(1, math.ceil(spec["duration_s"] / spec["step_s"]) + 1):
        end = min(spec["duration_s"], index * spec["step_s"])
        while command_index < len(spec["timeline"]) and spec["timeline"][command_index]["t"] <= end:
            event = spec["timeline"][command_index]
            expected_rows.append((event["t"], "before_control", active))
            expected_events.append(dict(t=event["t"], before=active, after=event["controls"]))
            active = event["controls"]
            expected_rows.append((event["t"], "after_control", active))
            command_index += 1
        expected_rows.append((end, "output", active))
    if [(r["time_s"], r["phase"], r["controls"]) for r in rows] != expected_rows or report[
        "events"
    ] != expected_events:
        raise ValueError("Incomplete coupled trajectory/control history")
    if report["completed_time_s"] != spec["duration_s"]:
        raise ValueError("Incomplete coupled horizon")
    at, coeffs, wstart = reference(spec)
    metrics = dict(
        temperature_absolute_k=0.0,
        flow_absolute_kg_s=0.0,
        omega_absolute_rad_s=0.0,
        energy_absolute_j=0.0,
        reference_energy_absolute_j=0.0,
        heat_integral_absolute_j=0.0,
        power_residual_absolute_w=0.0,
        restart_temperature_absolute_k=0.0,
    )
    for row in rows:
        ref = at(row["time_s"])
        for side, data in zip(("a", "b"), coeffs):
            n, o = data["n"], data["offset"]
            observed = row["loops"][side]
            tf = np.array([observed["loop"]["fluid_temperature_k"][name] for name in data["names"]])
            tw = np.array([observed["loop"]["wall_temperature_k"][name] for name in data["names"]])
            metrics["temperature_absolute_k"] = max(
                metrics["temperature_absolute_k"],
                float(max(abs(tf - ref[o + 2 : o + 2 + n]))),
                float(max(abs(tw - ref[o + 2 + n : o + 2 + 2 * n]))),
            )
            metrics["flow_absolute_kg_s"] = max(
                metrics["flow_absolute_kg_s"], abs(observed["loop"]["mass_flow_kg_s"] - ref[o])
            )
            metrics["omega_absolute_rad_s"] = max(
                metrics["omega_absolute_rad_s"], abs(observed["omega_rad_s"] - ref[o + 1])
            )
            local_energy = (
                observed["loop"]["stored_energy_change_j"] + observed["shaft_energy_change_j"]
            )
            motor, bearing, leak, head, external = ref[o + 2 + 2 * n : o + 2 + 2 * n + 5]
            coupling = ref[-3 if side == "a" else -2]
            metrics["reference_energy_absolute_j"] = max(
                metrics["reference_energy_absolute_j"],
                abs(local_energy - motor + bearing - external - coupling),
            )
            metrics["energy_absolute_j"] = max(
                metrics["energy_absolute_j"],
                abs(observed["total_energy_residual_j"]),
                abs(observed["shaft_energy_residual_j"]),
            )
            metrics["heat_integral_absolute_j"] = max(
                metrics["heat_integral_absolute_j"],
                abs(row["coupling_heat_into_loops_j"][side] - coupling),
            )
        nwall = len(row["wall_temperature_k"])
        metrics["temperature_absolute_k"] = max(
            metrics["temperature_absolute_k"],
            float(max(abs(np.array(row["wall_temperature_k"]) - ref[wstart : wstart + nwall]))),
        )
        metrics["energy_absolute_j"] = max(
            metrics["energy_absolute_j"],
            abs(row["total_energy_residual_j"]),
            abs(row["shared_energy_residual_j"]),
            max(abs(v) for v in row["wall_energy_residual_j"]),
        )
        metrics["reference_energy_absolute_j"] = max(
            metrics["reference_energy_absolute_j"],
            abs(row["shared_wall_energy_change_j"] + ref[-3] + ref[-2] - ref[-1]),
        )
        metrics["heat_integral_absolute_j"] = max(
            metrics["heat_integral_absolute_j"], abs(row["wall_external_heat_j"] - ref[-1])
        )
        metrics["power_residual_absolute_w"] = max(
            metrics["power_residual_absolute_w"],
            abs(row["reciprocal_heat_residual_w"]),
            max(abs(v) for v in row["loop_power_residual_w"].values()),
        )
    snapshot = json.loads((folder / "checkpoint.json").read_text())
    restored = CoupledPumpExchanger.from_snapshot(snapshot)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Coupled checkpoint differs from trajectory")
    extended = deepcopy(spec)
    extended["duration_s"] += 1
    ref, _, _ = reference(extended)
    extended_ref = ref(extended["duration_s"])
    continued = restored.advance(1, rtol=1e-11, max_step_s=0.025, atol_scale=0.01)
    errors = [
        max(
            abs(
                np.array(continued["wall_temperature_k"])
                - extended_ref[wstart : wstart + len(continued["wall_temperature_k"])]
            )
        )
    ]
    for side, data in zip(("a", "b"), coeffs):
        n, o = data["n"], data["offset"]
        actual = [
            continued["loops"][side]["loop"]["fluid_temperature_k"][name] for name in data["names"]
        ]
        errors.append(max(abs(np.array(actual) - extended_ref[o + 2 : o + 2 + n])))
    metrics["restart_temperature_absolute_k"] = float(max(errors))
    accepted = (
        report["status"] == "completed"
        and report["numerical_acceptance"]
        and all(np.isfinite(v) and v <= budget[k] for k, v in metrics.items())
    )
    return dict(accepted=bool(accepted), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(BUDGET.read_text())
    write(args.output / "budget.json", budget)
    records = []
    for name, policy, spec in matrix():
        folder = args.output / f"{name}_{policy}"
        run_coupled_exchanger_experiment(spec, folder)
        record = assess(folder, budget)
        record.update(name=name, policy=policy, folder=str(folder))
        records.append(record)
    passed = all(r["accepted"] for r in records)
    write(
        args.output / "verification.json",
        dict(
            schema_version=1,
            source_sha256=_source_fingerprint(),
            script_sha256=sha(__file__),
            reference_sha256=sha("scripts/coupled_exchanger_references.py"),
            budget_sha256=sha(BUDGET),
            cases=records,
            integrations=len(records),
            case_families=len(scenarios()),
            all_cases_accepted=passed,
            validated=False,
            plant_referenced=False,
        ),
    )
    print(json.dumps(dict(passed=passed, integrations=len(records))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
