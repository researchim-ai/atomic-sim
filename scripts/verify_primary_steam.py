"""Reconstruct primary/steam artifacts and compare independent absolute trajectories."""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.solver_trace import summarize_solver_trace
from physics.power.primary_steam import PrimarySteamCycle
from scripts.primary_steam_references import (
    REFERENCE_POLICY,
    energy_balances,
    observables,
    power_balance,
    recorded_state,
    reference,
)
from scripts.verify_turbine import require_finite, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.primary_steam_experiment import prepare

BUDGET = Path("validation/ai05_primary_steam_budget_v3.json")


def metric_key(path, unit=None):
    joined = ".".join(path)
    if unit == "1" or "quality" in joined:
        return "quality_absolute"
    if "_j_kg_k" in joined:
        return "specific_entropy_absolute_j_kg_k"
    if "_j_kg" in joined:
        return "specific_energy_absolute_j_kg"
    if "_m3" in joined:
        return "volume_absolute_m3"
    if "_kg_s2" in joined:
        return "flow_acceleration_absolute_kg_s2"
    if "_kg_s" in joined:
        return "primary_flow_absolute_kg_s"
    if "_kg" in joined:
        return "mass_absolute_kg"
    if unit == "Pa" or "_pa" in joined:
        return "pressure_absolute_pa"
    if unit == "rad/s" or "_rad_s" in joined:
        return "primary_omega_absolute_rad_s" if "primary" in path else "steam_omega_absolute_rad_s"
    if "_nm" in joined:
        return "torque_absolute_nm"
    if unit == "K" or "_k" in joined:
        return "temperature_absolute_k"
    if "_w" in joined:
        return "power_absolute_w"
    if "_j" in joined:
        return "energy_integral_absolute_j"
    raise ValueError("Unclassified numeric observation field: " + joined)


def compare_tree(actual, fresh, metrics, path=(), unit=None):
    if isinstance(fresh, dict):
        if not isinstance(actual, dict) or set(actual) != set(fresh):
            raise ValueError("Observation schema substitution")
        for key, value in fresh.items():
            compare_tree(
                actual[key],
                value,
                metrics,
                (*path, key),
                fresh.get("unit") if key == "value" else unit,
            )
    elif isinstance(fresh, (str, bool)) or fresh is None:
        if actual != fresh or type(actual) is not type(fresh):
            raise ValueError("Observation metadata mismatch")
    else:
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
        ):
            raise ValueError("Finite numeric observation required")
        if "controls" in path or path[-1] in ("time_s", "root_tolerance"):
            if actual != fresh:
                raise ValueError("Observation control/time/tolerance mismatch")
        else:
            key = metric_key(path, unit)
            metrics[key] = max(metrics[key], abs(float(actual) - float(fresh)))


def runtime_state(state, initial, data, offset):
    """Reconstruct delta-state storage; never accept a claimed residual as proof."""
    n = data["n"]
    psize = 2 * n + 3
    result = np.zeros(offset + 14)
    result[0] = state[0]
    result[1 : 1 + n] = state[2 : 2 + n] - initial[2 : 2 + n]
    result[1 + n : 1 + 2 * n] = state[2 + n : 2 + 2 * n] - initial[2 + n : 2 + 2 * n]
    motor, bearing, leak, head, heat = state[2 + 2 * n : offset]
    result[psize - 2] = head
    result[psize - 1] = math.fsum((heat, leak, -state[-1]))
    result[psize:offset] = [state[1], motor, bearing, leak]
    result[offset : offset + 7] = state[offset : offset + 7] - initial[offset : offset + 7]
    result[offset + 7 : offset + 13] = state[offset + 7 : offset + 13]
    result[-1] = state[-1]
    return result


def compare_subset(actual, expected, metrics, path=()):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare_subset(actual[key], value, metrics, (*path, key))
    else:
        compare_tree(actual, expected, metrics, path)


def compare_reference(actual, expected, metrics, data, offset, vessel):
    n = data["n"]
    metrics["steam_omega_absolute_rad_s"] = max(
        metrics["steam_omega_absolute_rad_s"],
        abs(
            math.sqrt(2 * actual[offset + 6] / data["steam_inertia"])
            - math.sqrt(2 * expected[offset + 6] / data["steam_inertia"])
        ),
    )
    metrics["kinetic_energy_absolute_j"] = max(
        metrics["kinetic_energy_absolute_j"],
        abs(actual[offset + 6] - expected[offset + 6]),
        abs(
            0.5
            * data["pump"]["inertia_kg_m2"]
            * (actual[1] - expected[1])
            * (actual[1] + expected[1])
        ),
        abs(
            sum(data["inertia"])
            * (actual[0] - expected[0])
            * (actual[0] + expected[0])
            / (2 * data["rho"])
        ),
    )
    metrics["primary_flow_absolute_kg_s"] = max(
        metrics["primary_flow_absolute_kg_s"], abs(actual[0] - expected[0])
    )
    metrics["primary_omega_absolute_rad_s"] = max(
        metrics["primary_omega_absolute_rad_s"], abs(actual[1] - expected[1])
    )
    metrics["temperature_absolute_k"] = max(
        metrics["temperature_absolute_k"],
        float(max(abs(actual[2 : 2 + 2 * n] - expected[2 : 2 + 2 * n]))),
        float(max(abs(actual[offset + 4 : offset + 6] - expected[offset + 4 : offset + 6]))),
    )
    metrics["energy_integral_absolute_j"] = max(
        metrics["energy_integral_absolute_j"],
        float(max(abs(actual[2 + 2 * n : offset] - expected[2 + 2 * n : offset]))),
        float(max(abs(actual[offset + 7 :] - expected[offset + 7 :]))),
    )
    for index in (0, 1):
        a, b = (
            actual[offset + 2 * index : offset + 2 * index + 2],
            expected[offset + 2 * index : offset + 2 * index + 2],
        )
        metrics["mass_absolute_kg"] = max(metrics["mass_absolute_kg"], abs(a[0] - b[0]))
        metrics["inventory_energy_absolute_j"] = max(
            metrics["inventory_energy_absolute_j"], abs(a[1] - b[1])
        )
        pa, xa, la, _ = vessel(index, float(a[0]), float(a[1]))
        pb, xb, lb, _ = vessel(index, float(b[0]), float(b[1]))
        metrics["pressure_absolute_pa"] = max(metrics["pressure_absolute_pa"], abs(pa - pb))
        metrics["quality_absolute"] = max(metrics["quality_absolute"], abs(xa - xb))
        metrics["temperature_absolute_k"] = max(
            metrics["temperature_absolute_k"], abs(la.T_k - lb.T_k)
        )


def expected_grid(spec, end, boundary):
    grid = [(0.0, "initial", spec["controls"])]
    active = spec["controls"]
    applied = 0
    commands = []
    for index in range(1, math.ceil(spec["duration_s"] / spec["step_s"]) + 1):
        target = min(spec["duration_s"], index * spec["step_s"])
        while applied < len(spec["timeline"]) and spec["timeline"][applied]["t"] <= min(
            target, end
        ):
            event = spec["timeline"][applied]
            if boundary and event["t"] == end:
                break
            grid.append((event["t"], "before_control", active))
            commands.append(
                dict(kind="controls", t=event["t"], before=active, after=event["controls"])
            )
            active = event["controls"]
            grid.append((event["t"], "after_control", active))
            applied += 1
        if boundary and target >= end:
            grid.append((end, "domain_exit", active))
            break
        grid.append((target, "output", active))
        if target == end:
            break
    return grid, commands


def assess(folder, budget=None):
    folder = Path(folder)
    budget = json.loads(BUDGET.read_text()) if budget is None else budget
    declared = budget["reference"]
    if (
        declared["method"] != "DOP853"
        or declared["relative_tolerance"] != REFERENCE_POLICY["rtol"]
        or declared["mechanical_absolute_tolerance"] != REFERENCE_POLICY["mechanical_atol"]
        or declared["max_step_s"] != REFERENCE_POLICY["max_step_s"]
    ):
        raise ValueError("Declared independent reference policy differs from executed policy")
    manifest = json.loads((folder / "manifest.json").read_text())
    required = {
        "input.json",
        "initialization.json",
        "report.json",
        "trajectory.json",
        "checkpoint.json",
        "solver_trace.json",
    }
    if manifest["source_sha256"] != _source_fingerprint() or not required.issubset(
        manifest["artifacts"]
    ):
        raise ValueError("Primary/steam source/artifact mismatch")
    for name, checksum in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != checksum:
            raise ValueError("Primary/steam checksum/path mismatch")
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    require_finite(report)
    require_finite(rows)
    normalized, model = prepare(spec)
    if normalized != spec:
        raise ValueError("Noncanonical primary/steam input")
    at, boundary, details = reference(spec)
    rhs, margins, vessel, data, offset, initial = details
    end = report["completed_time_s"]
    if (
        boundary is None
        and (report["status"] != "completed" or end != spec["duration_s"])
        or boundary is not None
        and (
            report["status"] != "domain_exit"
            or report["boundary"]["criterion"] != boundary["criterion"]
        )
    ):
        raise ValueError("Primary/steam termination differs from independent reference")
    expected, commands = expected_grid(spec, end, boundary)
    if [(row["time_s"], row["phase"], row["controls"]) for row in rows] != expected:
        raise ValueError("Primary/steam output grid/control prefix mismatch")
    if [e for e in report["events"] if e["kind"] == "controls"] != commands:
        raise ValueError("Primary/steam report history mismatch")
    if (
        report["start_time_s"] != 0
        or report["requested_duration_s"] != spec["duration_s"]
        or report["name"] != spec["name"]
        or report["error"] is not None
        or report["numerical_acceptance"] is not (report["status"] == "completed")
        or report["accepted_prefix"] is not True
        or type(report["accepted_output_intervals"]) is not int
        or report["accepted_output_intervals"]
        != sum(row["phase"] in ("output", "domain_exit") for row in rows)
    ):
        raise ValueError("Primary/steam report completion metadata mismatch")
    for key, field in (
        ("energy_absolute_j", "energy_residual_j"),
        ("power_absolute_w", "power_residual_w"),
        ("mass_absolute_kg", "mass_residual_kg"),
    ):
        if report[key] != max(abs(row[field]) for row in rows):
            raise ValueError("Primary/steam report maximum residual mismatch")
    metrics = {key: 0.0 for key in budget["limits"]}
    for row in rows:
        y = recorded_state(row, data)
        independent = at(row["time_s"])
        compare_reference(y, independent, metrics, data, offset, vessel)
        compare_subset(row, observables(y, row["controls"], data, offset, vessel, spec), metrics)
        compare_subset(
            row, observables(independent, row["controls"], data, offset, vessel, spec), metrics
        )
        inertia = spec["steam_model"]["turbine"]["inertia_kg_m2"]
        metrics["steam_omega_absolute_rad_s"] = max(
            metrics["steam_omega_absolute_rad_s"],
            abs(
                row["steam_cycle"]["omega_rad_s"] - math.sqrt(2 * independent[offset + 6] / inertia)
            ),
        )
        model._controls = deepcopy(row["controls"])
        model._primary._torque = row["controls"]["primary_motor_torque_nm"]
        model._steam._controls = deepcopy(row["controls"]["steam_cycle"])
        reconstructed = model.observation(runtime_state(y, initial, data, offset), row["time_s"])
        compare_tree({k: v for k, v in row.items() if k != "phase"}, reconstructed, metrics)
        balance = energy_balances(y, initial, data, offset, spec)
        other = energy_balances(independent, initial, data, offset, spec)
        metrics["stored_energy_absolute_j"] = max(
            metrics["stored_energy_absolute_j"],
            abs(balance["primary_stored_j"] - other["primary_stored_j"]),
            abs(balance["steam_stored_j"] - other["steam_stored_j"]),
        )
        metrics["energy_balance_absolute_j"] = max(
            metrics["energy_balance_absolute_j"],
            *(
                abs(balance[k])
                for k in ("primary_residual_j", "steam_residual_j", "total_residual_j")
            ),
            abs(balance["total_residual_j"] - row["energy_residual_j"]),
            abs(
                balance["primary_stored_j"]
                + balance["steam_stored_j"]
                - row["stored_energy_change_j"]
            ),
        )
        metrics["reference_energy_balance_absolute_j"] = max(
            metrics["reference_energy_balance_absolute_j"],
            *(
                abs(other[k])
                for k in ("primary_residual_j", "steam_residual_j", "total_residual_j")
            ),
        )
        metrics["mass_absolute_kg"] = max(
            metrics["mass_absolute_kg"],
            abs(balance["mass_residual_kg"]),
            abs(other["mass_residual_kg"]),
        )
        metrics["power_balance_absolute_w"] = max(
            metrics["power_balance_absolute_w"],
            abs(power_balance(y, rhs(row["time_s"], y, row["controls"]), data, offset, spec)),
        )
    if boundary:
        metrics["boundary_time_absolute_s"] = abs(end - boundary["t"])
    payload = json.loads((folder / "checkpoint.json").read_text())
    restored = PrimarySteamCycle.from_snapshot(payload)
    if payload["events"] != report["events"]:
        raise ValueError("Primary/steam report/checkpoint event mismatch")
    domain_events = [event for event in report["events"] if event["kind"] == "domain_limit"]
    if report["boundary"] != (domain_events[-1] if domain_events else None):
        raise ValueError("Primary/steam report terminal event mismatch")
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Primary/steam checkpoint differs from final trajectory")
    if boundary is None:
        extended = deepcopy(spec)
        extended.update(duration_s=0.1, timeline=[], controls=rows[-1]["controls"])
        continuation, other_boundary, _ = reference(extended, initial_state=at(end))
        if other_boundary is not None:
            raise ValueError("Independent continuation exits domain")
        continued = restored.advance_until_limit(
            0.1, rtol=1e-11, max_step_s=0.005, atol_scale=0.001
        )
        if continued["boundary"] is not None:
            raise ValueError("Production continuation exits domain")
        compare_reference(
            recorded_state(continued["observation"], data),
            continuation(0.1),
            metrics,
            data,
            offset,
            vessel,
        )
        compare_subset(
            continued["observation"],
            observables(continuation(0.1), extended["controls"], data, offset, vessel, spec),
            metrics,
        )
    else:
        before = restored.snapshot()
        held = restored.advance_until_limit(0.001)
        if held["boundary"] is None or restored.snapshot() != before:
            raise ValueError("Outgoing physical boundary was not held")
    records = json.loads((folder / "solver_trace.json").read_text())
    statistics = summarize_solver_trace(records, start_time_s=0, end_time_s=end)
    if statistics != report["solver_statistics"] or not statistics["available"]:
        raise ValueError("Primary/steam actual mesh/report mismatch")
    for record in records:
        if any(
            record[key] != spec["solver"][key]
            for key in ("method", "rtol", "atol_scale", "max_step_s")
        ):
            raise ValueError("Actual solver policy differs from input")
    passed = report["accepted_prefix"] is True and all(
        np.isfinite(value) and value <= budget["limits"][key] for key, value in metrics.items()
    )
    return dict(
        accepted=bool(passed),
        metrics=metrics,
        termination=report["status"],
        solver_statistics=statistics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.folder)
    write(args.output, result)
    print(json.dumps(result))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
