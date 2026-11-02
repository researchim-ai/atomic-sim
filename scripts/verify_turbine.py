"""Fixed-scope saturated-steam train acceptance, including declared domain exits."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.power.steam_turbine_train import SteamTurbineTrain
from scripts.turbine_references import reference
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.turbine_experiment import prepare, run_turbine_experiment

BUDGET = Path("validation/ai05_turbine_budget_v1.json")


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
        raise ValueError("Nonfinite turbine evidence")


def matrix():
    base = json.loads(Path("scenarios/ai05_turbine_load_rejection.json").read_text())
    for name in (
        "nominal",
        "load_rejection",
        "upper_exit",
        "trip_exit",
        "lossless_acceleration",
        "free_coast",
        "unity_generator",
        "isentropic",
        "higher_backpressure",
        "small_flow",
        "initial_upper",
        "superheated_exhaust",
    ):
        spec = deepcopy(base)
        spec.update(name=name, timeline=[])
        if name in ("load_rejection", "upper_exit"):
            spec["timeline"] = deepcopy(base["timeline"])
            if name == "upper_exit":
                spec["duration_s"] = 120
        elif name == "trip_exit":
            spec.update(
                duration_s=120,
                timeline=[dict(t=0, controls=dict(spec["controls"], mass_flow_kg_s=0))],
            )
        elif name == "lossless_acceleration":
            spec.update(duration_s=10)
            spec["model"]["bearing_drag_nm_s_rad"] = 0
            spec["controls"]["generator_connected"] = False
        elif name == "free_coast":
            spec["model"]["bearing_drag_nm_s_rad"] = 0
            spec["controls"].update(mass_flow_kg_s=0, generator_connected=False)
        elif name == "unity_generator":
            spec["model"]["generator_efficiency"] = 1
        elif name == "isentropic":
            spec["model"]["isentropic_efficiency"] = 1
        elif name == "higher_backpressure":
            spec["model"]["outlet_pressure_pa"] = 20000
        elif name == "small_flow":
            spec["controls"]["mass_flow_kg_s"] = 0.1
        elif name == "initial_upper":
            spec["initial_omega_rad_s"] = 400
            spec["controls"]["generator_connected"] = False
        elif name == "superheated_exhaust":
            spec["model"]["isentropic_efficiency"] = 0.1
        for policy in ("base", "subdivided"):
            candidate = deepcopy(spec)
            if policy == "subdivided":
                candidate["step_s"] /= 2
            yield name, policy, prepare(candidate)[0]


def assess(folder, budget):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    required = {
        "input.json",
        "thermodynamics.json",
        "trajectory.json",
        "checkpoint.json",
        "report.json",
    }
    if manifest["source_sha256"] != _source_fingerprint() or not required.issubset(
        manifest["artifacts"]
    ):
        raise ValueError("Turbine source/artifact mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError("Turbine checksum/path mismatch")
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    require_finite(report)
    require_finite(rows)
    at, boundary = reference(spec)
    end = report["completed_time_s"]
    if (
        boundary is None
        and (report["status"] != "completed" or end != spec["duration_s"])
        or boundary is not None
        and (report["status"] != "domain_exit" or report["boundary"]["limit"] != boundary["limit"])
    ):
        raise ValueError("Turbine completion/domain-exit classification differs from reference")
    expected = [(0.0, "initial", spec["controls"])]
    active = spec["controls"]
    applied = 0
    expected_controls = []
    for index in range(1, math.ceil(spec["duration_s"] / spec["step_s"]) + 1):
        target = min(spec["duration_s"], index * spec["step_s"])
        while applied < len(spec["timeline"]) and spec["timeline"][applied]["t"] <= min(
            target, end
        ):
            event = spec["timeline"][applied]
            if boundary and event["t"] == end:
                break
            expected.append((event["t"], "before_control", active))
            expected_controls.append(
                dict(kind="controls", t=event["t"], before=active, after=event["controls"])
            )
            active = event["controls"]
            expected.append((event["t"], "after_control", active))
            applied += 1
        if boundary and target >= end:
            expected.append((end, "domain_exit", active))
            break
        expected.append((target, "output", active))
        if target == end:
            break
    if [(r["time_s"], r["phase"], r["controls"]) for r in rows] != expected:
        raise ValueError("Turbine trajectory grid/control prefix mismatch")
    if [e for e in report["events"] if e["kind"] == "controls"] != expected_controls:
        raise ValueError("Turbine report command history mismatch")
    metrics = dict(
        omega_absolute_rad_s=0.0,
        energy_absolute_j=0.0,
        integral_absolute_j=0.0,
        fluid_power_absolute_w=0.0,
        boundary_time_absolute_s=0.0,
        restart_omega_absolute_rad_s=0.0,
    )
    initial_energy = 0.5 * spec["model"]["inertia_kg_m2"] * spec["initial_omega_rad_s"] ** 2
    keys = ("steam_work_j", "electric_energy_j", "generator_heat_j", "bearing_heat_j")
    for row in rows:
        ref = at(row["time_s"])
        metrics["omega_absolute_rad_s"] = max(
            metrics["omega_absolute_rad_s"], abs(row["omega_rad_s"] - ref[0])
        )
        metrics["energy_absolute_j"] = max(
            metrics["energy_absolute_j"],
            abs(row["energy_residual_j"]),
            abs(row["rotor_energy_j"] - initial_energy - ref[1] + sum(ref[2:])),
        )
        metrics["integral_absolute_j"] = max(
            metrics["integral_absolute_j"],
            max(abs(row[key] - ref[i + 1]) for i, key in enumerate(keys)),
        )
        metrics["fluid_power_absolute_w"] = max(
            metrics["fluid_power_absolute_w"], abs(row["fluid_power_residual_w"])
        )
    if boundary:
        metrics["boundary_time_absolute_s"] = abs(end - boundary["t"])
    snapshot = json.loads((folder / "checkpoint.json").read_text())
    restored = SteamTurbineTrain.from_snapshot(snapshot)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Turbine checkpoint differs from trajectory")
    recovery = deepcopy(spec)
    recovery.update(
        initial_omega_rad_s=restored.observation()["omega_rad_s"], duration_s=1, timeline=[]
    )
    recovery["controls"] = restored.boundary_controls
    if boundary:
        recovery["controls"].update(
            mass_flow_kg_s=0 if boundary["limit"] == "upper" else 10,
            generator_connected=boundary["limit"] == "upper",
        )
        restored.set_controls(recovery["controls"])
    independent, _ = reference(recovery)
    advanced = restored.advance_until_limit(1)
    if advanced["boundary"] is not None:
        raise ValueError("Unexpected boundary in recovery extension")
    metrics["restart_omega_absolute_rad_s"] = abs(
        advanced["observation"]["omega_rad_s"] - independent(1)[0]
    )
    accepted = report["accepted_prefix"] is True and all(
        np.isfinite(value) and value <= budget[key] for key, value in metrics.items()
    )
    return dict(accepted=bool(accepted), metrics=metrics, termination=report["status"])


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
        run_turbine_experiment(spec, folder)
        result = assess(folder, budget)
        result.update(name=name, policy=policy, folder=str(folder))
        records.append(result)
    passed = all(r["accepted"] for r in records)
    write(
        args.output / "verification.json",
        dict(
            schema_version=1,
            source_sha256=_source_fingerprint(),
            script_sha256=sha(__file__),
            reference_sha256=sha("scripts/turbine_references.py"),
            budget_sha256=sha(BUDGET),
            cases=records,
            integrations=len(records),
            case_families=len(records) // 2,
            all_cases_accepted=passed,
            validated=False,
            plant_referenced=False,
        ),
    )
    print(json.dumps(dict(passed=passed, integrations=len(records))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
