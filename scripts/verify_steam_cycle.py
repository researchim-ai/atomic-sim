"""Fixed-scope independent steam-cycle acceptance with physical-domain exits."""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.power.steam_cycle import SteamPowerCycle
from scripts.steam_cycle_references import reference
from scripts.verify_turbine import require_finite, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.steam_cycle_experiment import prepare, run_steam_cycle_experiment

BUDGET = Path("validation/ai05_steam_cycle_budget_v1.json")


def matrix():
    base = json.loads(Path("scenarios/ai05_dynamic_steam_cycle.json").read_text())
    for index, name in enumerate(
        (
            "nominal",
            "heater_step",
            "load_rejection",
            "sink_warming",
            "feed_loss",
            "valve_reduction",
            "speed_margin",
            "phase_margin",
            "pressure_margin",
            "isolated",
        )
    ):
        spec = deepcopy(base)
        spec.update(name=name, duration_s=5, timeline=[])
        c = spec["controls"]
        if name == "heater_step":
            spec["timeline"] = [
                dict(t=1.3, controls=dict(c, heater_power_w=c["heater_power_w"] * 1.2))
            ]
        elif name == "load_rejection":
            spec["timeline"] = [dict(t=1.3, controls=dict(c, generator_connected=False))]
        elif name == "sink_warming":
            spec["timeline"] = [
                dict(t=1.3, controls=dict(c, sink_temperature_k=c["sink_temperature_k"] + 5))
            ]
        elif name == "feed_loss":
            spec["timeline"] = [dict(t=1.3, controls=dict(c, feed_mass_flow_kg_s=0))]
        elif name == "valve_reduction":
            spec["timeline"] = [
                dict(
                    t=1.3,
                    controls=dict(
                        c, valve_opening=0.6, feed_mass_flow_kg_s=c["feed_mass_flow_kg_s"] * 0.6
                    ),
                )
            ]
        elif name == "speed_margin":
            spec.update(
                duration_s=1, timeline=[dict(t=0, controls=dict(c, generator_connected=False))]
            )
            spec["model"]["turbine"]["max_omega_rad_s"] = 300.1
        elif name == "phase_margin":
            spec.update(
                duration_s=1, timeline=[dict(t=0, controls=dict(c, feed_mass_flow_kg_s=0.2))]
            )
            spec["model"]["steam_generator"]["min_quality"] = 0.1 - 1e-6
        elif name == "pressure_margin":
            spec.update(
                duration_s=2,
                timeline=[dict(t=0, controls=dict(c, heater_power_w=c["heater_power_w"] * 1.2))],
            )
            spec["model"]["steam_generator"]["max_pressure_pa"] = 6e6 + 20
        elif name == "isolated":
            spec["model"]["hot_wall"]["fluid_ua_w_k"] = 0
            spec["model"]["cold_wall"].update(fluid_ua_w_k=0, ambient_ua_w_k=0)
            spec["model"]["turbine"]["bearing_drag_nm_s_rad"] = 0
            c.update(
                heater_power_w=0, valve_opening=0, feed_mass_flow_kg_s=0, generator_connected=False
            )
        for policy in ("base", "tight", "bdf") if index < 2 else ("base", "tight"):
            candidate = deepcopy(spec)
            if policy != "base":
                candidate["solver"].update(rtol=1e-10, max_step_s=0.025, atol_scale=0.01)
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
        raise ValueError("Steam-cycle source/artifact mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError("Steam-cycle checksum/path mismatch")
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
        and (
            report["status"] != "domain_exit"
            or report["boundary"]["criterion"] != boundary["criterion"]
        )
    ):
        raise ValueError("Steam-cycle termination differs from independent reference")
    expected = [(0.0, "initial", spec["controls"])]
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
            expected.append((event["t"], "before_control", active))
            commands.append(
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
        raise ValueError("Steam-cycle grid/control prefix mismatch")
    if [e for e in report["events"] if e["kind"] == "controls"] != commands:
        raise ValueError("Steam-cycle report control history mismatch")
    metrics = dict(
        pressure_absolute_pa=0.0,
        temperature_absolute_k=0.0,
        quality_absolute=0.0,
        omega_absolute_rad_s=0.0,
        mass_absolute_kg=0.0,
        energy_absolute_j=0.0,
        reference_energy_absolute_j=0.0,
        reference_mass_absolute_kg=0.0,
        power_absolute_w=0.0,
        boundary_time_absolute_s=0.0,
        restart_omega_absolute_rad_s=0.0,
    )
    origin = at(0)["state"]
    observer = prepare(spec)[1]
    for row in rows:
        ref = at(row["time_s"])
        y = ref["state"]
        for name in ("steam_generator", "condenser"):
            actual, other = row["vessels"][name], ref["vessels"][name]
            metrics["mass_absolute_kg"] = max(
                metrics["mass_absolute_kg"], abs(actual["mass_kg"] - other["mass_kg"])
            )
            metrics["reference_energy_absolute_j"] = max(
                metrics["reference_energy_absolute_j"],
                abs(actual["internal_energy_j"] - other["internal_energy_j"]),
            )
            for metric, key in (
                ("pressure_absolute_pa", "pressure_pa"),
                ("temperature_absolute_k", "temperature_k"),
                ("quality_absolute", "quality"),
            ):
                metrics[metric] = max(metrics[metric], abs(actual[key] - other[key]))
        reconstructed = np.zeros(13)
        for i, name in enumerate(("steam_generator", "condenser")):
            reconstructed[2 * i] = row["vessels"][name]["mass_kg"] - observer._origins[i][0]
            reconstructed[2 * i + 1] = (
                row["vessels"][name]["internal_energy_j"] - observer._origins[i][1]
            )
        reconstructed[4] = (
            row["hot_wall_temperature_k"] - spec["model"]["hot_wall"]["initial_temperature_k"]
        )
        reconstructed[5] = (
            row["cold_wall_temperature_k"] - spec["model"]["cold_wall"]["initial_temperature_k"]
        )
        reconstructed[6] = row["rotor_energy_j"] - observer._rotor0
        reconstructed[7:] = [
            row["energy_integrals_j"][k]
            for k in (
                "heater",
                "feed_pump",
                "electric",
                "generator_heat",
                "bearing_heat",
                "sink_heat",
            )
        ]
        observer._controls = row["controls"]
        fresh = observer.observation(reconstructed, row["time_s"])
        metrics["energy_absolute_j"] = max(
            metrics["energy_absolute_j"],
            abs(fresh["energy_residual_j"]),
            abs(fresh["stored_energy_change_j"] - row["stored_energy_change_j"]),
        )
        for key in (
            "steam_power_w",
            "electric_power_w",
            "generator_heat_w",
            "bearing_heat_w",
            "feed_pump_power_w",
            "hot_wall_heat_w",
            "cold_wall_heat_w",
            "sink_heat_w",
        ):
            metrics["power_absolute_w"] = max(
                metrics["power_absolute_w"], abs(fresh[key] - row[key])
            )
        for key, value in fresh["internal_enthalpy_ports_w"].items():
            metrics["power_absolute_w"] = max(
                metrics["power_absolute_w"], abs(row["internal_enthalpy_ports_w"][key] - value)
            )
        metrics["omega_absolute_rad_s"] = max(
            metrics["omega_absolute_rad_s"],
            abs(row["omega_rad_s"] - np.sqrt(2 * y[6] / spec["model"]["turbine"]["inertia_kg_m2"])),
        )
        metrics["mass_absolute_kg"] = max(metrics["mass_absolute_kg"], abs(row["mass_residual_kg"]))
        metrics["energy_absolute_j"] = max(
            metrics["energy_absolute_j"], abs(row["energy_residual_j"])
        )
        metrics["power_absolute_w"] = max(metrics["power_absolute_w"], abs(row["power_residual_w"]))
        energy = (
            y[1]
            - origin[1]
            + y[3]
            - origin[3]
            + spec["model"]["hot_wall"]["heat_capacity_j_k"] * (y[4] - origin[4])
            + spec["model"]["cold_wall"]["heat_capacity_j_k"] * (y[5] - origin[5])
            + y[6]
            - origin[6]
        )
        metrics["reference_energy_absolute_j"] = max(
            metrics["reference_energy_absolute_j"],
            abs(energy - y[7] - y[8] + sum(y[9:])),
            abs(row["stored_energy_change_j"] - energy),
        )
        metrics["reference_mass_absolute_kg"] = max(
            metrics["reference_mass_absolute_kg"], abs(y[0] + y[2] - origin[0] - origin[2])
        )
    if boundary:
        metrics["boundary_time_absolute_s"] = abs(end - boundary["t"])
    snap = json.loads((folder / "checkpoint.json").read_text())
    restored = SteamPowerCycle.from_snapshot(snap)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Steam-cycle checkpoint differs from trajectory")
    if boundary is None:
        extended = deepcopy(spec)
        extended["duration_s"] += 0.1
        independent, _ = reference(extended)
        result = restored.advance_until_limit(0.1, rtol=1e-10, max_step_s=0.025, atol_scale=0.01)
        if result["boundary"] is not None:
            raise ValueError("Unexpected steam-cycle extension boundary")
        ref_record = independent(extended["duration_s"])
        ref = ref_record["state"]
        continued_row = result["observation"]
        metrics["restart_omega_absolute_rad_s"] = abs(
            result["observation"]["omega_rad_s"]
            - np.sqrt(2 * ref[6] / spec["model"]["turbine"]["inertia_kg_m2"])
        )
    else:
        recovery = deepcopy(spec)
        recovery.update(
            duration_s=0.0001,
            timeline=[],
            controls=dict(
                spec["controls"], heater_power_w=0, feed_mass_flow_kg_s=0, generator_connected=True
            ),
        )
        if boundary["criterion"] == "steam_generator.pressure_high":
            recovery["controls"]["feed_mass_flow_kg_s"] = (
                2 * spec["controls"]["feed_mass_flow_kg_s"]
            )
        initial = restored._state.copy()
        for index, (mass, energy) in enumerate(restored._origins):
            initial[2 * index] += mass
            initial[2 * index + 1] += energy
        initial[4] += spec["model"]["hot_wall"]["initial_temperature_k"]
        initial[5] += spec["model"]["cold_wall"]["initial_temperature_k"]
        initial[6] += restored._rotor0
        independent, _ = reference(recovery, initial_state=initial)
        restored.set_controls(recovery["controls"])
        continued = restored.advance_until_limit(
            0.0001, rtol=1e-10, max_step_s=0.025, atol_scale=0.01
        )
        if continued["boundary"] is not None:
            raise ValueError("Returning command does not re-enter steam-cycle domain")
        ref_record = independent(0.0001)
        ref = ref_record["state"]
        continued_row = continued["observation"]
        metrics["restart_omega_absolute_rad_s"] = abs(
            continued["observation"]["omega_rad_s"]
            - np.sqrt(2 * ref[6] / spec["model"]["turbine"]["inertia_kg_m2"])
        )
    for name in ("steam_generator", "condenser"):
        actual, other = continued_row["vessels"][name], ref_record["vessels"][name]
        for metric, key in (
            ("pressure_absolute_pa", "pressure_pa"),
            ("temperature_absolute_k", "temperature_k"),
            ("quality_absolute", "quality"),
            ("mass_absolute_kg", "mass_kg"),
            ("reference_energy_absolute_j", "internal_energy_j"),
        ):
            metrics[metric] = max(metrics[metric], abs(actual[key] - other[key]))
    metrics["energy_absolute_j"] = max(
        metrics["energy_absolute_j"], abs(continued_row["energy_residual_j"])
    )
    metrics["mass_absolute_kg"] = max(
        metrics["mass_absolute_kg"], abs(continued_row["mass_residual_kg"])
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
        run_steam_cycle_experiment(spec, folder)
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
            reference_sha256=sha("scripts/steam_cycle_references.py"),
            helper_sha256=sha("scripts/verify_turbine.py"),
            budget_sha256=sha(BUDGET),
            cases=records,
            integrations=len(records),
            case_families=10,
            all_cases_accepted=passed,
            validated=False,
            plant_referenced=False,
        ),
    )
    print(json.dumps(dict(passed=passed, integrations=len(records))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
