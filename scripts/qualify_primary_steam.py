"""Fixed-scope three-level primary/steam qualification with independent IVPs."""

import argparse
import importlib.metadata
import json
import platform
import shutil
from copy import deepcopy
from pathlib import Path

from scripts.primary_steam_references import (
    observables,
    recorded_state,
    system,
)
from scripts.verify_primary_steam import (
    BUDGET,
    assess,
    compare_reference,
    compare_subset,
)
from scripts.verify_turbine import sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.primary_steam_experiment import prepare, run_primary_steam_experiment

DRIVERS = (
    "scripts/audit_primary_steam.py",
    "scripts/primary_steam_references.py",
    "scripts/verify_primary_steam.py",
    "scripts/qualify_primary_steam.py",
    "scripts/verify_turbine.py",
)
PHYSICAL_STOPS = {
    "primary_pressure_margin",
    "primary_subcooling_margin",
    "steam_speed_margin",
    "steam_phase_margin",
}


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(BUDGET),
        driver_sha256={p: sha(p) for p in DRIVERS},
        python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
    )


def specifications(budget):
    base = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    names = [
        "nominal",
        "heater_step",
        "motor_trip",
        "motor_reverse",
        "reverse_heat",
        "load_rejection",
        "feed_loss",
        "sink_warming",
        "valve_reduction",
        "steam_speed_margin",
        "steam_phase_margin",
        "primary_pressure_margin",
        "primary_subcooling_margin",
        "isolated",
        *budget["long_horizons_s"],
    ]
    result = {}
    for name in names:
        spec = deepcopy(base)
        spec.update(name=name, duration_s=5, step_s=0.25, timeline=[])
        c = spec["controls"]
        steam = c["steam_cycle"]
        cfg = spec["steam_model"]
        changed = None
        if name == "heater_step":
            changed = dict(c, primary_wall_heater_power_w=c["primary_wall_heater_power_w"] * 1.2)
        elif name == "motor_trip":
            changed = dict(c, primary_motor_torque_nm=0)
        elif name == "motor_reverse":
            changed = dict(c, primary_motor_torque_nm=-c["primary_motor_torque_nm"])
        elif name == "load_rejection":
            changed = dict(c, steam_cycle=dict(steam, generator_connected=False))
        elif name == "feed_loss":
            changed = dict(c, steam_cycle=dict(steam, feed_mass_flow_kg_s=0))
        elif name == "sink_warming":
            changed = dict(
                c, steam_cycle=dict(steam, sink_temperature_k=steam["sink_temperature_k"] + 3)
            )
        elif name == "valve_reduction":
            changed = dict(
                c,
                steam_cycle=dict(
                    steam, valve_opening=0.6, feed_mass_flow_kg_s=0.6 * steam["feed_mass_flow_kg_s"]
                ),
            )
        elif name == "reverse_heat":
            init = spec["primary"]["loop"]["initialization"]
            temp = cfg["hot_wall"]["initial_temperature_k"] - 10
            for key in ("fluid_temperature_k", "wall_temperature_k"):
                init[key] = {key: temp for key in init[key]}
            c["primary_wall_heater_power_w"] = 0
        elif name == "steam_speed_margin":
            cfg["turbine"]["max_omega_rad_s"] = cfg["turbine"]["initial_omega_rad_s"] + 0.1
            spec.update(
                duration_s=1,
                timeline=[
                    dict(t=0, controls=dict(c, steam_cycle=dict(steam, generator_connected=False)))
                ],
            )
        elif name == "steam_phase_margin":
            cfg["steam_generator"]["min_quality"] = cfg["steam_generator"]["initial_quality"] - 1e-6
            spec.update(
                duration_s=1,
                timeline=[
                    dict(t=0, controls=dict(c, steam_cycle=dict(steam, feed_mass_flow_kg_s=0.2)))
                ],
            )
        elif name in ("primary_pressure_margin", "primary_subcooling_margin"):
            initial, _, margins, _, _, _ = system(spec)
            values = margins(initial, c)
            if name == "primary_pressure_margin":
                actual = max(
                    spec["coupling"]["max_primary_pressure_pa"] - v
                    for key, v in values
                    if key.startswith("primary.") and key.endswith(".pressure_high")
                )
                spec["coupling"]["max_primary_pressure_pa"] = actual + 10
                spec["timeline"] = [
                    dict(
                        t=0,
                        controls=dict(c, primary_motor_torque_nm=2 * c["primary_motor_torque_nm"]),
                    )
                ]
            else:
                current = spec["coupling"]["minimum_subcooling_k"]
                spec["coupling"]["minimum_subcooling_k"] = (
                    current
                    + min(
                        v
                        for key, v in values
                        if key.startswith("primary.") and key.endswith(".subcooling")
                    )
                    - 1e-6
                )
            spec["duration_s"] = 1
        elif name == "isolated":
            cfg["hot_wall"]["fluid_ua_w_k"] = 0
            cfg["cold_wall"].update(fluid_ua_w_k=0, ambient_ua_w_k=0)
            cfg["turbine"]["bearing_drag_nm_s_rad"] = 0
            spec["primary"]["pump"]["drag_nm_s_rad"] = 0
            c.update(
                primary_motor_torque_nm=0,
                primary_wall_heater_power_w=0,
                steam_cycle=dict(
                    steam,
                    heater_power_w=0,
                    feed_mass_flow_kg_s=0,
                    valve_opening=0,
                    generator_connected=False,
                ),
            )
        elif name in budget["long_horizons_s"]:
            spec.update(duration_s=budget["long_horizons_s"][name], step_s=1)
            if name == "long_heater":
                spec["timeline"] = [
                    dict(
                        t=1.3,
                        controls=dict(
                            c, primary_wall_heater_power_w=1.2 * c["primary_wall_heater_power_w"]
                        ),
                    )
                ]
            elif name == "long_cycling":
                spec["timeline"] = [
                    dict(
                        t=3.7,
                        controls=dict(
                            c, primary_wall_heater_power_w=1.2 * c["primary_wall_heater_power_w"]
                        ),
                    ),
                    dict(
                        t=20.3,
                        controls=dict(
                            c, primary_wall_heater_power_w=0.8 * c["primary_wall_heater_power_w"]
                        ),
                    ),
                    dict(
                        t=35.1,
                        controls=dict(
                            c,
                            primary_motor_torque_nm=0.8 * c["primary_motor_torque_nm"],
                            primary_wall_heater_power_w=0.8 * c["primary_wall_heater_power_w"],
                        ),
                    ),
                    dict(t=45.7, controls=deepcopy(c)),
                    dict(
                        t=65.3,
                        controls=dict(
                            c,
                            steam_cycle=dict(
                                steam, sink_temperature_k=steam["sink_temperature_k"] + 3
                            ),
                        ),
                    ),
                    dict(
                        t=80.1,
                        controls=dict(
                            c,
                            primary_wall_heater_power_w=0.8 * c["primary_wall_heater_power_w"],
                            steam_cycle=dict(
                                steam,
                                valve_opening=0.8,
                                feed_mass_flow_kg_s=0.8 * steam["feed_mass_flow_kg_s"],
                                generator_conductance_nm_s_rad=0.8
                                * steam["generator_conductance_nm_s_rad"],
                            ),
                        ),
                    ),
                    dict(t=95.7, controls=deepcopy(c)),
                ]
        if changed is not None:
            spec["timeline"] = [dict(t=0.73, controls=changed)]
        result[name] = spec
    return result


def matrix(budget):
    for name, spec in specifications(budget).items():
        step = budget["coarse_step_s"].get(name, budget["coarse_step_s"]["default"])
        for index, level in enumerate(budget["levels"]):
            candidate = deepcopy(spec)
            candidate["solver"] = dict(
                budget["fixed_solver"], max_step_s=step / budget["step_ratio"] ** index
            )
            yield name, level, prepare(candidate)[0]
        if name in ("nominal", "heater_step"):
            candidate = deepcopy(spec)
            candidate["solver"] = dict(
                method="BDF",
                rtol=1e-12,
                atol_scale=1e-5,
                max_step_s=step / budget["step_ratio"] ** 2,
            )
            yield name, "bdf", prepare(candidate)[0]


def compare_levels(records, budget):
    results = []
    for name in sorted({r["name"] for r in records}):
        group = {r["policy"]: r for r in records if r["name"] == name}
        pairs = []
        for left, right in zip(budget["levels"], budget["levels"][1:]):
            a, b = group[left], group[right]
            rows = json.loads((Path(a["folder"]) / "trajectory.json").read_text())
            other = json.loads((Path(b["folder"]) / "trajectory.json").read_text())
            spec = json.loads((Path(a["folder"]) / "input.json").read_text())
            _, _, _, vessel, data, offset = system(spec)
            if len(rows) != len(other):
                raise ValueError("Primary/steam comparison lengths differ")
            metrics = {key: 0.0 for key in budget["limits"]}
            for x, y in zip(rows, other, strict=True):
                if (
                    x["phase"] != y["phase"]
                    or abs(x["time_s"] - y["time_s"]) > budget["limits"]["boundary_time_absolute_s"]
                ):
                    raise ValueError("Primary/steam comparison clocks differ")
                compare_reference(
                    recorded_state(x, data), recorded_state(y, data), metrics, data, offset, vessel
                )
                compare_subset(
                    x,
                    observables(recorded_state(y, data), y["controls"], data, offset, vessel, spec),
                    metrics,
                )
            ratio = (
                a["solver_statistics"]["max_reported_step_s"]
                / b["solver_statistics"]["max_reported_step_s"]
            )
            if ratio < budget["minimum_measured_step_ratio"]:
                raise ValueError(
                    "Primary/steam requested caps did not refine actual meshes: " + name
                )
            if any(value > budget["limits"][key] for key, value in metrics.items()):
                raise ValueError("Primary/steam level differences exceed unchanged budget: " + name)
            pairs.append(
                dict(
                    from_level=left,
                    to_level=right,
                    maximum_differences=metrics,
                    measured_step_ratio=ratio,
                )
            )
        results.append(
            dict(
                name=name,
                comparisons=pairs,
                observed_order=None,
                order_reason="Adaptive steps and fixed solver tolerances; no asymptotic order inferred from capped grids",
            )
        )
    return results


def reuse_packet(source, target, spec):
    source, target = Path(source), Path(target)
    if not source.exists():
        return None
    if json.loads((source / "input.json").read_text()) != spec:
        return None
    manifest = json.loads((source / "manifest.json").read_text())
    current = signature()
    if (
        manifest["source_sha256"] != current["source_sha256"]
        or manifest["python"] != current["python"]
        or manifest["versions"] != current["versions"]
    ):
        raise ValueError("Cannot reuse a different runtime/environment")
    required = {
        "input.json",
        "initialization.json",
        "report.json",
        "trajectory.json",
        "checkpoint.json",
        "solver_trace.json",
    }
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("Incomplete reuse packet")
    for name, h in manifest["artifacts"].items():
        path = source / name
        if Path(name).name != name or path.is_symlink() or sha(path) != h:
            raise ValueError("Corrupt/external reuse packet")
    target.mkdir(parents=True, exist_ok=False)
    for name in [*manifest["artifacts"], "manifest.json"]:
        shutil.copyfile(source / name, target / name)
    return dict(source_folder=str(source), manifest_sha256=sha(target / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(BUDGET.read_text())
    frozen = signature()
    write(args.output / "budget.json", budget)
    drivers = args.output / "drivers"
    drivers.mkdir()
    for path in DRIVERS:
        shutil.copyfile(path, drivers / Path(path).name)
    records = []
    for name, level, spec in matrix(budget):
        if signature() != frozen:
            raise ValueError("Runtime/budget/driver changed during qualification")
        folder = args.output / f"{name}_{level}"
        reused = (
            None
            if args.reuse_from is None
            else reuse_packet(args.reuse_from / folder.name, folder, spec)
        )
        if reused is None:
            run_primary_steam_experiment(spec, folder)
        record = assess(folder, budget)
        expected = "domain_exit" if name in PHYSICAL_STOPS else "completed"
        if record["termination"] != expected:
            raise ValueError("Requested primary/steam operating horizon was not completed: " + name)
        record.update(name=name, policy=level, folder=str(folder))
        if reused is not None:
            record["reused_from"] = reused
        records.append(record)
        print(
            json.dumps(
                dict(
                    name=name,
                    level=level,
                    accepted=record["accepted"],
                    mesh_steps=record["solver_statistics"]["reported_mesh_steps"],
                )
            ),
            flush=True,
        )
        if level == budget["levels"][-1]:
            try:
                compare_levels([r for r in records if r["name"] == name], budget)
            except ValueError as exc:
                write(
                    args.output / "rejection.json",
                    dict(
                        passed=False,
                        stage="temporal_refinement",
                        name=name,
                        error=str(exc),
                        signature=frozen,
                    ),
                )
                raise
    comparisons = compare_levels(records, budget)
    if signature() != frozen:
        raise ValueError("Qualification signature changed before verdict")
    result = dict(
        schema_version=1,
        signature=frozen,
        passed=all(r["accepted"] for r in records),
        cases=records,
        integrations=len(records),
        case_families=len(comparisons),
        temporal_comparisons=comparisons,
        reference_cache="Identical physical IVPs only; every packet assessment recomputed",
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "qualification.json", result)
    print(
        json.dumps({key: result[key] for key in ("passed", "integrations", "case_families")}),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
