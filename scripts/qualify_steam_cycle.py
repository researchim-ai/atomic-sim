"""Three real solver-step levels, long workloads and independently replayable evidence."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from copy import deepcopy
from pathlib import Path

from core.solver_trace import summarize_solver_trace
from scripts.verify_steam_cycle import assess
from scripts.verify_steam_cycle import matrix as original_matrix
from scripts.verify_turbine import sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.steam_cycle_experiment import prepare, run_steam_cycle_experiment

BUDGET = Path("validation/ai05_steam_cycle_refinement_budget_v2.json")
DRIVERS = (
    "scripts/qualify_steam_cycle.py",
    "scripts/verify_steam_cycle.py",
    "scripts/steam_cycle_references.py",
    "scripts/verify_turbine.py",
    "scripts/turbine_references.py",
)


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        budget_sha256=sha(BUDGET),
        driver_sha256={p: sha(p) for p in DRIVERS},
    )


def specifications(budget):
    result = {name: deepcopy(spec) for name, policy, spec in original_matrix() if policy == "base"}
    base = deepcopy(result["nominal"])
    for name, horizon in budget["long_horizons_s"].items():
        spec = deepcopy(base)
        spec.update(name=name, duration_s=horizon, step_s=1, timeline=[])
        c = spec["controls"]
        if name == "long_heater":
            spec["timeline"] = [
                dict(t=1.3, controls=dict(c, heater_power_w=c["heater_power_w"] * 1.2))
            ]
        elif name == "long_feed_bias":
            spec["timeline"] = [
                dict(t=1.3, controls=dict(c, feed_mass_flow_kg_s=c["feed_mass_flow_kg_s"] * 0.9))
            ]
        elif name == "long_cycling":
            spec["timeline"] = [
                dict(t=1.3, controls=dict(c, heater_power_w=c["heater_power_w"] * 1.2)),
                dict(t=15.7, controls=dict(c, heater_power_w=c["heater_power_w"] * 0.8)),
                dict(t=30.2, controls=deepcopy(c)),
                dict(t=45.3, controls=dict(c, sink_temperature_k=c["sink_temperature_k"] + 3)),
                dict(
                    t=65.9,
                    controls=dict(
                        c,
                        heater_power_w=c["heater_power_w"] * 0.8,
                        valve_opening=0.8,
                        feed_mass_flow_kg_s=c["feed_mass_flow_kg_s"] * 0.8,
                        generator_conductance_nm_s_rad=c["generator_conductance_nm_s_rad"] * 0.8,
                    ),
                ),
                dict(t=90.1, controls=deepcopy(c)),
            ]
        result[name] = spec
    return result


def matrix(budget):
    for name, spec in specifications(budget).items():
        base_step = budget["coarse_step_s"].get(name, budget["coarse_step_s"]["default"])
        for index, level in enumerate(budget["levels"]):
            candidate = deepcopy(spec)
            candidate["solver"].update(
                method="Radau",
                **budget["fixed_radau_tolerances"],
                max_step_s=base_step / budget["step_ratio"] ** index,
            )
            yield name, level, prepare(candidate)[0]
        if name in ("nominal", "heater_step"):
            candidate = deepcopy(spec)
            candidate["solver"].update(
                method="BDF",
                **budget["fixed_radau_tolerances"],
                max_step_s=base_step / budget["step_ratio"],
            )
            yield name, "bdf", prepare(candidate)[0]


def trace(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    spec = json.loads((folder / "input.json").read_text())
    records = json.loads((folder / "solver_trace.json").read_text())
    summary = summarize_solver_trace(
        records, start_time_s=report["start_time_s"], end_time_s=report["completed_time_s"]
    )
    if not summary["available"] or summary != report["solver_statistics"]:
        raise ValueError("Missing or inconsistent solver mesh/statistics")
    for item in records:
        if any(
            item[key] != spec["solver"][key]
            for key in ("method", "max_step_s", "rtol", "atol_scale")
        ):
            raise ValueError("Actual solver trace differs from declared policy")
    return summary


def compare_levels(cases, budget):
    records = []
    for name in sorted({c["name"] for c in cases}):
        group = {c["policy"]: c for c in cases if c["name"] == name}
        levels = [group[level] for level in budget["levels"]]
        differences = []
        for left, right in zip(levels, levels[1:]):
            a = json.loads((Path(left["folder"]) / "trajectory.json").read_text())
            b = json.loads((Path(right["folder"]) / "trajectory.json").read_text())
            if len(a) != len(b):
                raise ValueError("Different refinement comparison lengths")
            metrics = {key: 0.0 for key in budget["comparison_limits"]}
            for x, y in zip(a, b):
                if (
                    x["phase"] != y["phase"]
                    or abs(x["time_s"] - y["time_s"])
                    > budget["base_limits"]["boundary_time_absolute_s"]
                ):
                    raise ValueError("Different refinement clocks or terminal events")
                metrics["omega_absolute_rad_s"] = max(
                    metrics["omega_absolute_rad_s"], abs(x["omega_rad_s"] - y["omega_rad_s"])
                )
                for vessel in ("steam_generator", "condenser"):
                    for metric, key in (
                        ("pressure_absolute_pa", "pressure_pa"),
                        ("temperature_absolute_k", "temperature_k"),
                        ("quality_absolute", "quality"),
                    ):
                        metrics[metric] = max(
                            metrics[metric],
                            abs(x["vessels"][vessel][key] - y["vessels"][vessel][key]),
                        )
            if any(value > budget["comparison_limits"][key] for key, value in metrics.items()):
                raise ValueError("Temporal level difference exceeds budget")
            actual_ratio = (
                left["solver_statistics"]["max_reported_step_s"]
                / right["solver_statistics"]["max_reported_step_s"]
            )
            if actual_ratio < budget["minimum_measured_step_ratio"]:
                raise ValueError("Requested refinement did not refine the returned mesh")
            differences.append(
                dict(
                    from_level=left["policy"],
                    to_level=right["policy"],
                    maximum_differences=metrics,
                    measured_step_ratio=actual_ratio,
                )
            )
        # Do not turn EOS/roundoff noise or adaptive-step comparisons into a fake order.
        noise = {
            key: all(
                pair["maximum_differences"][key] <= budget["noise_floors"][key]
                for pair in differences
            )
            for key in budget["noise_floors"]
        }
        records.append(
            dict(
                name=name,
                comparisons=differences,
                at_or_below_noise=noise,
                observed_order=None,
                order_reason="Adaptive solver with a fixed tolerance and capped steps; report differences and noise floors without inventing an asymptotic order",
            )
        )
    return records


def reuse_packet(source, target, spec):
    """Reuse raw integration bytes only; every numerical assessment is recomputed."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return None
    if json.loads((source / "input.json").read_text()) != spec:
        return None
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["python"] != platform.python_version() or manifest["versions"] != {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy")
    }:
        raise ValueError("Cannot reuse a packet from different Python/solver dependencies")
    required = {
        "input.json",
        "initialization.json",
        "report.json",
        "trajectory.json",
        "checkpoint.json",
        "solver_trace.json",
    }
    if manifest["source_sha256"] != signature()["source_sha256"] or not required.issubset(
        manifest["artifacts"]
    ):
        raise ValueError("Cannot reuse a stale or incomplete runtime packet")
    for name, checksum in manifest["artifacts"].items():
        item = source / name
        if Path(name).name != name or item.is_symlink() or sha(item) != checksum:
            raise ValueError("Cannot reuse corrupt or external packet artifacts")
    target.mkdir(parents=True, exist_ok=False)
    for name in [*manifest["artifacts"], "manifest.json"]:
        shutil.copyfile(source / name, target / name)
    return dict(source_folder=str(source), manifest_sha256=sha(target / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="Same-runtime raw packets; assessments are always recomputed",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(BUDGET.read_text())
    if (
        sha(budget["base_budget_file"]) != budget["base_budget_sha256"]
        or json.loads(Path(budget["base_budget_file"]).read_text()) != budget["base_limits"]
    ):
        raise ValueError("Original numerical budgets changed")
    initial_signature = signature()
    write(args.output / "budget.json", budget)
    frozen = args.output / "drivers"
    frozen.mkdir()
    for path in DRIVERS:
        (frozen / Path(path).name).write_bytes(Path(path).read_bytes())
    started = time.monotonic()
    cases = []
    for name, level, spec in matrix(budget):
        if signature() != initial_signature:
            raise ValueError("Runtime/driver changed during qualification")
        folder = args.output / f"{name}_{level}"
        reused = (
            None
            if args.reuse_from is None
            else reuse_packet(args.reuse_from / folder.name, folder, spec)
        )
        if reused is None:
            run_steam_cycle_experiment(spec, folder)
        record = assess(folder, budget["base_limits"])
        record.update(name=name, policy=level, folder=str(folder), solver_statistics=trace(folder))
        if reused is not None:
            record["reused_from"] = reused
        cases.append(record)
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
    comparisons = compare_levels(cases, budget)
    if signature() != initial_signature:
        raise ValueError("Runtime/driver changed before qualification verdict")
    passed = all(c["accepted"] for c in cases)
    result = dict(
        schema_version=1,
        signature=initial_signature,
        passed=passed,
        cases=cases,
        integrations=len(cases),
        case_families=len(comparisons),
        temporal_comparisons=comparisons,
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "qualification.json", result)
    print(
        json.dumps(dict(passed=passed, integrations=len(cases), case_families=len(comparisons))),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
