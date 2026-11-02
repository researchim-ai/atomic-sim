"""Fixed-command initialization: 120s independent/constant-state checks and audit."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

from scripts.design_primary_steam import design_steady_primary_steam
from scripts.qualify_primary_steam import reuse_packet
from scripts.qualify_primary_steam import signature as assembly_signature
from scripts.qualify_primary_steam_design import stationary_metrics
from scripts.solve_fixed_primary_steam import (
    DRIVERS,
    SteadySolveError,
    inventory_anchors,
    solve_fixed_primary_steam,
)
from scripts.verify_primary_steam import assess
from scripts.verify_turbine import sha
from sdk.network_experiment import write
from sdk.primary_steam_experiment import run_primary_steam_experiment

BUDGET = Path("validation/ai05_fixed_boundary_qualification_budget_v3.json")


def signature():
    budget = json.loads(BUDGET.read_text())
    return dict(
        assembly=assembly_signature(),
        qualification_sha256=sha(__file__),
        budget_sha256=sha(BUDGET),
        steady_budget_sha256=sha(budget["steady_budget_file"]),
        base_budget_sha256=sha(budget["base_budget_file"]),
        initializer_driver_sha256={path: sha(path) for path in DRIVERS},
        stationary_verifier_sha256=sha("scripts/qualify_primary_steam_design.py"),
    )


def driver_hashes(captured):
    return {
        **captured["assembly"]["driver_sha256"],
        **captured["initializer_driver_sha256"],
        "scripts/qualify_fixed_primary_steam.py": captured["qualification_sha256"],
        "scripts/qualify_primary_steam_design.py": captured["stationary_verifier_sha256"],
    }


def perturb_guesses(spec):
    result = deepcopy(spec)
    init = result["primary"]["loop"]["initialization"]
    init["mass_flow_kg_s"] = -33
    for key in ("fluid_temperature_k", "wall_temperature_k"):
        init[key] = {name: 1000 for name in init[key]}
    result["primary"]["initial_omega_rad_s"] = 19
    result["steam_model"]["steam_generator"].update(initial_pressure_pa=7e6, initial_quality=0.2)
    result["steam_model"]["condenser"].update(initial_pressure_pa=12000, initial_quality=0.005)
    result["steam_model"]["hot_wall"]["initial_temperature_k"] = 600
    result["steam_model"]["cold_wall"]["initial_temperature_k"] = 330
    result["steam_model"]["turbine"]["initial_omega_rad_s"] = 350
    return result


def point(case):
    request = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    request["timeline"] = []
    if "primary_temperature_floor_k" in case:
        request["coupling"]["min_primary_temperature_k"] = case["primary_temperature_floor_k"]
    if case.get("ambient"):
        for node in request["primary"]["loop"]["model"]["topology"]["nodes"]:
            node["parameters"].update(ambient_ua_w_k=10, wall_power_w=100)
    if "condenser_pressure_pa" in case:
        request["steam_model"]["condenser"]["initial_pressure_pa"] = case["condenser_pressure_pa"]
    value, _ = design_steady_primary_steam(
        request, primary_mass_flow_kg_s=case["flow"], valve_opening=case["opening"]
    )
    anchors = inventory_anchors(value)
    if "quality_anchor" in case:
        anchors["steam_generator_quality"] = case["quality_anchor"]
    if "split_heat_w" in case:
        value["controls"]["primary_wall_heater_power_w"] -= case["split_heat_w"]
        value["controls"]["steam_cycle"]["heater_power_w"] += case["split_heat_w"]
    return value, anchors


def matrix(budget):
    for case in budget["cases"]:
        base, anchors = point(case)
        base = perturb_guesses(base)
        base.update(name="fixed_" + case["name"], duration_s=budget["horizon_s"], step_s=1)
        for policy, solver in budget["policies"].items():
            request = deepcopy(base)
            request["solver"] = deepcopy(solver)
            yield case["name"], policy, request, deepcopy(anchors)


def negative_matrix(budget):
    base, anchors = point(budget["cases"][0])
    for name in budget["negative_cases"]:
        request, targets = deepcopy(base), deepcopy(anchors)
        c = request["controls"]["steam_cycle"]
        expected = dict(
            heater_incompatible="hot_energy_compatibility",
            feed_pressure="pressure_compatibility",
            cooling="condenser_compatibility",
            unloaded_rotor="rotor_rank",
            rotor_band="rotor_band",
            inventory_mass="inventory_anchor",
            inventory_phase="inventory_anchor",
            thermal_rank="thermal_rank",
            pressure_endpoint="active_pressure_boundary",
            phase_boundary="active_state_boundary",
            shaft_boundary="active_state_boundary",
            primary_subcooling_boundary="active_state_boundary",
        )
        if name == "heater_incompatible":
            request["controls"]["primary_wall_heater_power_w"] += 1000
        elif name == "feed_pressure":
            c["feed_mass_flow_kg_s"] *= 100
        elif name == "cooling":
            c["sink_temperature_k"] = 500
        elif name == "unloaded_rotor":
            c["generator_connected"] = False
            request["steam_model"]["turbine"]["bearing_drag_nm_s_rad"] = 0
        elif name == "rotor_band":
            c["generator_connected"] = False
        elif name == "inventory_mass":
            targets["total_steam_mass_kg"] = 1
        elif name == "inventory_phase":
            targets["steam_generator_quality"] = 0.5
        elif name == "thermal_rank":
            for node in request["primary"]["loop"]["model"]["topology"]["nodes"]:
                node["parameters"]["fluid_wall_ua_w_k"] = 0
        elif name == "pressure_endpoint":
            request, targets = point(
                dict(
                    name=name,
                    flow=100,
                    opening=1,
                    condenser_pressure_pa=base["steam_model"]["condenser"]["min_pressure_pa"],
                )
            )
        elif name == "phase_boundary":
            targets["steam_generator_quality"] = base["steam_model"]["steam_generator"][
                "min_quality"
            ]
        elif name == "shaft_boundary":
            solved, _ = solve_fixed_primary_steam(base, **anchors)
            omega = solved["steam_model"]["turbine"]["initial_omega_rad_s"]
            request["steam_model"]["turbine"].update(
                min_omega_rad_s=omega, initial_omega_rad_s=omega
            )
        elif name == "primary_subcooling_boundary":
            from sdk.primary_steam_experiment import prepare

            solved, _ = solve_fixed_primary_steam(base, **anchors)
            margins = prepare(solved)[1].observation()["domain_margins"]
            request["coupling"]["minimum_subcooling_k"] += min(
                v["value"]
                for k, v in margins.items()
                if k.startswith("primary.") and k.endswith(".subcooling")
            )
        else:
            raise ValueError("Unknown declared negative fixed-boundary family")
        yield name, request, targets, expected[name]


def rejection(request, anchors, criterion):
    original = deepcopy(request)
    try:
        solve_fixed_primary_steam(request, **anchors)
    except SteadySolveError as error:
        if (
            request != original
            or error.report["criterion"] != criterion
            or error.report["commands_changed"]
        ):
            raise ValueError("Fixed-command rejection classification or input purity changed")
        return error.report
    raise ValueError("A declared incompatible point was accepted")


def audit(folder):
    folder = Path(folder)
    budget = json.loads(BUDGET.read_text())
    report = json.loads((folder / "qualification.json").read_text())
    if (
        report["signature"] != signature()
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Fixed-boundary runtime, drivers or budget mismatch")
    for name, digest in driver_hashes(report["signature"]).items():
        if sha(folder / "drivers" / Path(name).name) != digest:
            raise ValueError("Frozen fixed-boundary driver mismatch")
    expected = {
        (name, policy): (request, anchors) for name, policy, request, anchors in matrix(budget)
    }
    identities = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(identities) != len(set(identities))
        or set(identities) != set(expected)
        or report["passed"] is not True
        or report["integrations"] != len(expected)
        or report["case_families"] != len(budget["cases"])
    ):
        raise ValueError("Incomplete or duplicated fixed-boundary qualification matrix")
    paths = set()
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("External or shared fixed-boundary evidence folder")
        paths.add(path)
        if "reused_from" in record and record["reused_from"]["manifest_sha256"] != sha(
            path / "run/manifest.json"
        ):
            raise ValueError("Fixed-boundary raw reuse provenance mismatch")
        request, anchors = expected[(record["name"], record["policy"])]
        if (
            json.loads((path / "request.json").read_text()) != request
            or json.loads((path / "anchors.json").read_text()) != anchors
        ):
            raise ValueError("Fixed-command request or inventory target substitution")
        spec, solution = solve_fixed_primary_steam(request, **anchors)
        if (
            json.loads((path / "solution.json").read_text()) != solution
            or json.loads((path / "run/input.json").read_text()) != spec
            or spec["controls"] != request["controls"]
        ):
            raise ValueError("Fixed-command solution, commands or initial state substitution")
        fresh = assess(path / "run")
        if (
            not fresh["accepted"]
            or fresh["termination"] != "completed"
            or any(fresh[key] != record[key] for key in fresh)
            or json.loads((path / "run/report.json").read_text())["completed_time_s"]
            != budget["horizon_s"]
        ):
            raise ValueError("Fixed-boundary independent trajectory replay failed")
        stationary = stationary_metrics(path / "run", budget)
        if stationary != record["stationary_metrics"]:
            raise ValueError("Fixed-boundary constant-state/counter replay mismatch")
        print(
            json.dumps(dict(name=record["name"], policy=record["policy"], audited=True)), flush=True
        )
    expected_negative = {
        name: (req, anchors, criterion) for name, req, anchors, criterion in negative_matrix(budget)
    }
    names = [r["name"] for r in report["negative_cases"]]
    if len(names) != len(set(names)) or set(names) != set(expected_negative):
        raise ValueError("Incomplete negative fixed-boundary matrix")
    for record in report["negative_cases"]:
        path = Path(record["folder"]).resolve()
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("External or shared negative evidence folder")
        paths.add(path)
        request, anchors, criterion = expected_negative[record["name"]]
        fresh = rejection(request, anchors, criterion)
        if (
            fresh != record["diagnostics"]
            or json.loads((path / "request.json").read_text()) != request
            or json.loads((path / "anchors.json").read_text()) != anchors
            or json.loads((path / "diagnostics.json").read_text()) != fresh
        ):
            raise ValueError("Negative fixed-boundary diagnostic replay mismatch")
    if signature() != report["signature"]:
        raise ValueError("Fixed-boundary signature changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        signature=signature(),
        qualification_sha256=sha(folder / "qualification.json"),
        integrations=len(expected),
        case_families=len(budget["cases"]),
        negative_checks=len(expected_negative),
        operating_horizon_s=budget["horizon_s"],
        all_commands_preserved=True,
        warmup_steps=0,
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    if args.audit:
        write(args.output, audit(args.audit))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(BUDGET.read_text())
    captured = signature()
    write(args.output / "budget.json", budget)
    drivers = args.output / "drivers"
    drivers.mkdir()
    for name in driver_hashes(captured):
        shutil.copyfile(name, drivers / Path(name).name)
    records = []
    for name, policy, request, anchors in matrix(budget):
        if signature() != captured:
            raise ValueError("Fixed-boundary inputs changed during qualification")
        path = args.output / (name + "_" + policy)
        path.mkdir()
        write(path / "request.json", request)
        write(path / "anchors.json", anchors)
        spec, solution = solve_fixed_primary_steam(request, **anchors)
        if request["controls"] != spec["controls"]:
            raise ValueError("Boundary commands changed")
        write(path / "solution.json", solution)
        reused = None
        if args.reuse_from is not None:
            source = args.reuse_from / path.name / "run"
            if (source / "manifest.json").is_file():
                reused = reuse_packet(source, path / "run", spec)
        if reused is None:
            run_primary_steam_experiment(spec, path / "run")
        record = assess(path / "run")
        stationary = stationary_metrics(path / "run", budget)
        if not record["accepted"] or record["termination"] != "completed":
            raise ValueError("Fixed steady trajectory not accepted")
        record.update(name=name, policy=policy, folder=str(path), stationary_metrics=stationary)
        if reused is not None:
            record["reused_from"] = reused
        records.append(record)
        print(json.dumps(dict(name=name, policy=policy, accepted=True)), flush=True)
    negatives = []
    for name, request, anchors, criterion in negative_matrix(budget):
        diagnostics = rejection(request, anchors, criterion)
        path = args.output / "negative" / name
        path.mkdir(parents=True)
        write(path / "request.json", request)
        write(path / "anchors.json", anchors)
        write(path / "diagnostics.json", diagnostics)
        negatives.append(dict(name=name, folder=str(path), diagnostics=diagnostics))
    if signature() != captured:
        raise ValueError("Fixed-boundary source changed before verdict")
    report = dict(
        schema_version=1,
        signature=captured,
        passed=True,
        cases=records,
        negative_cases=negatives,
        integrations=len(records),
        case_families=len(budget["cases"]),
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "qualification.json", report)
    print(
        json.dumps(dict(passed=True, integrations=len(records), negative_checks=len(negatives))),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
