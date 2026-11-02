"""Independent steady-design admission: signed flow, heat losses and 120s stationarity."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from scripts.design_primary_steam import design_steady_primary_steam
from scripts.primary_steam_references import observables, recorded_state, system
from scripts.qualify_primary_steam import signature as assembly_signature
from scripts.verify_primary_steam import assess, compare_reference, compare_subset
from scripts.verify_turbine import sha
from sdk.network_experiment import write
from sdk.primary_steam_experiment import run_primary_steam_experiment

BUDGET = Path("validation/ai05_primary_steam_design_budget_v3.json")


def signature():
    budget = json.loads(BUDGET.read_text())
    return dict(
        assembly=assembly_signature(),
        design_budget_sha256=sha(BUDGET),
        base_budget_sha256=sha(budget["base_budget_file"]),
        designer_sha256=sha("scripts/design_primary_steam.py"),
        qualification_sha256=sha(__file__),
    )


def matrix(budget):
    base = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    base["timeline"] = []
    cases = [
        ("forward", 100.0, 1.0, False),
        ("reverse", -100.0, 1.0, False),
        ("low_flow", 10.0, 1.0, False),
        ("low_reverse", -10.0, 1.0, False),
        ("partial_admission", 100.0, 0.6, False),
        ("partial_reverse", -100.0, 0.6, False),
        ("ambient_loss", 100.0, 1.0, True),
    ]
    for name, flow, opening, ambient in cases:
        request = deepcopy(base)
        request["name"] = "steady_" + name
        request.update(
            duration_s=budget["long_stationary_horizon_s"]
            if name in ("forward", "reverse")
            else budget["other_horizon_s"],
            step_s=1,
        )
        if ambient:
            for node in request["primary"]["loop"]["model"]["topology"]["nodes"]:
                node["parameters"].update(ambient_ua_w_k=10, wall_power_w=100)
        for policy, solver in budget["policies"].items():
            candidate = deepcopy(request)
            candidate["solver"] = deepcopy(solver)
            yield name, policy, flow, opening, candidate


def steady_residuals(spec, budget):
    initial, rhs, _, _, data, offset = system(spec)
    rates = rhs(0, initial, spec["controls"])
    n = data["n"]
    cfg = spec["steam_model"]
    powers = np.r_[
        data["capacity"] * rates[2 : 2 + n],
        data["wall_capacity"] * rates[2 + n : 2 + 2 * n],
        rates[offset + 1],
        rates[offset + 3],
        cfg["hot_wall"]["heat_capacity_j_k"] * rates[offset + 4],
        cfg["cold_wall"]["heat_capacity_j_k"] * rates[offset + 5],
        rates[offset + 6],
    ]
    result = dict(
        maximum_stock_power_residual_w=float(max(abs(powers))),
        maximum_primary_momentum_residual_pa=float(abs(sum(data["inertia"]) * rates[0])),
        maximum_primary_shaft_residual_nm=float(abs(data["pump"]["inertia_kg_m2"] * rates[1])),
        maximum_steam_mass_rate_kg_s=float(max(abs(rates[offset]), abs(rates[offset + 2]))),
    )
    if any(not np.isfinite(value) or value > budget[key] for key, value in result.items()):
        raise ValueError("Independent steady residual budget exceeded")
    return result


def stationary_metrics(folder, budget):
    folder = Path(folder)
    spec = json.loads((folder / "input.json").read_text())
    base = json.loads(Path(budget["base_budget_file"]).read_text())
    initial, rhs, _, vessel, data, offset = system(spec)
    rates = rhs(0, initial, spec["controls"])
    n = data["n"]
    ledgers = [*range(2 + 2 * n, offset), *range(offset + 7, offset + 14)]
    metrics = {key: 0.0 for key in base["limits"]}
    for row in json.loads((folder / "trajectory.json").read_text()):
        # Exact steady physics: constant physical state and linear counters.
        # Small independently measured algebraic residuals are bounded above.
        exact = initial.copy()
        exact[ledgers] += rates[ledgers] * row["time_s"]
        compare_reference(recorded_state(row, data), exact, metrics, data, offset, vessel)
        compare_subset(
            row, observables(exact, spec["controls"], data, offset, vessel, spec), metrics
        )
    if any(value > base["limits"][key] for key, value in metrics.items()):
        raise ValueError("Designed state drift or counter error exceeds stationary budget")
    return metrics


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "qualification.json").read_text())
    budget = json.loads(BUDGET.read_text())
    if (
        report["signature"] != signature()
        or json.loads((folder / "budget.json").read_text()) != budget
    ):
        raise ValueError("Steady-design qualification signature mismatch")
    expected = {(n, p): (flow, opening, request) for n, p, flow, opening, request in matrix(budget)}
    identities = [(r["name"], r["policy"]) for r in report["cases"]]
    if (
        len(set(identities)) != len(identities)
        or set(identities) != set(expected)
        or report["passed"] is not True
        or report["integrations"] != len(expected)
        or report["case_families"] != 7
    ):
        raise ValueError("Incomplete/duplicated steady-design qualification")
    paths = set()
    for record in report["cases"]:
        path = Path(record["folder"]).resolve()
        if not path.is_relative_to(folder.resolve()) or path in paths:
            raise ValueError("Shared or external steady-design evidence")
        paths.add(path)
        flow, opening, request = expected[(record["name"], record["policy"])]
        if json.loads((path / "request.json").read_text()) != request:
            raise ValueError("Steady-design request substitution")
        spec, design = design_steady_primary_steam(
            request, primary_mass_flow_kg_s=flow, valve_opening=opening
        )
        if (
            json.loads((path / "design.json").read_text()) != design
            or json.loads((path / "run/input.json").read_text()) != spec
        ):
            raise ValueError("Steady-design chosen boundaries/state substitution")
        residuals = steady_residuals(spec, budget)
        stationary = stationary_metrics(path / "run", budget)
        fresh = assess(path / "run")
        if (
            not fresh["accepted"]
            or fresh["termination"] != "completed"
            or any(fresh[k] != record[k] for k in fresh)
        ):
            raise ValueError("Steady-design independent trajectory replay mismatch")
        if (
            record["independent_steady_residuals"] != residuals
            or record["stationary_metrics"] != stationary
        ):
            raise ValueError("Steady-design residual/stationarity replay mismatch")
        print(
            json.dumps(dict(name=record["name"], policy=record["policy"], audited=True)), flush=True
        )
    if report["signature"] != signature():
        raise ValueError("Steady-design source changed during audit")
    return dict(
        schema_version=1,
        passed=True,
        integrations=len(expected),
        case_families=7,
        signature=signature(),
        qualification_sha256=sha(folder / "qualification.json"),
        long_stationary_horizon_s=budget["long_stationary_horizon_s"],
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        write(args.output, audit(args.audit))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(BUDGET.read_text())
    captured = signature()
    write(args.output / "budget.json", budget)
    records = []
    for name, policy, flow, opening, request in matrix(budget):
        if signature() != captured:
            raise ValueError("Steady-design source changed during qualification")
        folder = args.output / f"{name}_{policy}"
        folder.mkdir()
        spec, design = design_steady_primary_steam(
            request, primary_mass_flow_kg_s=flow, valve_opening=opening
        )
        write(folder / "request.json", request)
        write(folder / "design.json", design)
        residuals = steady_residuals(spec, budget)
        run_primary_steam_experiment(spec, folder / "run")
        result = assess(folder / "run")
        if result["termination"] != "completed":
            raise ValueError("Designed steady operating horizon not completed")
        result.update(
            name=name,
            policy=policy,
            folder=str(folder),
            independent_steady_residuals=residuals,
            stationary_metrics=stationary_metrics(folder / "run", budget),
        )
        records.append(result)
        print(json.dumps(dict(name=name, policy=policy, accepted=result["accepted"])), flush=True)
    if signature() != captured:
        raise ValueError("Steady-design source changed before verdict")
    write(
        args.output / "qualification.json",
        dict(
            schema_version=1,
            passed=all(r["accepted"] for r in records),
            cases=records,
            integrations=len(records),
            case_families=7,
            signature=captured,
            validated=False,
            plant_referenced=False,
        ),
    )
    passed = all(r["accepted"] for r in records)
    print(json.dumps(dict(passed=passed, integrations=len(records), case_families=7)), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
