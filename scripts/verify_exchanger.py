"""Fixed-scope thermal exchanger acceptance; independently replayable bundles."""

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from physics.thermal.two_stream_exchanger import TwoStreamExchanger
from scripts.exchanger_references import cocurrent_equal_capacity_outlets, reference
from sdk.exchanger_experiment import prepare, run_exchanger_experiment
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write

BUDGET = Path("validation/ai05_exchanger_budget_v1.json")


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
        raise ValueError("Nonfinite exchanger evidence value")


def scenarios():
    base = json.loads(Path("scenarios/ai05_exchanger_counterflow.json").read_text())
    cases = []
    for name in (
        "counterflow",
        "cocurrent",
        "reverse",
        "stagnant",
        "one_stopped",
        "ambient_heater",
        "unequal_cells",
        "steady",
        "switching",
        "initial_command",
    ):
        spec = deepcopy(base)
        spec.update(name=name, duration_s=60, step_s=2, timeline=[])
        if name == "cocurrent":
            spec["controls"]["b_mass_flow_kg_s"] *= -1
        elif name == "reverse":
            for side in ("a", "b"):
                spec["controls"][f"{side}_mass_flow_kg_s"] *= -1
        elif name == "stagnant":
            spec["controls"].update(a_mass_flow_kg_s=0, b_mass_flow_kg_s=0)
        elif name == "one_stopped":
            spec["controls"]["a_mass_flow_kg_s"] = 0
        elif name == "ambient_heater":
            spec["model"].update(ambient_ua_w_k=[5, 10, 15], wall_power_w=[100, 200, 300])
        elif name == "unequal_cells":
            spec["model"].update(
                a_cell_mass_kg=[1, 2, 4],
                b_cell_mass_kg=[4, 3, 1],
                wall_capacity_j_k=[3000, 5000, 8000],
                a_wall_ua_w_k=[100, 300, 700],
            )
        elif name == "steady":
            spec["initialization"] = dict(mode="steady")
        elif name in ("switching", "initial_command"):
            spec["timeline"] = deepcopy(base["timeline"][:2])
            if name == "initial_command":
                spec["timeline"][0]["t"] = 0
        cases.append((name, spec))
    for mode in ("antisymmetric", "symmetric", "tank"):
        spec = deepcopy(base)
        spec.update(name=mode, duration_s=30, step_s=2, timeline=[])
        cfg = spec["model"]
        cfg["cells"] = 1
        for key, value in cfg.items():
            if isinstance(value, list):
                cfg[key] = value[:1]
        cfg.update(
            a_cp_j_kg_k=1000,
            b_cp_j_kg_k=1000,
            a_cell_mass_kg=[1],
            b_cell_mass_kg=[1],
            wall_capacity_j_k=[1000],
            a_wall_ua_w_k=[100],
            b_wall_ua_w_k=[100],
        )
        spec["controls"].update(a_mass_flow_kg_s=0, b_mass_flow_kg_s=0)
        temperatures = (330, 330, 300) if mode == "symmetric" else (330, 300, 315)
        spec["initialization"]["temperature_k"] = dict(
            zip(("a", "b", "wall"), ([v] for v in temperatures))
        )
        if mode == "tank":
            spec["controls"]["a_mass_flow_kg_s"] = 0.4
            cfg["a_wall_ua_w_k"] = [0]
        cases.append((mode, spec))
    return cases


def matrix():
    for name, spec in scenarios():
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
        "initialization.json",
        "trajectory.json",
        "checkpoint.json",
        "report.json",
    }
    if manifest["source_sha256"] != _source_fingerprint() or not required.issubset(
        manifest["artifacts"]
    ):
        raise ValueError("Exchanger source/artifact manifest mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError("Exchanger artifact checksum/path mismatch")
    spec = json.loads((folder / "input.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = json.loads((folder / "trajectory.json").read_text())
    require_finite(rows)
    require_finite(report)
    expected_rows = [(0.0, "initial", spec["controls"])]
    active, command_index = spec["controls"], 0
    expected_events = []
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
    if [(r["time_s"], r["phase"], r["controls"]) for r in rows] != expected_rows:
        raise ValueError("Exchanger trajectory grid/control history mismatch")
    if report["events"] != expected_events or report["completed_time_s"] != spec["duration_s"]:
        raise ValueError("Exchanger report horizon/control history mismatch")
    n = spec["model"]["cells"]
    at = reference(spec)
    cfg = spec["model"]
    capacity = np.concatenate(
        (
            np.array(cfg["a_cell_mass_kg"]) * cfg["a_cp_j_kg_k"],
            np.array(cfg["b_cell_mass_kg"]) * cfg["b_cp_j_kg_k"],
            cfg["wall_capacity_j_k"],
        )
    )
    initial = at(0)[: 3 * n]
    metrics = dict(
        temperature_absolute_k=0.0,
        energy_absolute_j=0.0,
        port_power_absolute_w=0.0,
        reference_energy_absolute_j=0.0,
        restart_temperature_absolute_k=0.0,
        analytic_temperature_absolute_k=0.0,
    )
    if not rows or rows[0]["time_s"] != 0 or rows[-1]["time_s"] != spec["duration_s"]:
        raise ValueError("Incomplete exchanger trajectory")
    for row in rows:
        stamp = row["time_s"]
        ref = at(stamp)
        actual = np.concatenate([row["temperature_k"][k] for k in ("a", "b", "wall")])
        metrics["temperature_absolute_k"] = max(
            metrics["temperature_absolute_k"], float(max(abs(actual - ref[: 3 * n])))
        )
        metrics["energy_absolute_j"] = max(
            metrics["energy_absolute_j"],
            abs(row["energy_residual_j"]),
            max(abs(v) for cells in row["cell_energy_residual_j"] for v in cells),
        )
        metrics["port_power_absolute_w"] = max(
            metrics["port_power_absolute_w"], abs(row["port_power_residual_w"])
        )
        metrics["reference_energy_absolute_j"] = max(
            metrics["reference_energy_absolute_j"],
            abs(float(capacity @ (actual - initial) - sum(ref[-4:]))),
            float(max(abs(capacity * (actual - initial) - ref[3 * n : 6 * n]))),
        )
        name = spec["name"]
        if name in ("antisymmetric", "symmetric", "tank"):
            if name == "antisymmetric":
                analytic = 315 + np.array([15, -15, 0]) * np.exp(-0.1 * stamp)
            elif name == "symmetric":
                analytic = 320 + np.array([10, 10, -20]) * np.exp(-0.3 * stamp)
            else:
                # Only a is uncoupled; b and wall continue exchanging heat.
                analytic = [370 - 40 * np.exp(-0.4 * stamp)]
                actual = actual[:1]
            metrics["analytic_temperature_absolute_k"] = max(
                metrics["analytic_temperature_absolute_k"], float(max(abs(actual - analytic)))
            )
    snapshot = json.loads((folder / "checkpoint.json").read_text())
    restored = TwoStreamExchanger.from_snapshot(snapshot)
    if restored.observation() != {k: v for k, v in rows[-1].items() if k != "phase"}:
        raise ValueError("Exchanger checkpoint differs from trajectory")
    extended = deepcopy(spec)
    extended["duration_s"] += 1
    ref = reference(extended)(extended["duration_s"])[: 3 * n]
    continued = restored.advance(1)
    actual = np.concatenate([continued["temperature_k"][k] for k in ("a", "b", "wall")])
    metrics["restart_temperature_absolute_k"] = float(max(abs(actual - ref)))
    accepted = (
        report["status"] == "completed"
        and report["numerical_acceptance"]
        and all(np.isfinite(v) and v <= budget[k] for k, v in metrics.items())
    )
    return dict(accepted=bool(accepted), metrics=metrics)


def spatial_refinement(budget):
    records = []
    for n in budget["spatial_refinement_cells"]:
        spec = deepcopy(scenarios()[0][1])
        spec["model"].update(
            cells=n,
            a_cp_j_kg_k=4000,
            b_cp_j_kg_k=4000,
            a_cell_mass_kg=[6 / n] * n,
            b_cell_mass_kg=[6 / n] * n,
            wall_capacity_j_k=[15000 / n] * n,
            a_wall_ua_w_k=[1200 / n] * n,
            b_wall_ua_w_k=[1200 / n] * n,
            ambient_ua_w_k=[0] * n,
            wall_power_w=[0] * n,
        )
        spec["controls"].update(a_mass_flow_kg_s=0.5, b_mass_flow_kg_s=0.5)
        spec["initialization"] = dict(mode="steady")
        model = prepare(spec)[1]
        observed = model.observation()
        actual = np.array([observed["ports"][s]["outlet_temperature_k"] for s in ("a", "b")])
        expected = cocurrent_equal_capacity_outlets(spec)
        residual = max(
            abs(v) for row in model.initialization_report()["energy_rate_residual_w"] for v in row
        )
        records.append(
            dict(
                cells=n,
                outlet_temperature_k=actual.tolist(),
                continuum_temperature_k=expected.tolist(),
                error_k=float(max(abs(actual - expected))),
                steady_residual_w=residual,
            )
        )
    ratios = [a["error_k"] / b["error_k"] for a, b in zip(records, records[1:])]
    passed = all(r >= budget["spatial_min_error_ratio"] for r in ratios) and all(
        r["steady_residual_w"] <= budget["steady_residual_absolute_w"] for r in records
    )
    return dict(
        passed=bool(passed),
        levels=records,
        error_ratios=ratios,
        scope="Cocurrent equal-capacity-rate continuum reference; not a general mesh qualification",
    )


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
        run_exchanger_experiment(spec, folder)
        record = assess(folder, budget)
        record.update(name=name, policy=policy, folder=str(folder))
        records.append(record)
    spatial = spatial_refinement(budget)
    passed = all(r["accepted"] for r in records) and spatial["passed"]
    result = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        script_sha256=sha(__file__),
        reference_sha256=sha("scripts/exchanger_references.py"),
        budget_sha256=sha(BUDGET),
        all_cases_accepted=passed,
        cases=records,
        integrations=len(records),
        case_families=len(scenarios()),
        spatial_refinement=spatial,
        validated=False,
        plant_referenced=False,
    )
    write(args.output / "verification.json", result)
    print(json.dumps(dict(passed=passed, integrations=len(records))))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
