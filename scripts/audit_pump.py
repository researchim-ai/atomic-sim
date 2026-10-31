"""Recompute pump/valve bundle verdicts, raw energy and temporal comparisons."""

import argparse
import json
from pathlib import Path

from scripts.verify_pump import assess, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads((folder / "budget.json").read_text())
    valved = "reference_sha256" in report
    budget_path = (
        "validation/ai05_valve_budget_v1.json" if valved else "validation/ai05_pump_budget_v1.json"
    )
    script_path = "scripts/verify_valve.py" if valved else "scripts/verify_pump.py"
    if valved and report["reference_sha256"] != sha("scripts/verify_pump.py"):
        raise ValueError("Valve reference implementation hash mismatch")
    if (
        report["source_sha256"] != _source_fingerprint()
        or report["budget_sha256"] != sha(budget_path)
        or budget != json.loads(Path(budget_path).read_text())
        or report["script_sha256"] != sha(script_path)
    ):
        raise ValueError("Pump source, budget or reference implementation hash mismatch")
    if not report["all_cases_accepted"] or report["integrations"] != len(report["cases"]):
        raise ValueError("Incomplete pump verification")
    comparisons = []
    for record in report["cases"]:
        path = Path(record["folder"])
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Pump case outside evidence folder")
        fresh = assess(path, budget)
        if not fresh["accepted"] or any(
            fresh[k] != record[k] for k in ("metrics", "events", "restart_energy_abs_j")
        ):
            raise ValueError("Pump acceptance metrics differ from replay")
    names = sorted({r["name"] for r in report["cases"]})
    for name in names:
        records = {r["policy"]: r for r in report["cases"] if r["name"] == name}
        if not {"base", "tight"}.issubset(records):
            raise ValueError("Missing temporal refinement")
        a = json.loads((Path(records["base"]["folder"]) / "trajectory.json").read_text())
        b = json.loads((Path(records["tight"]["folder"]) / "trajectory.json").read_text())
        if [(r["time_s"], r["phase"]) for r in a] != [(r["time_s"], r["phase"]) for r in b]:
            raise ValueError("Different temporal output grid")
        metrics = dict(
            mass_flow_abs_kg_s=0.0, omega_abs_rad_s=0.0, temperature_abs_k=0.0, integral_abs_j=0.0
        )
        if valved:
            metrics["opening_abs"] = 0.0
        for x, y in zip(a, b):
            if valved:
                metrics["opening_abs"] = max(
                    metrics["opening_abs"], abs(x["valve_opening"] - y["valve_opening"])
                )
            metrics["mass_flow_abs_kg_s"] = max(
                metrics["mass_flow_abs_kg_s"],
                abs(x["loop"]["mass_flow_kg_s"] - y["loop"]["mass_flow_kg_s"]),
            )
            metrics["omega_abs_rad_s"] = max(
                metrics["omega_abs_rad_s"], abs(x["omega_rad_s"] - y["omega_rad_s"])
            )
            for key in ("fluid_temperature_k", "wall_temperature_k"):
                metrics["temperature_abs_k"] = max(
                    metrics["temperature_abs_k"],
                    max(abs(x["loop"][key][n] - y["loop"][key][n]) for n in x["loop"][key]),
                )
            for key in ("motor_work_j", "bearing_heat_j", "leakage_heat_j"):
                metrics["integral_abs_j"] = max(metrics["integral_abs_j"], abs(x[key] - y[key]))
        if any(v > budget[k] for k, v in metrics.items()):
            raise ValueError("Pump temporal refinement exceeds budget")
        comparisons.append(dict(name=name, metrics=metrics))
    return dict(
        schema_version=1,
        passed=True,
        source_sha256=_source_fingerprint(),
        verification_sha256=sha(folder / "verification.json"),
        audit_script_sha256=sha(__file__),
        integrations=len(report["cases"]),
        case_families=len(names),
        temporal_comparisons=comparisons,
        validated=False,
        plant_referenced=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.folder)
    write(args.output, result)
    print(json.dumps(dict(passed=result["passed"], integrations=result["integrations"])))


if __name__ == "__main__":
    main()
