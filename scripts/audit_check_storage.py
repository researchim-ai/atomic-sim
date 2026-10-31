"""Recompute storage acceptance and common-grid temporal differences from bundles."""

import argparse
import json
from pathlib import Path

from scripts.verify_check_storage import assess, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    folder = Path(folder)
    r = json.loads((folder / "verification.json").read_text())
    budget = json.loads((folder / "budget.json").read_text())
    if (
        r["source_sha256"] != _source_fingerprint()
        or r["script_sha256"] != sha("scripts/verify_check_storage.py")
        or r["reference_sha256"] != sha("scripts/check_storage_references.py")
        or r["budget_sha256"] != sha("validation/ai05_check_storage_budget_v1.json")
        or budget != json.loads(Path("validation/ai05_check_storage_budget_v1.json").read_text())
    ):
        raise ValueError("Storage source, reference or budget hash mismatch")
    if r["coefficient_reference_sha256"] != sha("scripts/storage_references.py"):
        raise ValueError("Coefficient reference hash mismatch")
    if not r["all_cases_accepted"] or r["integrations"] != len(r["cases"]):
        raise ValueError("Incomplete storage series")
    for record in r["cases"]:
        path = Path(record["folder"])
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Storage bundle outside evidence folder")
        fresh = assess(path, budget)
        if not fresh["accepted"] or any(
            fresh[k] != record[k] for k in ("metrics", "events", "restart_energy_abs_j")
        ):
            raise ValueError("Storage metrics or verdict differs from replay")
    names = sorted({c["name"] for c in r["cases"]})
    comparisons = []
    for name in names:
        cases = {c["policy"]: c for c in r["cases"] if c["name"] == name}
        if not {"base", "tight"}.issubset(cases):
            raise ValueError("Missing storage time refinement")
        a = json.loads((Path(cases["base"]["folder"]) / "trajectory.json").read_text())
        b = json.loads((Path(cases["tight"]["folder"]) / "trajectory.json").read_text())
        if [(x["time_s"], x["phase"]) for x in a] != [(x["time_s"], x["phase"]) for x in b]:
            raise ValueError("Different storage comparison grids")
        metrics = dict(
            omega_abs_rad_s=0.0,
            mass_flow_abs_kg_s=0.0,
            transfer_mass_abs_kg=0.0,
            pressure_abs_pa=0.0,
            temperature_abs_k=0.0,
            boundary_work_abs_j=0.0,
        )
        for x, y in zip(a, b):
            metrics["omega_abs_rad_s"] = max(
                metrics["omega_abs_rad_s"], abs(x["omega_rad_s"] - y["omega_rad_s"])
            )
            metrics["mass_flow_abs_kg_s"] = max(
                metrics["mass_flow_abs_kg_s"], abs(x["mass_flow_kg_s"] - y["mass_flow_kg_s"])
            )
            metrics["transfer_mass_abs_kg"] = max(
                metrics["transfer_mass_abs_kg"], abs(x["transfer_mass_kg"] - y["transfer_mass_kg"])
            )
            for key, metric in (
                ("pressure_pa", "pressure_abs_pa"),
                ("liquid_temperature_k", "temperature_abs_k"),
                ("gas_temperature_k", "temperature_abs_k"),
                ("gas_boundary_work_j", "boundary_work_abs_j"),
            ):
                metrics[metric] = max(
                    metrics[metric], max(abs(x[key][n] - y[key][n]) for n in x[key])
                )
        if any(value > budget[key] for key, value in metrics.items()):
            raise ValueError("Storage time refinement exceeds budget")
        comparisons.append(dict(name=name, metrics=metrics))
    if len(names) != r["case_families"]:
        raise ValueError("Storage family count mismatch")
    return dict(
        schema_version=1,
        passed=True,
        source_sha256=_source_fingerprint(),
        verification_sha256=sha(folder / "verification.json"),
        audit_script_sha256=sha(__file__),
        integrations=len(r["cases"]),
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
    r = audit(args.folder)
    write(args.output, r)
    print(json.dumps(dict(passed=r["passed"], integrations=r["integrations"])))


if __name__ == "__main__":
    main()
