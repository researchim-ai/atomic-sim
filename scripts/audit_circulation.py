"""Recompute circulation verdicts, dimensional initial residuals and energy data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import circulation_references as refs
from scripts import verify_circulation as verifier
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        path = (folder / name).resolve()
        if (
            not path.is_relative_to(folder.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Circulation artifact path/hash mismatch")
    report = json.loads((folder / "verification.json").read_text())
    budget = json.loads((folder / "budget.json").read_text())
    if report["source_sha256"] != _source_fingerprint():
        raise ValueError("Runtime source mismatch")
    if (
        report["script_sha256"] != hashlib.sha256(Path(verifier.__file__).read_bytes()).hexdigest()
        or report["reference_sha256"]
        != hashlib.sha256(Path(refs.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Verification/reference source mismatch")
    kinds = {spec["name"]: kind for spec, kind in verifier.cases()}
    entries = []
    all_rows = {}
    maximum_energy = 0.0
    maximum_rounding = 0.0
    for case in report["cases"]:
        family = case["id"][4:] if case["id"].startswith("bdf_") else case["id"]
        kind = kinds[family]
        runs = [case["base"], case["tight"]] if "base" in case else [case]
        for run in runs:
            path = folder / Path(run["path"]).name
            spec = json.loads((path / "input.json").read_text())
            recomputed, rows = verifier.assess(spec, kind, path, budget)
            if recomputed["checks"] != run["checks"] or recomputed["accepted"] != run["accepted"]:
                raise ValueError("Recomputed circulation verdict differs")
            for key, value in run["metrics"].items():
                if not np.isclose(value, recomputed["metrics"][key], rtol=1e-12, atol=1e-14):
                    raise ValueError("Recomputed metric differs: " + key)
            _, model = verifier.load(path)
            all_rows[path.name] = rows
            initial = model.snapshot()
            tf0 = np.array([initial["initial_temperature_k"][n] for n in model.names])
            tw0 = np.array([initial["initial_wall_temperature_k"][n] for n in model.names])
            kinetic0 = (
                model.inertia_total * initial["initial_mass_flow_kg_s"] ** 2 / (2 * model.rho)
            )
            for row in rows:
                tf = np.array([row["fluid_temperature_k"][n] for n in model.names])
                tw = np.array([row["wall_temperature_k"][n] for n in model.names])
                change = float(
                    model.capacity @ (tf - tf0)
                    + model.wall_capacity @ (tw - tw0)
                    + row["kinetic_energy_j"]
                    - kinetic0
                )
                defect = change - row["external_head_work_j"] - row["external_heat_j"]
                maximum_energy = max(maximum_energy, abs(defect))
                if abs(defect) > budget["energy_abs_j"]:
                    raise ValueError("Energy recomputed from physical temperatures exceeds budget")
                # Absolute temperatures round small increments on serialization.
                # This bound checks consistency only; the energy budget above is unchanged.
                rounding = abs(change - row["stored_energy_change_j"])
                bound = (
                    2
                    * np.finfo(float).eps
                    * (model.capacity @ abs(tf) + model.wall_capacity @ abs(tw))
                    + 1e-12
                )
                if rounding > bound:
                    raise ValueError("Stored relative energy disagrees beyond temperature roundoff")
                maximum_rounding = max(maximum_rounding, rounding)
            initialization = json.loads((path / "initialization.json").read_text())
            if initialization["mode"] == "steady":
                if (
                    abs(initialization["momentum_rate_pa"]) > budget["steady_momentum_rate_abs_pa"]
                    or max(
                        abs(v)
                        for field in ("fluid_energy_rates_w", "wall_energy_rates_w")
                        for v in initialization[field]
                    )
                    > budget["steady_thermal_rate_abs_w"]
                ):
                    raise ValueError("Dimensional steady initialization residual failed")
            entries.append(dict(id=path.name, passed=True, samples=len(rows)))
        if "base" in case:
            a = all_rows[Path(case["base"]["path"]).name]
            b = all_rows[Path(case["tight"]["path"]).name]
            flow = max(abs(x["mass_flow_kg_s"] - y["mass_flow_kg_s"]) for x, y in zip(a, b))
            temp = max(
                abs(x[field][n] - y[field][n])
                for x, y in zip(a, b)
                for field in ("fluid_temperature_k", "wall_temperature_k")
                for n in x[field]
            )
            if (
                flow != case["time_error"]["time_flow_abs_kg_s"]
                or temp != case["time_error"]["time_temperature_abs_k"]
            ):
                raise ValueError("Temporal metric mismatch")
            if case["accepted"] != (
                case["base"]["accepted"]
                and case["tight"]["accepted"]
                and flow <= budget["time_flow_abs_kg_s"]
                and temp <= budget["time_temperature_abs_k"]
            ):
                raise ValueError("Temporal acceptance mismatch")
    if len(entries) != report["integrations"]:
        raise ValueError("Integration count mismatch")
    return dict(
        passed=True,
        source_sha256=report["source_sha256"],
        verification_sha256=hashlib.sha256((folder / "verification.json").read_bytes()).hexdigest(),
        integrations=len(entries),
        cases=entries,
        maximum_recomputed_energy_residual_j=float(maximum_energy),
        maximum_energy_temperature_roundoff_j=float(maximum_rounding),
        all_cases_accepted=report["all_cases_accepted"],
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
    print(json.dumps(dict(passed=True, integrations=result["integrations"])))


if __name__ == "__main__":
    main()
