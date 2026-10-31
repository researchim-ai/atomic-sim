"""Recompute inventories, liquid admissibility and admission from saved raw states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from sdk.euler_qualification import EulerQualification


def audit(folder):
    gate = EulerQualification(folder)
    budget = json.loads((folder / "budget.json").read_text())
    runs = []
    for path in sorted(folder.glob("*/trajectory.json")):
        spec = json.loads((path.parent / "input.json").read_text())
        model = EulerLiquidChannel(EulerConfig(**spec["config"]))
        saved = json.loads(path.read_text())
        report = json.loads((path.parent / "report.json").read_text())
        minima = dict(mass_kg=float("inf"), internal_energy_j=float("inf"))
        mach = 0.0
        maximum_tv = 0.0
        residuals = dict(energy_abs_j=0.0, momentum_abs_kg_m_s=0.0, mass_relative=0.0)
        for row in saved:
            y = np.array(row["state"])
            q = y[: 3 * model.config.cells].reshape(-1, 3)
            states = [model.cell(cell) for cell in q]
            minima["mass_kg"] = min(minima["mass_kg"], float(min(q[:, 0])))
            internal = q[:, 2] - q[:, 1] ** 2 / (2 * q[:, 0])
            minima["internal_energy_j"] = min(minima["internal_energy_j"], float(min(internal)))
            mach = max(mach, *(abs(s[1]) / s[3] for s in states))
            temperatures = np.array([s[0].T_k for s in states])
            maximum_tv = max(maximum_tv, float(sum(abs(np.diff(temperatures)))))
            mass, momentum, energy = model.inventories(y)
            recomputed = dict(
                mass_relative=abs(mass - model.initial_mass_kg - y[-3]) / model.initial_mass_kg,
                momentum_abs_kg_m_s=abs(momentum - model.initial_momentum_kg_m_s - y[-2]),
                energy_abs_j=abs(energy - model.initial_energy_j - y[-1]),
            )
            for key, value in recomputed.items():
                residuals[key] = max(residuals[key], float(value))
        if residuals != report["balances"]:
            raise ValueError("Saved report balances disagree with raw conserved states")
        passed = report["status"] == "completed" and all(
            v <= budget[k] for k, v in residuals.items()
        )
        if passed != report["accepted"]:
            raise ValueError("Recomputed acceptance mismatch")
        runs.append(
            dict(
                case=path.parent.name,
                accepted=passed,
                minima=minima,
                max_mach=mach,
                sampled_temperature_tv_k=maximum_tv,
                balances=residuals,
            )
        )
    admitted = []
    for entry in gate.matrix["entries"]:
        for cells in entry.get("qualified_cells", []):
            spec = gate.approved_input(entry["id"], cells)
            gate.verify_request(entry["id"], cells, **spec)
            admitted.append(dict(case=entry["id"], cells=cells))
    return dict(
        passed=True,
        matrix_sha256=hashlib.sha256((folder / "qualification.json").read_bytes()).hexdigest(),
        source_sha256=gate.matrix["source_sha256"],
        runs=runs,
        admitted=admitted,
        validated=False,
        plant_referenced=False,
        note="Sampled admissibility and balances; no global positivity/TVD proof",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.folder)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                passed=result["passed"], runs=len(result["runs"]), admitted=len(result["admitted"])
            )
        )
    )


if __name__ == "__main__":
    main()
