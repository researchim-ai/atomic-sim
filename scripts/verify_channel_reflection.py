"""Audit reflection symmetry of complete total-energy pipe experiment bundles."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_isobaric_runs import load

# Fixed audit tolerances; reflection is a discrete symmetry, not a mesh study.
TOLERANCES = dict(flow_kg_s=1e-6, pressure_pa=1.0, temperature_k=1e-5, energy_j=1e-3)


def verify(forward, reverse):
    a, b = (load(folder, 8.0) for folder in (forward, reverse))
    if any(x[0]["engine"] != "compatible-total-energy-channel" for x in (a, b)):
        raise ValueError("Reflection audit requires compatible total-energy engines")
    if a[4]["source_sha256"] != b[4]["source_sha256"]:
        raise ValueError("Runtime source fingerprints differ")
    if a[0]["solver"] != b[0]["solver"] or set(a[2]) != set(b[2]):
        raise ValueError("Reflection audit requires identical solver and sample times")
    expected = dict(a[0]["hydraulics"])
    for quantity in ("pressure_pa", "temperature_k"):
        expected["left_" + quantity], expected["right_" + quantity] = (
            expected["right_" + quantity],
            expected["left_" + quantity],
        )
    expected["initial_flow_kg_s"] *= -1
    for key in (
        "junction_lengths_m",
        "junction_loss_coefficients",
        "initial_pressure_profile_pa",
        "initial_temperature_profile_k",
        "initial_wall_temperature_profile_k",
    ):
        if key in expected:
            expected[key] = list(reversed(expected[key]))
    if expected != b[0]["hydraulics"]:
        raise ValueError("Hydraulic inputs are not exact reflections")
    groups = dict(
        pressure_pa=("min_pressure_pa", "max_pressure_pa", "midpoint_pressure_pa"),
        temperature_k=(
            "min_temperature_k",
            "max_temperature_k",
            "mean_wall_temperature_k",
            "min_wall_temperature_k",
            "max_wall_temperature_k",
        ),
        energy_j=(
            "stored_energy_j",
            "kinetic_energy_j",
            "thermal_energy_j",
            "pressure_work_integral_j",
            "dissipation_integral_j",
            "density_work_integral_j",
        ),
    )
    measured = {
        key: max(abs(row[field] - b[2][t][field]) for t, row in a[2].items() for field in fields)
        for key, fields in groups.items()
    }
    measured["flow_kg_s"] = max(
        abs(row[left] + b[2][t][right])
        for t, row in a[2].items()
        for left, right in (
            ("inlet_mass_flow_kg_s", "outlet_mass_flow_kg_s"),
            ("outlet_mass_flow_kg_s", "inlet_mass_flow_kg_s"),
        )
    )
    # Check every final physical component, not just extrema of the CSV.
    pa, pb = (
        json.loads((Path(folder) / "checkpoint.json").read_text()) for folder in (forward, reverse)
    )
    ya, yb = (np.array(p["state"], dtype=float) for p in (pa, pb))
    n = a[0]["hydraulics"]["cells"]
    reflected = np.concatenate(
        (
            ya[: 2 * n].reshape(n, 2)[::-1].ravel(),
            -ya[2 * n : 3 * n + 1][::-1],
            ya[3 * n + 1 : 4 * n + 1][::-1],
            ya[4 * n + 1 :],
        )
    )
    scale = np.maximum(np.abs(reflected), 1.0)
    final_relative = float(max(abs(yb - reflected) / scale))
    checks = {key: value <= TOLERANCES[key] for key, value in measured.items()}
    checks["final_state_scaled"] = final_relative <= 1e-6
    return dict(
        passed=all(checks.values()),
        checks=checks,
        measured=measured,
        tolerances=TOLERANCES,
        final_state_scaled=final_relative,
        source_sha256=a[4]["source_sha256"],
        validated=False,
        plant_referenced=False,
        scope="Discrete reflection symmetry for an 8-second reduced liquid pipe; not physical validation",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("forward", type=Path)
    p.add_argument("reverse", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = verify(args.forward, args.reverse)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
