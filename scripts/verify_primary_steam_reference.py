"""Four independent IVP precisions; confirm selected reference on a dense clock."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.primary_steam_references import REFERENCE_POLICY, observables, reference
from scripts.qualify_primary_steam import signature
from scripts.verify_primary_steam import BUDGET, compare_reference, compare_subset
from scripts.verify_turbine import sha
from sdk.network_experiment import write

POLICIES = (
    dict(rtol=2e-12, mechanical_atol=1e-10, max_step_s=0.025),
    dict(rtol=2e-13, mechanical_atol=1e-12, max_step_s=0.0125),
    REFERENCE_POLICY,
    dict(rtol=2.5e-14, mechanical_atol=1e-13, max_step_s=0.003125),
)
EXCLUDED = {
    "energy_balance_absolute_j",
    "reference_energy_balance_absolute_j",
    "power_balance_absolute_w",
    "boundary_time_absolute_s",
    "stored_energy_absolute_j",
    "volume_absolute_m3",
    "flow_acceleration_absolute_kg_s2",
}


def verify(folder):
    folder = Path(folder)
    spec = json.loads((folder / "input.json").read_text())
    budget = json.loads(BUDGET.read_text())
    captured = signature()
    solutions = [reference(spec, precision=policy) for policy in POLICIES]
    if any(boundary is not None for _, boundary, _ in solutions):
        raise ValueError(
            "This reference-convergence scope requires a completed interior trajectory"
        )
    clock = sorted(
        set(
            [
                *map(float, np.linspace(0, spec["duration_s"], 1001)),
                *[event["t"] for event in spec["timeline"]],
            ]
        )
    )
    pairs = []
    for index, (left, right) in enumerate(zip(solutions, solutions[1:])):
        a, _, details = left
        b, _, _ = right
        _, _, vessel, data, offset, _ = details
        metrics = {key: 0.0 for key in budget["limits"]}
        for t in clock:
            controls = spec["controls"]
            for event in spec["timeline"]:
                if event["t"] <= t:
                    controls = event["controls"]
            x, y = a(t), b(t)
            compare_reference(x, y, metrics, data, offset, vessel)
            compare_subset(
                observables(x, controls, data, offset, vessel, spec),
                observables(y, controls, data, offset, vessel, spec),
                metrics,
            )
        compared = {key: float(value) for key, value in metrics.items() if key not in EXCLUDED}
        fraction = max(value / budget["limits"][key] for key, value in compared.items())
        pairs.append(
            dict(
                from_policy=index,
                to_policy=index + 1,
                maximum_differences=compared,
                maximum_fraction_of_admission_budget=fraction,
            )
        )
    if captured != signature():
        raise ValueError("Reference-convergence inputs changed during study")
    return dict(
        schema_version=1,
        passed=pairs[-1]["maximum_fraction_of_admission_budget"] <= 0.25,
        signature=captured,
        driver_sha256=sha(__file__),
        input_sha256=sha(folder / "input.json"),
        policies=POLICIES,
        compared_clock_points=len(clock),
        comparisons=pairs,
        selected_policy=REFERENCE_POLICY,
        required_confirmation_fraction=0.25,
        scope="State, inventory, counters and derived powers/enthalpy faces; conservation and terminal events are checked separately by the full assembly auditor",
        observed_order=None,
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.folder)
    write(args.output, result)
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                confirmation_fraction=result["comparisons"][-1][
                    "maximum_fraction_of_admission_budget"
                ],
            )
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
