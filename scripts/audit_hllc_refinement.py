"""Recompute refined contact qualification from hashed raw trajectories."""

import argparse
import json
from pathlib import Path

from core.status import DomainError
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from scripts.qualify_hllc import (
    exact_over_time,
    observed_order,
    profile_difference,
    time_difference,
)
from sdk.euler_qualification import EulerQualification


def audit(folder):
    gate = EulerQualification(folder)
    budget = json.loads((folder / "budget.json").read_text())

    def load(name):
        path = folder / name
        spec = json.loads((path / "input.json").read_text())
        return dict(
            model=EulerLiquidChannel(EulerConfig(**spec["config"])),
            rows=json.loads((path / "trajectory.json").read_text()),
            report=json.loads((path / "report.json").read_text()),
        )

    entries = []
    for sign, velocity, suffix in (
        (sign, velocity, suffix)
        for sign, velocity in (("negative", -1.0), ("positive", 1.0))
        for suffix in ("refined", "asymptotic", "time_resolved")
        if any(e["id"] == f"step_{sign}_{suffix}" for e in gate.matrix["entries"])
    ):
        case = f"step_{sign}_{suffix}"
        row = next(e for e in gate.matrix["entries"] if e["id"] == case)
        coarse = next(e for e in gate.matrix["entries"] if e["id"] == f"step_{sign}")
        if coarse["accepted"]:
            raise ValueError("Original pre-asymptotic rejection must remain explicit")
        runs = [load(Path(p).name) for p in row["paths"][:-1]]
        tight = load(Path(row["paths"][-1]).name)
        exact = [exact_over_time(run, velocity=velocity, shape="step") for run in runs]
        orders = [
            observed_order(a["temperature_l1_k"], b["temperature_l1_k"])
            for a, b in zip(exact, exact[1:])
        ]
        temporal = time_difference(runs[-1], tight)
        spatial = profile_difference(runs[-2], runs[-1])["temperature_l1_k"]
        fine = exact[-1]
        checks = dict(
            completed=all(
                r["report"]["accepted"] and r["report"]["restart_passed"] for r in runs + [tight]
            ),
            spatial=fine["temperature_l1_k"] <= budget["step_l1_k"],
            order=all(p is not None and p >= budget["step_min_order"] for p in orders),
            temporal=temporal <= budget["temperature_time_abs_k"]
            and temporal <= budget["time_spatial_ratio"] * spatial + budget["temperature_floor_k"],
            pressure=fine["pressure_error_pa"] <= budget["contact_pressure_abs_pa"],
            bounds=fine["min_temperature_k"] >= 499.99 - budget["overshoot_k"]
            and fine["max_temperature_k"] <= 500.01 + budget["overshoot_k"],
        )
        if checks != row["checks"] or all(checks.values()) != row["accepted"]:
            raise ValueError("Refinement acceptance disagrees with raw trajectories")
        if (
            exact != row["exact_errors"]
            or orders != row["orders"]
            or temporal != row["time_linf_k"]
        ):
            raise ValueError("Refinement metrics disagree with raw trajectories")
        # Both old rejected and arbitrary intermediate-grid requests must fail.
        for name, cells in ((f"step_{sign}", 64), (case, 64)):
            try:
                gate.approved_input(name, cells)
            except DomainError:
                pass
            else:
                raise ValueError("Gate admitted an unqualified front fixture")
        if row["accepted"]:
            cells = row["grids"][-1]
            gate.verify_request(case, cells, **gate.approved_input(case, cells))
        entries.append(
            dict(
                id=case,
                accepted=row["accepted"],
                checks=checks,
                orders=orders,
                time_linf_k=temporal,
                temperature_l1_k=fine["temperature_l1_k"],
            )
        )
    return dict(
        passed=True,
        all_refined_accepted=all(e["accepted"] for e in entries),
        all_asymptotic_accepted=bool([e for e in entries if e["id"].endswith("_asymptotic")])
        and all(e["accepted"] for e in entries if e["id"].endswith("_asymptotic")),
        all_time_resolved_accepted=bool([e for e in entries if e["id"].endswith("_time_resolved")])
        and all(e["accepted"] for e in entries if e["id"].endswith("_time_resolved")),
        entries=entries,
        source_sha256=gate.matrix["source_sha256"],
        validated=False,
        plant_referenced=False,
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
            dict(passed=result["passed"], all_refined_accepted=result["all_refined_accepted"])
        )
    )


if __name__ == "__main__":
    main()
