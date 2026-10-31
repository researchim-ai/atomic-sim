"""Tighten time accuracy without discarding hashed, source-identical grid runs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from physics.hydraulics.euler_channel import EulerLiquidChannel
from scripts.qualify_hllc import (
    exact_over_time,
    observed_order,
    profile_difference,
    run_case,
    time_difference,
    write,
)
from sdk.euler_qualification import EulerQualification
from sdk.experiment import _source_fingerprint


def resolve(parent, output, budget_path):
    gate = EulerQualification(parent)
    source = _source_fingerprint()
    budget = json.loads(budget_path.read_text())
    original = json.loads((parent / "budget.json").read_text())
    for key, value in original.items():
        if key != "protocol_version" and budget[key] != value:
            raise ValueError("Parent numerical criteria may not be changed")
    output.mkdir()
    # Copies retain every rejected parent entry. The original evidence is immutable.
    for path in parent.iterdir():
        if path.name in ("manifest.json", "qualification.json", "budget.json"):
            continue
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)
    write(output / "budget.json", budget)
    matrix = json.loads((parent / "qualification.json").read_text())
    reuse = []

    def load(path):
        return dict(
            model=EulerLiquidChannel.from_snapshot(
                json.loads((path / "checkpoint.json").read_text())
            ),
            rows=json.loads((path / "trajectory.json").read_text()),
            report=json.loads((path / "report.json").read_text()),
            path=str(path),
        )

    for sign, velocity in (("negative", -1.0), ("positive", 1.0)):
        case = f"step_{sign}_time_resolved"
        previous = f"step_{sign}_asymptotic_512_tight"
        base_path = output / f"{case}_512"
        canonical = json.loads((parent / previous / "input.json").read_text())
        if canonical["solver"] != budget["front_time_base_solver"]:
            raise ValueError("Reused base does not match registered solver policy")
        shutil.copytree(parent / previous, base_path)
        base = load(base_path)
        tight = run_case(
            output / f"{case}_512_tight",
            base["model"].config,
            canonical["duration_s"],
            budget,
            solver_policy=budget["front_time_tight_solver"],
        )
        runs = [
            load(output / f"step_{sign}_refined_128"),
            load(output / f"step_{sign}_asymptotic_256"),
            base,
        ]
        exact = [exact_over_time(r, velocity=velocity, shape="step") for r in runs]
        orders = [
            observed_order(a["temperature_l1_k"], b["temperature_l1_k"])
            for a, b in zip(exact, exact[1:])
        ]
        td = time_difference(base, tight)
        spatial = profile_difference(runs[-2], base)["temperature_l1_k"]
        fine = exact[-1]
        checks = dict(
            completed=all(
                r["report"]["accepted"] and r["report"]["restart_passed"] for r in runs + [tight]
            ),
            spatial=fine["temperature_l1_k"] <= budget["step_l1_k"],
            order=all(p is not None and p >= budget["step_min_order"] for p in orders),
            temporal=td <= budget["temperature_time_abs_k"]
            and td <= budget["time_spatial_ratio"] * spatial + budget["temperature_floor_k"],
            pressure=fine["pressure_error_pa"] <= budget["contact_pressure_abs_pa"],
            bounds=fine["min_temperature_k"] >= 499.99 - budget["overshoot_k"]
            and fine["max_temperature_k"] <= 500.01 + budget["overshoot_k"],
        )
        matrix["entries"].append(
            dict(
                id=case,
                accepted=all(checks.values()),
                checks=checks,
                grids=[128, 256, 512],
                original_entry=f"step_{sign}_asymptotic",
                exact_errors=exact,
                orders=orders,
                time_linf_k=td,
                fine_spatial_difference_k=spatial,
                qualified_cells=[512] if all(checks.values()) else [],
                paths=[r["path"] for r in runs + [tight]],
                scope="Same spatial thresholds, stricter qualified base solver; former time failure retained",
            )
        )
        reuse.append(
            dict(
                case=case, copied_from=previous, input_sha256=gate.hashes[f"{previous}/input.json"]
            )
        )
        print(
            json.dumps(
                dict(case=case, accepted=all(checks.values()), checks=checks, time_linf_k=td)
            ),
            flush=True,
        )
    if source != _source_fingerprint():
        raise RuntimeError("Source changed during temporal qualification")
    matrix.update(
        all_cases_accepted=all(e["accepted"] for e in matrix["entries"]),
        source_sha256=source,
        budget_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_matrix_sha256=gate.hashes["qualification.json"],
        reused_base_runs=reuse,
        additional_integrations=2,
        unique_integrations=len(list(parent.glob("*/report.json"))) + 2,
        scope="Hash-verified parent campaign plus stricter time controls; rejected entries retained",
    )
    write(output / "qualification.json", matrix)
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    write(output / "manifest.json", dict(sha256=hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget", type=Path, default=Path("validation/ai04_qualification_budget_v4.json")
    )
    args = parser.parse_args()
    resolve(args.parent, args.output, args.budget)


if __name__ == "__main__":
    main()
