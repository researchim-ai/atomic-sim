"""Hash-checked full-horizon comparison for pressure-driven channels."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.compare_isobaric_runs import load


def compare(run, reference, budgets):
    p = Path(budgets)
    budget = json.loads(p.read_text())
    a, b = (load(folder, budget["horizon_s"]) for folder in (run, reference))
    if (
        a[0]["engine"] != "pressure-driven-wall-channel"
        or a[0]["engine"] != b[0]["engine"]
        or a[0]["hydraulics"] != b[0]["hydraulics"]
    ):
        raise ValueError("Comparison requires identical pressure-driven physics")
    if set(a[2]) - set(b[2]):
        raise ValueError("Reference is missing comparison times")
    fields = {
        "flow_absolute_kg_s": ("inlet_mass_flow_kg_s", "outlet_mass_flow_kg_s"),
        "pressure_absolute_pa": ("min_pressure_pa", "max_pressure_pa"),
        "fluid_temperature_absolute_k": ("min_temperature_k", "max_temperature_k"),
        "wall_temperature_absolute_k": ("min_wall_temperature_k", "max_wall_temperature_k"),
    }
    measured = {
        key: max(abs(row[field] - b[2][t][field]) for t, row in a[2].items() for field in fs)
        for key, fs in fields.items()
    }
    checks = {key: value <= budget[key] for key, value in measured.items()}
    checks.update(
        {
            key: a[3][key] <= budget[key] and b[3][key] <= budget[key]
            for key in ("mass_relative", "energy_relative")
        }
    )
    return dict(
        schema_version=1,
        kind="numerical_verification",
        validated=False,
        plant_referenced=False,
        horizon_s=budget["horizon_s"],
        comparison_times=len(a[2]),
        measured=measured,
        checks=checks,
        passed=all(checks.values()),
        budget_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        source_hashes=[a[4]["source_sha256"], b[4]["source_sha256"]],
        runs={str(run): a[1], str(reference): b[1]},
        scope="Single-phase low-Mach internal-energy channel; not total mechanical-energy qualification",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = compare(args.run, args.reference, args.budgets)
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(passed=r["passed"], measured=r["measured"])))
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
