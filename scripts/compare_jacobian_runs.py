"""Strict same-physics, same-tolerances Jacobian-mode equivalence audit."""

import argparse
import json
from pathlib import Path

from scripts.compare_dynamic_runs import compare as compare_trajectory
from scripts.compare_isobaric_runs import load


def compare(analytic, central, budgets):
    horizon = json.loads(Path(budgets).read_text())["horizon_s"]
    a, b = (load(p, horizon) for p in (analytic, central))
    if a[4]["source_sha256"] != b[4]["source_sha256"]:
        raise ValueError("Jacobian audit requires identical runtime source hashes")
    excluded = {"jacobian_mode", "jacobian_relative_step"}
    settings = [{k: v for k, v in r[0]["solver"].items() if k not in excluded} for r in (a, b)]
    if settings[0] != settings[1] or a[0]["step_s"] != b[0]["step_s"]:
        raise ValueError("Jacobian audit must not change tolerances, time method or output grid")
    if (
        a[0]["solver"].get("jacobian_mode") != "analytic"
        or b[0]["solver"].get("jacobian_mode", "finite_difference") != "finite_difference"
        or "jacobian_relative_step" not in b[0]["solver"]
    ):
        raise ValueError("Expected analytic and explicit central-difference modes")
    if a[4]["versions"] != b[4]["versions"]:
        raise ValueError("Different NumPy/SciPy versions in Jacobian audit")
    result = compare_trajectory(analytic, central, budgets)
    da, db = (r[1]["solver_diagnostics"] for r in (a, b))
    result["checks"]["analytic_callback_used"] = da["analytic_jacobian_successes"] > 0
    size = 4 * a[0]["hydraulics"]["cells"] + (
        6 if a[0]["hydraulics"].get("energy_formulation") == "compatible_total" else 3
    )
    result["checks"]["sparse_callback_recorded"] = 0 < da["last_analytic_jacobian_nnz"] < size**2
    result["checks"]["central_callback_used"] = (
        db["accepted_jacobian_evaluations"] > 0 and db.get("analytic_jacobian_calls", 0) == 0
    )
    result["passed"] = all(result["checks"].values())
    result["kind"] = "jacobian_equivalence_verification"
    result["work"] = dict(
        analytic=da,
        central=db,
        central_to_analytic_rhs_callback_ratio=db["accepted_rhs_evaluations"]
        / max(da["accepted_rhs_evaluations"], 1),
        analytic_wall_time_s=a[1]["wall_time_s"],
        central_wall_time_s=b[1]["wall_time_s"],
    )
    result["scope"] = (
        "Same sampled trajectory and numerical settings; callback counts are not all EOS work; wall times are diagnostic, not a controlled speed benchmark"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analytic", type=Path)
    parser.add_argument("central", type=Path)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.analytic, args.central, args.budgets)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(dict(passed=result["passed"], measured=result["measured"], work=result["work"]))
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
