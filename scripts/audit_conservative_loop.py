"""Strict numerical-evidence checks, not authentication or plant validation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.conservative_eos_loop import signature
from scripts.conservative_loop_references import cases

SOURCES = (
    "scripts/qualify_conservative_loop.py",
    "scripts/conservative_loop_references.py",
    "scripts/storage_graph_references.py",
    "scripts/audit_conservative_loop.py",
)
FIELDS = {
    "masses_kg": "reference_mass_atol_kg",
    "velocities_m_s": "reference_velocity_atol_m_s",
    "temperatures_k": "reference_temperature_atol_k",
    "pressures_pa": "reference_pressure_atol_pa",
}


def provenance():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES}


def audit(report):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def scalar(value, label):
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Invalid nonnegative finite metric: " + label,
        )
        return value

    require(isinstance(report, dict), "Report must be an object")
    for key, expected in dict(
        schema_version=1,
        case_count=9,
        trajectory_count=27,
        independent_ivps=18,
        observations=189,
        horizon_s=2.0,
    ).items():
        require(
            type(report.get(key)) is type(expected) and report[key] == expected,
            "Invalid coverage: " + key,
        )
    require(
        report.get("validated") is False and report.get("plant_referenced") is False,
        "Unsupported validation claim",
    )
    require(report.get("signature") == signature(), "Runtime/source/budget drift")
    require(report.get("provenance") == provenance(), "Reference/auditor provenance drift")
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 27, "Wrong trajectory count")
    budget = json.loads(Path("validation/ai05_conservative_loop_budget_v1.json").read_text())
    expected = {(name, cap) for name, _, _ in cases() for cap in [0.002, 0.001, 0.0005]}
    seen = set()
    steps = {}
    for row in rows:
        require(isinstance(row, dict), "Trajectory must be an object")
        name, cap = row.get("case"), row.get("max_step_s")
        require(isinstance(name, str), "Invalid case name")
        scalar(cap, "max_step_s")
        pair = (name, cap)
        require(pair in expected and pair not in seen, "Missing/duplicate/unexpected trajectory")
        seen.add(pair)
        require(
            type(row.get("observations")) is int and row["observations"] == 7,
            "Wrong observation count",
        )
        require(row.get("restart_exact") is True, "Restart not exact")
        step = scalar(row.get("observed_internal_step_s"), "observed step")
        require(0 < step <= cap * (1 + 1e-12), "Invalid measured step")
        steps[pair] = step
        require(
            scalar(row.get("maximum_courant"), "CFL") <= budget["maximum_courant"],
            "CFL exceeds budget",
        )
        errors = row.get("max_errors")
        require(isinstance(errors, dict) and set(errors) == set(FIELDS), "Wrong error fields")
        for field, limit in FIELDS.items():
            require(scalar(errors[field], field) <= budget[limit], "Reference tolerance: " + field)
        residuals = row.get("max_balance_residuals")
        names = {
            "mass_residual_kg",
            "momentum_residual_kg_m_s",
            "energy_residual_j",
            "entropy_residual_j_k",
        }
        require(isinstance(residuals, dict) and set(residuals) == names, "Wrong balance fields")
        # Actual per-observation balance bounds are checked by the engine, with state scaling.
        # This summary cannot independently reconstruct those state-dependent bounds.
        for field in names:
            scalar(residuals[field], field)
        ledgers = row.get("max_ledger_errors")
        require(isinstance(ledgers, list) and len(ledgers) == 6, "Wrong ledger layout")
        for i, value in enumerate(ledgers):
            scalar(value, "ledger " + str(i))
        require(max(ledgers[:2]) <= budget["reference_energy_atol_j"], "Energy ledger tolerance")
        require(
            max(ledgers[2:4]) <= budget["reference_entropy_atol_j_k"], "Entropy ledger tolerance"
        )
    require(seen == expected, "Incomplete trajectory matrix")
    for name, _, _ in cases():
        values = [steps[(name, cap)] for cap in [0.0005, 0.001, 0.002]]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])), "Time grids not distinct")
    return dict(
        status="passed",
        trajectories=27,
        observations=189,
        validated=False,
        plant_referenced=False,
        scope="Report coverage, finite metrics, reference tolerances and source identity; not a rerun or signature",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(Path(args.report).read_text())), sort_keys=True))


if __name__ == "__main__":
    main()
