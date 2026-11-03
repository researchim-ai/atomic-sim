"""Open-channel source-bound evidence and independent full-energy comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.open_channel_references import IndependentOpen, cases
from scripts.open_eos_channel import BOUNDARY_LEDGERS, BUDGET, OpenEOSChannel, signature
from scripts.solve_hydraulic_graph import sha

CAPS = (0.001, 0.0005, 0.00025)
TIMES = (0.02, 0.05, 0.1, 0.15, 0.2)
FIELDS = {
    "masses_kg": "reference_mass_atol_kg",
    "velocities_m_s": "reference_velocity_atol_m_s",
    "temperatures_k": "reference_temperature_atol_k",
    "pressures_pa": "reference_pressure_atol_pa",
}


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/qualify_open_channel.py",
                "scripts/open_channel_references.py",
                "scripts/conservative_loop_references.py",
                "scripts/storage_graph_references.py",
            )
        },
    )


def qualify():
    identity = capture()
    budget = json.loads(BUDGET.read_text())
    rows = []
    for name, spec, schedule in cases():
        refs = [
            IndependentOpen(spec).integrate(TIMES, schedule, method)
            for method in ["DOP853", "Radau"]
        ]
        for cap in CAPS:
            e = OpenEOSChannel(spec, max_step_s=cap)
            errors = dict.fromkeys(FIELDS, 0.0)
            balance = dict.fromkeys(
                [
                    "mass_residual_kg",
                    "momentum_residual_kg_m_s",
                    "energy_residual_j",
                    "entropy_residual_j_k",
                ],
                0.0,
            )
            ledger_errors = np.zeros(14)
            previous = 0.0
            restored = None
            maxstep = courant = 0.0
            for j, end in enumerate(TIMES):
                command = next((powers for when, powers in schedule if when == previous), None)

                def advance(target):
                    if command is None:
                        return target.advance(end - previous)
                    return target.step(
                        command,
                        end - previous,
                        expected_revision=target.revision,
                        command_id="heater.move",
                    )

                result = advance(e)
                state = result["state"]
                if restored is not None:
                    advance(restored)
                    if restored.checkpoint() != e.checkpoint():
                        raise AssertionError("Open restart differs")
                if end == 0.1:
                    restored = OpenEOSChannel(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(e.checkpoint())))
                for key in balance:
                    balance[key] = max(balance[key], abs(state[key]))
                maxstep = max(maxstep, result["work"]["maximum_internal_step_s"])
                courant = max(courant, result["work"]["maximum_courant"])
                local = [
                    state[k]
                    for k in [
                        "external_heat_j",
                        "exported_drag_heat_j",
                        "external_entropy_j_k",
                        "generated_entropy_j_k",
                        "gravity_impulse_kg_m_s",
                        "drag_impulse_kg_m_s",
                    ]
                ]
                actual = np.array(
                    local
                    + [
                        state["boundary_transport"][side][key]
                        for side in ["left", "right"]
                        for key in BOUNDARY_LEDGERS
                    ]
                )
                for reference in refs:
                    r = reference[j]
                    for field in FIELDS:
                        errors[field] = max(
                            errors[field],
                            float(np.max(abs(np.array(list(state[field].values())) - r[field]))),
                        )
                    ledger_errors = np.maximum(ledger_errors, abs(actual - r["ledgers"]))
                previous = end
            for field, limit in FIELDS.items():
                if errors[field] > budget[limit]:
                    raise AssertionError((name, cap, field, errors[field], budget[limit]))
            limits = (
                [budget["reference_energy_atol_j"]] * 2
                + [budget["reference_entropy_atol_j_k"]] * 2
                + [budget["reference_momentum_atol_kg_m_s"]] * 2
            )
            limits += [
                budget[k]
                for k in [
                    "reference_mass_atol_kg",
                    "reference_momentum_atol_kg_m_s",
                    "reference_energy_atol_j",
                    "reference_entropy_atol_j_k",
                ]
            ] * 2
            if np.any(ledger_errors > limits):
                raise AssertionError((name, cap, "ledgers", ledger_errors.tolist(), limits))
            if name == "opposed_reversal" and state["mean_velocity_m_s"] >= 0:
                raise AssertionError("Expected signed reversal missing")
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=5,
                    errors=errors,
                    max_ledger_errors=ledger_errors.tolist(),
                    max_balance_residuals=balance,
                    observed_step_s=maxstep,
                    maximum_courant=courant,
                    restart_exact=True,
                    final_mean_velocity_m_s=state["mean_velocity_m_s"],
                )
            )
    if identity != capture():
        raise AssertionError("Qualification source drift")
    report = dict(
        schema_version=1,
        signature=identity,
        cases=12,
        trajectories=36,
        independent_programs=24,
        independent_ivp_segments=120,
        observations=180,
        horizon_s=0.2,
        rows=rows,
        validated=False,
        plant_referenced=False,
    )
    audit(report)
    return report


def audit(report):
    def require(condition):
        if not condition:
            raise ValueError("Invalid open-channel evidence")

    def metric(value, limit=float("inf"), nonnegative=True):
        require(
            type(value) in (int, float)
            and np.isfinite(value)
            and (value >= 0 or not nonnegative)
            and abs(value) <= limit
        )

    require(isinstance(report, dict) and report.get("signature") == capture())
    for key, value in dict(
        schema_version=1,
        cases=12,
        trajectories=36,
        independent_programs=24,
        independent_ivp_segments=120,
        observations=180,
        horizon_s=0.2,
    ).items():
        require(type(report.get(key)) is type(value) and report[key] == value)
    require(report.get("validated") is False and report.get("plant_referenced") is False)
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 36)
    budget = json.loads(BUDGET.read_text())
    seen = set()
    steps = {}
    expected = {(name, cap) for name, _, _ in cases() for cap in CAPS}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("case"), str))
        cap = row.get("max_step_s")
        metric(cap, max(CAPS))
        pair = (row["case"], cap)
        require(pair in expected and pair not in seen)
        seen.add(pair)
        require(
            row.get("restart_exact") is True
            and type(row.get("observations")) is int
            and row["observations"] == 5
        )
        metric(row.get("observed_step_s"), cap * (1 + 1e-12))
        require(row["observed_step_s"] > 0)
        steps[pair] = row["observed_step_s"]
        metric(row.get("maximum_courant"), budget["maximum_courant"])
        errors = row.get("errors")
        require(isinstance(errors, dict) and set(errors) == set(FIELDS))
        for field, limit in FIELDS.items():
            metric(errors[field], budget[limit])
        values = row.get("max_ledger_errors")
        require(isinstance(values, list) and len(values) == 14)
        limits = (
            [budget["reference_energy_atol_j"]] * 2
            + [budget["reference_entropy_atol_j_k"]] * 2
            + [budget["reference_momentum_atol_kg_m_s"]] * 2
        )
        limits += [
            budget[k]
            for k in [
                "reference_mass_atol_kg",
                "reference_momentum_atol_kg_m_s",
                "reference_energy_atol_j",
                "reference_entropy_atol_j_k",
            ]
        ] * 2
        for v, limit in zip(values, limits):
            metric(v, limit)
        balances = row.get("max_balance_residuals")
        require(
            isinstance(balances, dict)
            and set(balances)
            == {
                "mass_residual_kg",
                "momentum_residual_kg_m_s",
                "energy_residual_j",
                "entropy_residual_j_k",
            }
        )
        # Engine enforces full-state scaled balances. Summary maxima cannot reconstruct scales.
        for value in balances.values():
            metric(value)
        metric(row.get("final_mean_velocity_m_s"), nonnegative=False)
        if row["case"] == "opposed_reversal":
            require(row["final_mean_velocity_m_s"] < 0)
    require(seen == expected)
    for name, _, _ in cases():
        values = [steps[(name, cap)] for cap in sorted(CAPS)]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])))
    return dict(
        passed=True,
        trajectories=36,
        observations=180,
        validated=False,
        plant_referenced=False,
        scope="Source identity, complete trajectory matrix, numerical reference tolerances and signed reversal; no numerical rerun or authentication",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "audit"])
    parser.add_argument("path")
    args = parser.parse_args()
    target = Path(args.path)
    if args.mode == "audit":
        print(json.dumps(audit(json.loads(target.read_text())), sort_keys=True))
    else:
        report = qualify()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit(report), sort_keys=True))


if __name__ == "__main__":
    main()
