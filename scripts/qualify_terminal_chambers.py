"""Source-bound qualification of finite terminal-volume Euler assemblies."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.solve_hydraulic_graph import sha
from scripts.terminal_chamber_references import IndependentTerminal, cases
from scripts.terminal_eos_chambers import BUDGET, CHAMBER_LEDGERS, TerminalEOSChambers, signature

CAPS = (0.001, 0.0005, 0.00025)
TIMES = (0.02, 0.05, 0.1, 0.15, 0.2)
FIELDS = {
    "masses_kg": "reference_mass_atol_kg",
    "velocities_m_s": "reference_velocity_atol_m_s",
    "temperatures_k": "reference_temperature_atol_k",
    "pressures_pa": "reference_pressure_atol_pa",
}
CHAMBERS = {
    "m": "reference_mass_atol_kg",
    "u": "reference_velocity_atol_m_s",
    "T": "reference_temperature_atol_k",
    "p": "reference_pressure_atol_pa",
}


def capture():
    return dict(
        runtime=signature(),
        qualification={
            p: sha(p)
            for p in (
                "scripts/qualify_terminal_chambers.py",
                "scripts/terminal_chamber_references.py",
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
    for name, spec in cases():
        refs = [
            IndependentTerminal(spec).integrate(TIMES, method) for method in ["DOP853", "Radau"]
        ]
        for cap in CAPS:
            e = TerminalEOSChambers(spec, max_step_s=cap)
            errors = dict.fromkeys(FIELDS, 0.0)
            chamber_errors = dict.fromkeys(CHAMBERS, 0.0)
            balances = dict.fromkeys(
                [
                    "mass_residual_kg",
                    "momentum_residual_kg_m_s",
                    "energy_residual_j",
                    "entropy_residual_j_k",
                ],
                0.0,
            )
            ledger_errors = np.zeros(4)
            previous = 0.0
            restored = None
            maxstep = stability = 0.0
            for j, end in enumerate(TIMES):
                state = e.advance(end - previous)
                if restored is not None:
                    restored.advance(end - previous)
                    if restored.checkpoint() != e.checkpoint():
                        raise AssertionError("Terminal restart differs")
                if end == 0.1:
                    restored = TerminalEOSChambers(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(e.checkpoint())))
                for key in balances:
                    balances[key] = max(balances[key], abs(state[key]))
                maxstep = max(maxstep, state["work"]["maximum_internal_step_s"])
                stability = max(stability, state["work"]["maximum_stability_fraction"])
                for reference in refs:
                    r = reference[j]
                    for field in FIELDS:
                        errors[field] = max(
                            errors[field],
                            float(
                                np.max(
                                    abs(
                                        np.array(list(state["channel"][field].values()))
                                        - r["channel"][field]
                                    )
                                )
                            ),
                        )
                    for field in CHAMBERS:
                        chamber_errors[field] = max(
                            chamber_errors[field],
                            float(
                                np.max(
                                    abs(
                                        np.array(
                                            [
                                                state["chambers"][side][field]
                                                for side in ["left", "right"]
                                            ]
                                        )
                                        - r["chambers"][field]
                                    )
                                )
                            ),
                        )
                    ledger_errors = np.maximum(
                        ledger_errors,
                        abs(
                            np.array([state["chamber_ledgers"][key] for key in CHAMBER_LEDGERS])
                            - r["ledgers"]
                        ),
                    )
                previous = end
            for key, limit in FIELDS.items():
                if errors[key] > budget[limit]:
                    raise AssertionError((name, cap, "channel", key, errors[key], budget[limit]))
            for key, limit in CHAMBERS.items():
                if chamber_errors[key] > budget[limit]:
                    raise AssertionError(
                        (name, cap, "chambers", key, chamber_errors[key], budget[limit])
                    )
            limits = [
                budget["reference_energy_atol_j"],
                budget["reference_entropy_atol_j_k"],
                budget["reference_entropy_atol_j_k"],
                budget["reference_momentum_atol_kg_m_s"],
            ]
            if np.any(ledger_errors > limits):
                raise AssertionError((name, cap, "ledgers", ledger_errors.tolist(), limits))
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=5,
                    channel_errors=errors,
                    chamber_errors=chamber_errors,
                    max_ledger_errors=ledger_errors.tolist(),
                    max_balance_residuals=balances,
                    observed_step_s=maxstep,
                    stability_fraction=stability,
                    restart_exact=True,
                )
            )
    if identity != capture():
        raise AssertionError("Qualification sources changed")
    report = dict(
        schema_version=1,
        signature=identity,
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=80,
        observations=120,
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
            raise ValueError("Invalid terminal chamber evidence")

    def metric(value, limit=float("inf")):
        require(type(value) in (int, float) and np.isfinite(value) and 0 <= value <= limit)

    require(isinstance(report, dict) and report.get("signature") == capture())
    for key, value in dict(
        schema_version=1,
        cases=8,
        trajectories=24,
        independent_programs=16,
        independent_ivp_segments=80,
        observations=120,
        horizon_s=0.2,
    ).items():
        require(type(report.get(key)) is type(value) and report[key] == value)
    require(report.get("validated") is False and report.get("plant_referenced") is False)
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 24)
    budget = json.loads(BUDGET.read_text())
    expected = {(name, cap) for name, _ in cases() for cap in CAPS}
    seen = set()
    steps = {}
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
        metric(row.get("stability_fraction"), 1.0)
        for label, fields in [("channel_errors", FIELDS), ("chamber_errors", CHAMBERS)]:
            values = row.get(label)
            require(isinstance(values, dict) and set(values) == set(fields))
            for key, limit in fields.items():
                metric(values[key], budget[limit])
        values = row.get("max_ledger_errors")
        require(isinstance(values, list) and len(values) == 4)
        limits = [
            budget["reference_energy_atol_j"],
            budget["reference_entropy_atol_j_k"],
            budget["reference_entropy_atol_j_k"],
            budget["reference_momentum_atol_kg_m_s"],
        ]
        for value, limit in zip(values, limits):
            metric(value, limit)
        values = row.get("max_balance_residuals")
        require(
            isinstance(values, dict)
            and set(values)
            == {
                "mass_residual_kg",
                "momentum_residual_kg_m_s",
                "energy_residual_j",
                "entropy_residual_j_k",
            }
        )
        # Full-state relative balances are checked by the engine, not reconstructed from maxima.
        for value in values.values():
            metric(value)
    require(seen == expected)
    for name, _ in cases():
        values = [steps[(name, cap)] for cap in sorted(CAPS)]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])))
    return dict(
        passed=True,
        trajectories=24,
        observations=120,
        validated=False,
        plant_referenced=False,
        scope="Source identity, complete matrix and reference tolerances; no numerical rerun or authentication",
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
