"""Source-bound qualification of joint controlled finite-inventory assemblies."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.controlled_assembly_references import IndependentAssembly, cases, initial_controls
from scripts.controlled_eos_assembly import (
    BUDGET,
    CIRCUITS,
    WALL_LEDGERS,
    ControlledEOSAssembly,
    signature,
)
from scripts.inertial_graph import digest
from scripts.solve_hydraulic_graph import sha

CAPS = (0.001, 0.0005, 0.00025)
TIMES = (0.01, 0.025, 0.05, 0.075, 0.1)
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
                "scripts/qualify_controlled_assembly.py",
                "scripts/controlled_assembly_references.py",
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
    for name, spec, schedule in cases():
        refs = [
            IndependentAssembly(spec).integrate(TIMES, schedule, method)
            for method in ["DOP853", "Radau"]
        ]
        for cap in CAPS:
            e = ControlledEOSAssembly(spec, max_step_s=cap)
            errors = dict.fromkeys(FIELDS, 0.0)
            chambers = dict.fromkeys(CHAMBERS, 0.0)
            wall = energy = entropy = maxstep = stability = 0.0
            ledger_errors = np.zeros(5)
            previous = 0.0
            restored = None
            for j, end in enumerate(TIMES):
                command = next((p for when, p in schedule if when == previous), None)

                def advance(target):
                    if command is None:
                        return target.advance(end - previous)
                    return target.step(
                        command,
                        end - previous,
                        expected_revision=target.revision,
                        command_id="online.heat",
                    )

                s = advance(e)
                if restored is not None:
                    advance(restored)
                    if restored.checkpoint() != e.checkpoint():
                        raise AssertionError("Mixed restart differs")
                if end == 0.05:
                    restored = ControlledEOSAssembly(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(e.checkpoint())))
                for reference in refs:
                    r = reference[j]
                    wall = max(
                        wall,
                        float(
                            np.max(
                                abs(
                                    np.array(list(s["wall_temperatures_k"].values()))
                                    - r["wall_temperatures_k"]
                                )
                            )
                        ),
                    )
                    ledger_errors = np.maximum(
                        ledger_errors,
                        abs(
                            np.array([s["wall_ledgers"][k] for k in WALL_LEDGERS])
                            - r["wall_ledgers"]
                        ),
                    )
                    for i, circuit in enumerate(CIRCUITS):
                        actual = s["circuits"][circuit]
                        target = r["circuits"][i]
                        for key in FIELDS:
                            errors[key] = max(
                                errors[key],
                                float(
                                    np.max(
                                        abs(
                                            np.array(list(actual["channel"][key].values()))
                                            - target["channel"][key]
                                        )
                                    )
                                ),
                            )
                        for key in CHAMBERS:
                            chambers[key] = max(
                                chambers[key],
                                float(
                                    np.max(
                                        abs(
                                            np.array(
                                                [
                                                    actual["chambers"][side][key]
                                                    for side in ["left", "right"]
                                                ]
                                            )
                                            - target["chambers"][key]
                                        )
                                    )
                                ),
                            )
                energy = max(energy, abs(s["energy_residual_j"]))
                entropy = max(entropy, abs(s["entropy_residual_j_k"]))
                maxstep = max(maxstep, s["work"]["maximum_internal_step_s"])
                stability = max(stability, s["work"]["maximum_stability_fraction"])
                previous = end
            for mapping, limits in [(errors, FIELDS), (chambers, CHAMBERS)]:
                for key, limit in limits.items():
                    if mapping[key] > budget[limit]:
                        raise AssertionError((name, cap, key, mapping[key], budget[limit]))
            if wall > budget["reference_temperature_atol_k"]:
                raise AssertionError((name, cap, "wall", wall))
            limits = [
                budget[k]
                for k in [
                    "reference_energy_atol_j",
                    "reference_entropy_atol_j_k",
                    "reference_energy_atol_j",
                    "reference_entropy_atol_j_k",
                    "reference_entropy_atol_j_k",
                ]
            ]
            if np.any(ledger_errors > limits):
                raise AssertionError((name, cap, "ledgers", ledger_errors.tolist(), limits))
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=5,
                    channel_errors=errors,
                    chamber_errors=chambers,
                    wall_error_k=wall,
                    max_wall_ledger_errors=ledger_errors.tolist(),
                    energy_residual_j=energy,
                    entropy_residual_j_k=entropy,
                    observed_step_s=maxstep,
                    stability_fraction=stability,
                    restart_exact=True,
                    commands_committed=len(s["history"]),
                    end_control_sha256=digest(s["active_heat_power_w"]),
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
        horizon_s=0.1,
        rows=rows,
        validated=False,
        plant_referenced=False,
    )
    audit(report)
    return report


def audit(report):
    def require(condition):
        if not condition:
            raise ValueError("Invalid controlled assembly evidence")

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
        horizon_s=0.1,
    ).items():
        require(type(report.get(key)) is type(value) and report[key] == value)
    require(report.get("validated") is False and report.get("plant_referenced") is False)
    rows = report.get("rows")
    require(isinstance(rows, list) and len(rows) == 24)
    budget = json.loads(BUDGET.read_text())
    expected = {(name, cap) for name, _, _ in cases() for cap in CAPS}
    seen = set()
    steps = {}
    expected_controls = {
        name: (len(schedule), digest(schedule[-1][1] if schedule else initial_controls(spec)))
        for name, spec, schedule in cases()
    }
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("case"), str))
        cap = row.get("max_step_s")
        metric(cap, max(CAPS))
        pair = (row["case"], cap)
        require(pair in expected and pair not in seen)
        seen.add(pair)
        require(
            type(row.get("observations")) is int
            and row["observations"] == 5
            and row.get("restart_exact") is True
        )
        require(
            type(row.get("commands_committed")) is int
            and (row["commands_committed"], row.get("end_control_sha256"))
            == expected_controls[row["case"]]
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
        metric(row.get("wall_error_k"), budget["reference_temperature_atol_k"])
        values = row.get("max_wall_ledger_errors")
        require(isinstance(values, list) and len(values) == 5)
        limits = [
            budget[k]
            for k in [
                "reference_energy_atol_j",
                "reference_entropy_atol_j_k",
                "reference_energy_atol_j",
                "reference_entropy_atol_j_k",
                "reference_entropy_atol_j_k",
            ]
        ]
        for value, limit in zip(values, limits):
            metric(value, limit)
        # Scaled balances are checked on complete states by the engine.
        metric(row.get("energy_residual_j"))
        metric(row.get("entropy_residual_j_k"))
    require(seen == expected)
    for name, _, _ in cases():
        values = [steps[(name, cap)] for cap in sorted(CAPS)]
        require(all(b / a >= 1.2 for a, b in zip(values, values[1:])))
    return dict(
        passed=True,
        trajectories=24,
        observations=120,
        validated=False,
        plant_referenced=False,
        scope="Source identity, complete matrix, reference tolerances and commanded final mask; no numerical rerun or authentication",
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
