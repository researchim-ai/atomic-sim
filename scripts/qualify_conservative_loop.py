"""Reproducible numerical comparison; does not grant industrial acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.audit_conservative_loop import audit, provenance
from scripts.conservative_eos_loop import ConservativeEOSLoop, signature
from scripts.conservative_loop_references import IndependentLoop, cases


def qualify():
    budget = json.loads(Path("validation/ai05_conservative_loop_budget_v1.json").read_text())
    rows = []
    times = [0.1, 0.5, 0.55, 1.0, 1.5, 1.55, 2.0]
    fields = {
        "masses_kg": "reference_mass_atol_kg",
        "velocities_m_s": "reference_velocity_atol_m_s",
        "temperatures_k": "reference_temperature_atol_k",
        "pressures_pa": "reference_pressure_atol_pa",
    }
    for name, spec, schedule in cases():
        refs = {
            method: IndependentLoop(spec).integrate(2.0, schedule, method, times)
            for method in ["DOP853", "Radau"]
        }
        for cap in [0.002, 0.001, 0.0005]:
            engine = ConservativeEOSLoop(spec, max_step_s=cap)
            errors = {key: 0.0 for key in fields}
            residuals = {
                key: 0.0
                for key in [
                    "mass_residual_kg",
                    "momentum_residual_kg_m_s",
                    "energy_residual_j",
                    "entropy_residual_j_k",
                ]
            }
            ledger_errors = np.zeros(6)
            max_step = courant = 0.0
            previous = 0.0
            restored = None
            for index, end in enumerate(times):
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

                result = advance(engine)
                if restored is not None:
                    advance(restored)
                    if restored.checkpoint() != engine.checkpoint():
                        raise AssertionError("Restart trajectory differs")
                if end == 1.0:
                    restored = ConservativeEOSLoop(spec, max_step_s=cap)
                    restored.restore(json.loads(json.dumps(engine.checkpoint())))
                state = engine.snapshot()["state"]
                max_step = max(max_step, result["work"]["maximum_internal_step_s"])
                courant = max(courant, result["work"]["maximum_courant"])
                for key in residuals:
                    residuals[key] = max(residuals[key], abs(state[key]))
                for reference in refs.values():
                    row = reference[index]
                    for key, limit in fields.items():
                        error = float(np.max(abs(np.array(list(state[key].values())) - row[key])))
                        errors[key] = max(errors[key], error)
                        if error > budget[limit]:
                            raise AssertionError((name, cap, end, key, error, budget[limit]))
                    from scripts.conservative_eos_loop import LEDGERS

                    ledger_errors = np.maximum(
                        ledger_errors, abs(np.array([state[k] for k in LEDGERS]) - row["ledgers"])
                    )
                previous = end
            if max(ledger_errors[:2]) > budget["reference_energy_atol_j"]:
                raise AssertionError((name, "energy ledger", ledger_errors.tolist()))
            if max(ledger_errors[2:4]) > budget["reference_entropy_atol_j_k"]:
                raise AssertionError((name, "entropy ledger", ledger_errors.tolist()))
            rows.append(
                dict(
                    case=name,
                    max_step_s=cap,
                    observations=len(times),
                    max_errors=errors,
                    max_balance_residuals=residuals,
                    max_ledger_errors=ledger_errors.tolist(),
                    observed_internal_step_s=max_step,
                    maximum_courant=courant,
                    restart_exact=True,
                )
            )
    return dict(
        schema_version=1,
        scope="Synthetic periodic first-order Euler loop",
        signature=signature(),
        provenance=provenance(),
        case_count=9,
        trajectory_count=len(rows),
        independent_ivps=18,
        observations=7 * len(rows),
        horizon_s=2.0,
        validated=False,
        plant_referenced=False,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/ai05_conservative_loop_campaign_v1.json")
    args = parser.parse_args()
    result = qualify()
    audit(result)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["rows", "signature"]}))


if __name__ == "__main__":
    main()
