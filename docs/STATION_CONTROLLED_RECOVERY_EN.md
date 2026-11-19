# Evaluating controlled recovery — v108

[Русский](STATION_CONTROLLED_RECOVERY_RU.md)

`atomic_model_station_tasks.compare_recovery` compares two branches started from the same station checkpoint. The candidate acts in one branch; the default baseline holds the commands present at the fork. The caller's environment is not advanced.

The existing `Task(kind="recovery")` continues to score an observed return to a band. The new API asks whether candidate actions achieved recovery within a declared window when the baseline did not. Historical task metrics retain their original meaning.

## Success contract

The initial valid flow measurement must be outside the band. The candidate must complete every declared step, return the measured flow to the band and sustain a final dwell. Missing or invalid measurements interrupt the dwell and prevent a valid successful comparison. A protection trip does not count as recovery.

Both branches must complete without execution errors and pass the energy-accounting gates. A failed baseline policy or solver cannot make the candidate look successful. Attempted commands, failures and their phases remain in the report.

For invalid comparisons, `relative_iae_gain` is `null`. A failed or incomplete branch cannot publish a spurious positive reward from its empty or partial trace.

Success also requires different commands, an unrecovered baseline and a declared minimum relative reduction in sampled normalized absolute error:

`IAE = Σ abs(measured_flow − target) / target × Δt`.

Gain is `(IAE_baseline − IAE_candidate) / max(IAE_baseline, 1e-12 s)`. Samples use the ordinary tick's right endpoint. This does not prove continuous containment of the true physical flow. The contract deliberately rejects cases where both branches recover, even when the candidate is faster; that requires a different objective.

## Branches and policy inputs

Both branches copy physical state, controller memory, measurement filters, queues and model RNG state. Each policy is instantiated by a fresh factory and receives only measured observations, the public action schema and the objective. No checkpoint, latent physical state or fault schedule is supplied to it.

Comparison reports are privileged evaluator data and contain final checkpoints. They are not agent observations. This interface restriction does not sandbox arbitrary Python code. External policies must version their own code, weights and randomness; `policy_id` is an identifier, not a serialized policy.

## Example

```python
from atomic_model_station_env import StationEnv, EnvConfig
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_tasks import (
    RecoveryContract, compare_recovery, restart_flow_policy,
)

env = StationEnv(EnvConfig(max_steps=25, tick_s=.05,
                          automatic=False, rtol=1e-10))
observation, _ = env.reset()
target = valid_reading(observation, "feed_flow", "feed_flow_kg_s")
for _ in range(4):
    env.step({"operator": {"feed_pump_on": False}})

contract = RecoveryContract("feed_flow", target, steps=20, dwell_s=.2)
result = compare_recovery(env, contract, lambda: restart_flow_policy,
                          policy_id="restart-flow/v1")
```

All preparation uses native commands and integration. No flow or inventory state is manually projected. The baseline restart policy requests pump activation when the measured flow is below the target band. Supported channels are `feed_flow` and `condensate_flow`. The contract covers its declared steps from the fork; it need not cover the entire environment episode.

## Reproduction and limits

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. .venv-layout/bin/python -m scripts.qualify_station_controlled_recovery --output artifacts/my-controlled-recovery
```

The protocol retains five comparisons: controlled restart, no action, natural recovery, an invalid candidate command and an invalid baseline command. It also restores the original checkpoint and requires an exact repeat of the complete positive comparison.

The example covers one second after the fork in one research model. Deliberate negative controls are not a population estimate of agent performance. Longer tasks, equipment diversity, independent object holdouts and industrial utility need separate qualification. NPP-14.03/04 remain open; the plan remains at 50/76 completed items.

## Delivery results

The controlled case passed every gate. Sampled normalized IAE fell by 69.70% relative to holding the disabled pump. Recovery including the required dwell was established by 0.90 s after the fork, with 0.30 s final dwell. The baseline did not recover. These figures describe the declared sampled research case only.

All four negative controls were rejected as intended. Both naturally recovering branches met their individual goals without receiving controlled-recovery credit. Invalid comparisons returned null gain. A restored fork repeated the complete positive comparison exactly.

One hundred and three targeted tests passed, including twenty-nine new tests. An installed wheel outside the checkout repeated the twenty-nine new tests and all eight JSON documents exactly, without excluding fields. The full library suite and long R1 matrix were not run; the scientific station backend is unchanged.

[Machine-readable receipt](benchmarks/station_controlled_recovery_v108.json).
