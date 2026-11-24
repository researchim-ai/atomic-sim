# Configuration-derived station holdouts — v116

`atomic_model_station_holdouts` audits train/dev/test factors directly from
`EnvConfig` and `Fault`. Renaming cases, faults or family labels cannot create
new physical conditions. Policies retain the existing observation/action-only
interface; configuration manifests remain evaluator material.

`audit_splits(cases, assignment, axes=...)` requires all three splits and
cross-split disjoint values for every requested factor. Supported factors are
`inventory_deadband_m` (automatic control required), `equilibrium_start`,
`film_model`, `fault_mechanism`, `fault_magnitude`, and `fault_onset_s`.
Every fault in a compound case participates in the audit. Onset uses seconds,
not tick labels. Seeds, solver settings, episode length and arbitrary labels
cannot establish physical parameter holdouts. Numeric representation and
signed zero cannot disguise common factors. Future faults, zero sensor
bias/drift and unit cooling-loss multipliers cannot create held-out conditions.
A factor with only two values cannot support three disjoint splits.

`evaluate_holdouts(...)` audits before invoking any policy, runs the native
station evaluator, and keeps all attempts and their costs. Its outer summary
separates full horizons, execution failures and early terminations. An early
SCRAM is retained without being counted as a completed full horizon. Completing
a horizon is distinct from achieving a control objective.

Run locally from the repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m pytest -q tests/test_station_holdouts.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.qualify_station_holdouts \
  --output artifacts/my-station-holdouts
```

After wheel installation, use `python -I -m scripts.qualify_station_holdouts
--output <fresh-directory>` outside the checkout. `episodes.json` retains costs;
`scientific.json` removes only measured wall-time fields for exact comparison.
Both are SHA-256 sealed for integrity, without implying independent authorship.

The contract matrix uses three five-tick, 0.01 s configurations: deadbands
0/0.0004/0.0008 m, pressure biases 150/300/450 kPa, and onsets 0.01/0.02/0.03 s.
Hold, invalid-command and SCRAM policies produce nine retained attempts.
Three deliberate leakage controls must be rejected.

Exact-value disjointness proves neither OOD difficulty nor statistical
independence. Jointly varying factors do not isolate causal effects. Fault
onset is a schedule within an episode, not a held-out operating-history window.
A pre-run audit proves configurations; recorded episodes establish execution.
This API currently supports one station geometry. Object holdout is refused;
`unseen_objects_validated` and `operating_history_holdout` remain false.
**NPP-14.04 remains open**, including independent objects and operating histories.

See the [sealed v116 receipt](benchmarks/station_physical_holdouts_v116.json)
for local execution and artifact hashes. The earlier 3367-test library receipt
belongs to its frozen source epoch and is not presented as rerun for v116.
