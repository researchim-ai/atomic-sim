# Seconds-scale moderator transients — v132

[Русский](MODERATOR_TRANSIENTS_RU.md) · [Underlying model](PWR_MODERATOR_EN.md)

This package checks whether local moderator feedback produces a numerically
resolved response in the coupled PWR after a public rod-drive command.
The earlier v131 short run reached a correction of only `1.30e-11`.
The new runs start from the hot stationary state, move the rod by one percent
of its stroke, and allow coolant density to change. Inventories are never
assigned manually. Coefficients, physical balance tolerances and applicability
limits are not fitted or relaxed.

## Matched scenarios

`insertion` increases inserted rod fraction by `0.01`, using the operating
speed of `0.01` strokes per second. Observations occur at `0.1`, `0.5`, `1`
and `2` seconds. `return` requests the initial position at one second and
adds an observation at three seconds. A JSON checkpoint is saved at `0.5 s`,
while the rod is moving.

Each scenario has four trajectories: `LocalModeratorPWRSession` and the
preceding bulk law in `StationaryChannelPWRSession`, each with
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`.
The core has two neutron nodes and the adapter's standard fluid mesh.
Initial physical states must agree within `1e-12` relative discrepancy;
commands and observation times must match. Power comparisons use fission
power, excluding decay heat.

This is a controlled **internal comparison within one code family**.
The bulk law is neither an independent code nor an experimental reference.
A larger difference does not establish greater physical accuracy.

## Acceptance criteria

The maximum difference between the two fine fission-power traces is divided
by initial fission power. It must exceed `1e-4` and be at least 20 times the
largest coarse/fine power discrepancy from either law. This distinguishes
the observed law effect from sensitivity to these numerical presets; it
does not estimate physical model uncertainty.

Additional gates require:

- full coupled-state coarse/fine discrepancy at most `1e-4`;
- a reactivity correction of at least `1e-6` and density change of at least
  `0.01 kg/m³`;
- existing mass, energy, boron, neutron and precursor balances;
- frozen reference fields and inventory mapping defects at most `1e-12`;
- exact fine continuation from the mid-motion JSON checkpoint;
- final target position and complete rod-arrival events at `1 s`, or `1 and 2 s`;
- unchanged state following an invalid command.

Balance checks are recomputed from retained snapshots rather than trusted
from their cached `checks` field. Full trajectories and unsuccessful
preparatory attempts remain in local artifacts.

Accepted delivery: **129 scoped source tests and 98 installed-wheel tests**, all eight phases passed, whole scientific JSON equal. Insertion/return power effects were **0.39276% / 0.58440%** of initial fission power. Maximum power preset discrepancy was `7.62e-11` of initial power; full-state discrepancy was `1.50e-7`. Maximum reactivity correction was `4.11e-5` and density change `0.301 kg/m³`.

## Rod-arrival event fix

These runs exposed a missing event: analytic arrival time could fall a few
ULPs beyond an interval endpoint even though rod position had already rounded
exactly to its target. No later interval would emit the event.

The shared `rod_arrival_event` accepts an endpoint arrival only when position
**equals the target exactly** and clock discrepancy lies within a ULP bound.
It does not alter position, speed or the physical RHS. Normal advancement,
hybrid feed-event continuation and the piecewise heat-history reader use the
same rule. The event timestamp cannot exceed the current endpoint; an early
arrival outside the roundoff bound is rejected. A separate retained-run
comparison checks that all observed insertion physical states are unchanged
by this metadata fix.

Changing shared source files changes station source-bound signatures.
Historical hour-long receipts retain their original version scope and are
not transferred to v132. Checkpoints with different source signatures are
not migrated. No new hour-long run is included here.

## Local reproduction

From the checkout root with dependencies installed:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component moderator-transients \
  --output artifacts/moderator-transients-local --phase-timeout-s 1200
```

Use a fresh output directory. The profile runs scoped tests and scientific
cases, builds an sdist, builds a wheel from that sdist, installs it into a
fresh environment, and repeats tests and calculations outside the checkout.
Entire scientific JSON documents, including events and checkpoints, must
match. Scientific dependencies are reused from the local environment;
no network or CI is required. This component profile does not replace the
full library suite. Results and delivery hashes are recorded in the
[v132 receipt](benchmarks/moderator_transients_v132.json).

## Remaining scope

The horizons are **2 and 3 seconds**, with eight trajectories and two local
fine restart suffixes. Boron transients, a chemical actuator, spatial
convergence, hour-long nominal operation, full SCRAM, coastdown and long
heat-removal transients are not qualified here. The local adapter is not
connected to StationEnv. Coefficients remain illustrative; evaluated local
cross sections and experimental validation are still absent. NPP-12.02/03
remain open; root completion remains **58/76**.
