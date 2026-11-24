# Spatial qualification of boron exchange — v138

[Русский](BORON_MESH_RU.md) · [Finite exchange component](PWR_BORON_EN.md)

Both 0 and 2000 ppm supply cases undergo joint mesh refinement with one
physical geometry and coefficient set:

- 2 neutron / 2 fluid / 4 thermal cells;
- 4 neutron / 4 fluid / 8 thermal cells;
- 8 neutron / 8 fluid / 16 thermal cells.

Span is 4 m and extra channel K is 1. Thermal faces align with source/fluid
interfaces and momentum includes both boundary halves. Tank, pump, composition
and flow-actuator parameters are not fitted to the finest mesh.

## Concentration increments

The 22 coupled-PWR profiles are supplemented by concentration changes in
fluid cells and on the neutron mesh, each relative to its own initialization:

```text
delta_c(x,t) = c(x,t) - c(x,0)
```

Initial density and mass distribution can differ across discretizations.
Absolute boron stocks alone can be insensitive to a small concentration
change against a large initial inventory. Increment fields isolate composition
changes. Neutron ppm still comes from mass-weighted mapping of actual M/B;
it is never directly assigned. Initial references must belong to the same
mesh, with no broadcasting of missing cells.

Exact overlap norms use a **1 ppm** scale: the `0.25` L2 budget means
0.25 ppm RMS, not a quarter of initial concentration. Opposing local errors
cannot disappear behind equal integrals.

## Scenarios and contract

Each mesh has its own hot stationary PWR. Initial primary states must match
between chemical cases; equipment differs only in initial supply boron.
Stationary initialization is not presented as a physical transient.

Supply opens to a 50 kg/s target at zero and closes by command at 2 s.
Observations are `0`, `0.1`, `0.5`, `1`, `2`, `2.5 s`, including the exact
closing-flow tail. Every case/mesh uses both `max_step_s=0.1, rtol=2e-8`
and `max_step_s=0.05, rtol=2e-9`: **12 native trajectories**.
Both finest fine trajectories have exact JSON restarts at 0.5 s; additional
restart suffixes on smaller meshes are not claimed.

For all 24 fields, maximum L2 across observations must contract:
4/8 ≤ 0.9 × 2/4. Spatial differences must exceed temporal preset sensitivity
by at least 20 times. Existing individual profile budgets and the new
0.25 ppm increment budget apply. This active chemical case uses a 4/8
total-power budget of `0.005` of initial power, and full-state temporal
agreement within `1e-4`. These are this package's numerical budgets, not
physical uncertainty or acceptance under a stricter budget from another case.

Additional gates require global mass/energy/boron and neutron/precursor
balances, exact actuator-integral tank depletion, correct primary-boron
direction, local heat/entropy identities, unchanged per-length capacities/UA,
complete inertance/reference kinetic storage, individual hydraulic face
coefficients, M/B mapping, frozen references, base stationary rank/residuals,
invalid-command rollback and a resolved chemical power effect on every mesh.

Every observation, field and family is retained. Neither continuum error
nor independent experimental validation is estimated.

## Restore reuses its verified base

`BoronExchangePWRSession.restore` still restores the base PWR through source
checks, a recomputed equilibrium and exact checkpoint comparison. It copies
that verified object, initializes fresh tanks from configuration, then replays
commands. The duplicate base solve has been removed. A test requires one
base constructor call and exact continuation. Saved tank states are not
trusted as initialization.

The first delivery attempt exposed a diagnostic bug: the inherited face-flow
list included appended tank states. The slice now ends at the channel-owned
internal faces. A regression test checks its length, values and geometry audit
with nonzero exchange flow; the failed attempt is retained.

Channel and adapter source changes update source-bound signatures throughout
the channel PWR family. Earlier checkpoints are not automatically rebound. A separate physical-trajectory comparison
against v137 is a numerical regression check only.

Accepted delivery: **88 scoped source tests, 78 additional channel regressions and 47 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Neutron concentration-increment L2 for 4/8 was `0.13190705 ppm` and `0.14289137 ppm`. Maximum full-state temporal discrepancy was `1.41225e-05`. All 13 negative controls passed, including wrong fluid/neutron fronts and initial reference, coefficient distribution, energy, transfer and restart.

## Local reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-mesh \
  --output artifacts/boron-mesh-local --phase-timeout-s 1200
```

Use a fresh output directory. Scoped tests and the matrix are followed by
sdist → wheel, a fresh installation outside checkout and full scientific
replay. Entire scientific JSON must match; its reader has an explicit
16 MiB bound. No CI or network is used. Metrics and hashes are in the
[v138 receipt](benchmarks/boron_mesh_v138.json).

Scope is these 2.5 seconds, single-phase water, illustrative boron worth,
linearized tanks and an externally powered pump. Long-duration regimes,
plant CVCS heat regeneration, chemistry/solubility, physical depletion and
power-loss events, StationEnv and experimental validation remain open.
No new hour or full library suite is executed. Root completion remains
**58/76, with 18 open tasks**.
