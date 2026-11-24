# Joint PWR mesh refinement — v136

[Русский](COUPLED_MESH_RU.md) · [Hydraulic geometry](FLUID_GEOMETRY_EN.md)

`coupling.refinement.coupled_pwr_mesh` returns compatible configurations for
`LocalModeratorPWRSession`, refining neutron, liquid, fuel and cladding meshes
together:

- resolution 2: 2 neutron, 2 fluid and 4 thermal cells;
- resolution 4: 4 neutron, 4 fluid and 8 thermal cells;
- resolution 8: 8 neutron, 8 fluid and 16 thermal cells.

Defaults are a 4 m span and extra channel K = 1. Fluid uses complete dual
geometry; the thermal mesh includes source/fluid interfaces with two children
per interval. Existing 8/8/16 cell budgets remain. This recipe constructs
a new calculation, without remapping an active physical state.

```python
from coupling.refinement import coupled_pwr_mesh
from coupling.pwr_moderator import LocalModeratorPWRSession

session = LocalModeratorPWRSession(**coupled_pwr_mesh(4))
initial = session.snapshot()
rod = initial["actuator_status"]["rod_position"]
result = session.advance(0.1, {"rod_target": rod + 0.01})
```

Explicit `length_m` and `loss_coefficient` are supported, but changing them
changes the physical problem rather than just its resolution. This matrix
is qualified only at 4 m and K = 1.

## Matched physical problem

Each mesh has a separate hot stationary solve at the same requested power
and initial inventory constraints. Rod trims are retained, followed by the
same `+0.01` inserted-fraction increment from each trim. Absolute commands
and spatial initial states need not be identical.

Only cell counts may change between compared configurations. Neutron
coefficients, length, area, loss, mechanical closure, legacy external boundary
parameters and channel reference density must match. Both capacities and UA
keep the same per-length densities. Every hydraulic face coefficient is
checked individually, including both boundary halves; matching totals alone
is insufficient.

This is a controlled comparison of discretizations within one synthetic
model. The finest mesh is neither an independent code nor an experimental
reference.

## Twenty-two profiles

The scientific record retains separate fields for:

- fluid mass, internal energy, boron, temperature, pressure and density;
- fuel/cladding temperatures, three decay reservoirs, fission, nuclear heat
  and both intercomponent heat transfers;
- neutron fission power and all six individual precursor families.

Overlap norms distinguish cell-average fields from cell-integrated inventories
and powers; different-sized state vectors are not compared directly across
meshes. Differences are taken before averaging, so opposing local errors
cannot disappear behind equal total power.

Common scales use initial fission power per length, each initial reservoir
per length, and initial mean precursor fields on the finest mesh.
Temperature scales are 1000 K for fuel and 600 K for cladding/fluid;
pressure uses 15.5 MPa and density 1000 kg/m³. These are numerical
normalizations, not physical uncertainty estimates.

## Acceptance contract

Observations are at `0`, `0.1`, `0.5`, `1`, `2 s`. All resolutions use both
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`.
The finest fine run has an exact JSON restart at `0.5 s`, during rod motion:
six trajectories and one restart suffix.

For every profile, maximum normalized L2 across observations, including
initialization, must meet three requirements:

- 4/8 discrepancy at most 0.9 times the 2/4 discrepancy;
- spatial discrepancy at least twenty times the largest temporal preset
  discrepancy across all meshes;
- 4/8 discrepancy within its declared budget: 0.01 for temperatures, 0.001
  for pressure, 0.02 for other fluid fields and 0.05 for remaining thermal
  and neutron profiles.

Other gates check full-state temporal discrepancy ≤ `1e-4`, 4/8 total-power
discrepancy ≤ `1e-3` of initial power, coefficients within `1e-12`, mapped
mass/boron within `1e-12`, local heat/entropy identities, geometry/momentum/
kinetic work, previous global budgets and neutron/precursor balances.
Stationary rank/residuals, frozen references, rod arrivals and invalid-command
rollback are required too.

Aggregate maxima do not replace trajectories: every observation's profiles
remain in the scientific JSON. No convergence order or continuum-error
estimate is inferred from three meshes. This package does not accept a cheap
model replacement for arbitrary agent control.

Accepted delivery: **136 scoped source tests and 92 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Maximum neutron-power L2 for 2/4 → 4/8 was `0.01063574 → 0.00520071`. Full-state temporal discrepancy was `1.63116e-07`; local hydraulic coefficient partition defect was `2.79796e-16`. All 13 negative controls rejected corrupt budgets, individual families, coefficients, area, reference, inventory maps, restart and matrix completeness.

## Local reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component coupled-mesh \
  --output artifacts/coupled-mesh-local --phase-timeout-s 1200
```

Use a fresh output directory. Eight phases cover scoped tests, scientific
runs, sdist → wheel, fresh installation and replay outside checkout.
Whole scientific JSON documents must match. Local dependencies are reused;
no CI or network is used. Metrics and hashes are in the
[v136 receipt](benchmarks/coupled_mesh_v136.json).

## Scope

Joint refinement applies to this two-second single-phase case, with constant
UA, synthetic one-group data and fixed-reference-density kinetic storage.
It does not qualify other plant scenarios. Boron transients, applicable
physical closures, real nuclear data, StationEnv and long-duration regimes
remain open. No new hour-long run or full library suite is executed.
NPP-12.02/03/04 remain open; root completion is **58/76, with 18 tasks open**.
