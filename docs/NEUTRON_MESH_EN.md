# Neutron-mesh refinement in coupled PWR — v133

[Русский](NEUTRON_MESH_RU.md) · [Previous transients](MODERATOR_TRANSIENTS_EN.md)

This package measures sensitivity of spatial fission power and all six
precursor families to neutron cell count in `LocalModeratorPWRSession`.
Local fuel temperatures and coolant density/boron enter the neutron RHS;
cell-integrated fission returns conservatively to the thermal model.

## Conditional refinement

Uniform neutron meshes have 2, 4 and 8 cells over 4 m. Fuel/cladding faces
`(0, 0.7, 2.2, 4)` and fluid faces `(0, 1, 2, 4)` stay fixed. Diffusion,
fission, feedback and precursor coefficients, boundaries and equipment do
not change. This is **conditional neutron-mesh refinement with fixed thermal
and fluid discretization**, not full multiphysics spatial convergence.

Each mesh gets its own stationary solve at the same requested power and
initial inventory constraints. Spatial states and stationary rod trims may
differ; trims are retained. No constitutive parameters are fitted. Each
trajectory requests the same `+0.01` inserted-fraction increment from its
own trim. Absolute command values are therefore not claimed identical.

Observation times are `0`, `0.1`, `0.5`, `1`, `2 s`. Every mesh uses both
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`.
Eight nodes is the largest allowed neutron mesh in this adapter, serving
as a numerical reference rather than a proven continuum solution.

## Cross-mesh norms

`coupling.mesh_comparison.profile_error` integrates the difference between
piecewise constant reconstructions directly on exact cell overlaps:

```text
L1 = sum(overlap[j,i] * abs(a[i]-b[j])) / length
L2 = sqrt(sum(overlap[j,i] * (a[i]-b[j])**2) / length)
Linf = max(abs(a[i]-b[j])) over nonempty overlaps
```

Cell-integrated powers are first divided by their own cell widths. Power
profiles use the common line-power normalization `P_initial / length`.
Each precursor family is compared separately as a cell-average field,
normalized by that family's initial mean on the eight-node fine mesh.

Fine values are not averaged onto the coarse mesh before taking absolute
differences. Opposing subcell errors such as `[+1, -1]` against zero have
zero signed integral discrepancy but nonzero spatial norms. Cancellation
must not hide these errors. Signed integral discrepancy is retained too.
Invalid endpoints, nonnumeric inputs, missing semantics, nonpositive scales
and unrepresentable outputs are rejected. Scaled computation avoids
intermediate squaring overflow for large representable norms. Tests cover
normalization independence as well: an absolute error of `1e-100` remains
representable with a normalization of `1e300`. The computation amplitude
comes from field differences rather than the requested normalization.
Subtraction precedes scaling so a large equal background cannot erase a small
local error. Tests also cover
hand integrals on nonmatching meshes, equal profiles, integral semantics,
opposing errors and all precursor families.

## Acceptance criteria

Full snapshots, checkpoints and every observation's errors are retained.
Aggregate gates apply to maximum discrepancies across all observation times,
including initialization:

- 4/8 normalized L2 at most `0.05` for power and each precursor family;
- 4/8 maximum L2 at most `0.9` times the corresponding 2/4 maximum;
- 4/8 spatial discrepancy at least 20 times the largest coarse/fine temporal
  L2 discrepancy across all three meshes;
- 4/8 total fission-power difference at most `1e-3` of initial power;
- full-state temporal preset discrepancy at most `1e-4`;
- matched initial power, fixed thermal meshes, and nonuniform relative flux
  with a span of at least `0.01`;
- stationary residual/rank checks, existing mass/energy/boron and
  neutron/precursor balances, frozen references and inventory mapping;
- exact eight-node fine JSON restart at `0.5 s`, complete rod arrivals and
  invalid-command rollback.

Balance checks are recomputed from retained snapshots. No convergence order
or continuum-error estimate is inferred from the three meshes. This package
also checks six negative controls on copies: corrupt energy, the sixth
precursor family, matrix completeness, diffusion coefficient, restart result
or frozen reference. Each must be rejected; native files are never overwritten.
The package
does not accept arbitrary scenarios or a cheaper replacement for the whole
model; NPP-12.04 remains open.

Accepted delivery: **111 scoped source tests and 66 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Maximum normalized power L2: **0.00949610 → 0.00468011** for 2/4 → 4/8. Maximum full-state temporal discrepancy was `1.55566e-07`; the receipt retains every time and precursor-family comparison.

## Local reproduction and limits

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component neutron-mesh \
  --output artifacts/neutron-mesh-local --phase-timeout-s 1200
```

Use a fresh output directory. Scoped tests and six trajectories run from
source, followed by sdist → wheel, a fresh local installation, and tests
and scientific calculations outside the checkout. Entire scientific JSON
documents must match. Local scientific dependencies are reused; no CI or
network is needed. Metrics and hashes are in the
[v133 receipt](benchmarks/neutron_mesh_v133.json).

Synthetic one-group nuclear data and illustrative feedback remain. This
is neither independent-code nor experimental validation. Closures,
thermal/fluid mesh refinement, boron transients, StationEnv and long-duration
regimes remain separate work. The full library suite and a new hour-long
run are not executed here. Root completion remains **58/76, with 18 open tasks**.
