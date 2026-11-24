# Complete dual geometry for the liquid channel — v135

[Русский](FLUID_GEOMETRY_RU.md) · [Thermal mesh](THERMAL_MESH_EN.md)

`FluidGrid(momentum_geometry="complete_dual")` includes boundary half cells
in the added channel's inertance, pressure loss and reference kinetic energy.
It works with `ChannelPWRSession` and its stationary/local-moderator subclasses.
The previous geometry is named `legacy` and remains the default. Existing
flow owners, actuator commands and physical admission limits are retained.

```python
import numpy as np
from atomic_model_station import CoreConfig
from coupling.pwr_channel import FluidGrid
from coupling.pwr_fuel import FuelGrid
from coupling.pwr_moderator import LocalModeratorPWRSession

session = LocalModeratorPWRSession(
    core=CoreConfig(kind="diffusion", cells=4),
    fuel_grid=FuelGrid(tuple(np.linspace(0., 4., 9))),
    fluid_grid=FluidGrid(
        faces_m=tuple(np.linspace(0., 4., 5)),
        internal_loss_coefficient=1.,
        momentum_geometry="complete_dual",
    ),
)
geometry = session.snapshot()["momentum_geometry"]
```

In the new mode, the historical `internal_loss_coefficient` name denotes
the total extra K over the **whole channel**, including boundary halves.
Its legacy meaning remains internal links only. A single-cell complete-dual
channel supports nonzero K through its two boundary halves. The fluid
state budget remains 1–8 cells.

## Geometry and boundary ownership

Previously the added internal links spanned the first to last cell centres:
`L*(1-1/N)` on a uniform mesh. Added channel inertance therefore changed
with N; a single cell had no internal links.
`coupling.hydraulic_geometry.dual_geometry` includes inlet-to-first-centre,
centre-to-centre and last-centre-to-outlet distances. On uniform or nonuniform
meshes their sum is L. At constant channel area A:

```text
I_face = dual_length / A
sum(I_face) = L / A
K_face = K_total * dual_length / L
sum(K_face) = K_total
```

The previous boundary links are treated as external equivalent components.
If supplied dimensions already include core segments, a plant-specific
geometry decomposition is needed. No measured PWR/VVER geometry is inferred.

Each half cell is combined with its existing pump/outlet flow owner:

```text
I_old = L_old / A_old
I_half = half_length / A_channel
I_effective = I_old + I_half
L_effective = A_old * I_effective

I_effective / rho_reference_effective
    = I_old / rho_reference_old + I_half / rho_reference_channel

K_effective = K_old + K_half * (A_old / A_channel)**2
```

The effective reference density belongs to the inherited kinetic-energy
closure; it **does not replace EOS water density**. Actual donor density
still enters thermodynamics, transport and friction. K conversion preserves
pressure loss when channel and legacy areas differ. No duplicate boundary
flow or energy state is introduced.

For uniform flow the added kinetic energy is
`0.5 * L/A * q**2 / rho_reference_channel`, independent of N. Nonuniform flow
keeps individual face contributions and their compensating thermal work.
This remains a **fixed-reference-density** kinetic closure, not a complete
variable-density mechanical-energy model.

Fluid snapshots now include `face_acceleration_kg_s2` and
`face_driving_pressure_pa`, allowing `I*q_dot = driving_pressure` checks.
The geometry passport retains legacy and combined boundary coefficients.
These are diagnostics, not additional inventories.

## Conditional fluid-mesh qualification

The native matrix uses 2/4/8 uniform fluid cells over 4 m, fixed four neutron
and eight thermal cells, and channel K = 1. Separate hot stationary solves
use the same requested power and initial inventory anchors. Rod trims are
retained, followed by the same `+0.01` inserted-fraction increment from each
trim. Absolute commands and spatial initial states need not be identical.

Each mesh runs with `max_step_s=0.1, rtol=2e-8` and
`max_step_s=0.05, rtol=2e-9`. Observations are at `0`, `0.1`, `0.5`, `1`, `2 s`.
The eight-cell fine run has an exact JSON restart at `0.5 s`, during rod
motion: six trajectories and one restart suffix in total.

Exact overlap norms compare temperature, pressure, density and cell-integrated
mass/internal-energy/boron inventories. Each 4/8 maximum L2 must be at most
0.9 times its 2/4 maximum and at least 20 times the corresponding temporal
preset discrepancy. Scales and per-observation errors are retained.

Normalized L2 budgets are 0.02 for inventories/density, 0.01 for temperature
at a 600 K scale, and 0.001 for pressure at 15.5 MPa. Other gates require
full-state temporal discrepancy ≤ `1e-4`, 4/8 total-power discrepancy ≤ `1e-3`
of initial power, previous budgets, stationary rank/residuals, frozen
reference, rod arrival and rollback. These are numerical budgets for this
case, not physical uncertainty estimates.

Actual working-link coefficients must preserve added inertance, uniform-flow
reference kinetic coefficient and pressure-loss coefficient within `1e-12`
relative defect. Momentum residual is limited to `1e-5 Pa`; normalized kinetic
work defect to `1e-10`. Tests also cover nonuniform meshes, one cell, unequal
boundary areas and unequal reference densities.

Accepted delivery: **114 scoped source tests and 67 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Temperature L2 for 2/4 → 4/8 was `0.00859940 → 0.00439547`. Maximum geometry defect was `5.15572e-15`, momentum residual `2.14236e-10 Pa`, and full-state temporal discrepancy `1.73856e-07`. Nine negative controls rejected corrupt inertance, kinetic storage, loss, acceleration, kinetic work, energy, reference, restart and matrix completeness.

## Local reproduction and limits

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component fluid-geometry \
  --output artifacts/fluid-geometry-local --phase-timeout-s 1200
```

Use a fresh output directory. Scoped tests and scientific runs are followed
by sdist → wheel, fresh installation and replay outside checkout with whole
scientific JSON equality. Local dependencies are reused; no CI or network
is used. Results and hashes are in the [v135 receipt](benchmarks/fluid_geometry_v135.json).

A separate legacy regression compares five retained physical states against
the v134 scenario, requiring bitwise equality. This is one numerical regression
case, not checkpoint migration. Adapter source changed; old source-bound local
PWR checkpoints are not accepted automatically.

Qualification is conditional on single-phase forward liquid flow, horizontal
equivalent geometry, constant UA and synthetic neutron data. Continuum error,
combined all-mesh spatial acceptance, experimental validation, StationEnv and
long-duration regimes remain open. No new hour-long run or full library suite
is executed. Root completion remains **58/76, with 18 tasks open**.
