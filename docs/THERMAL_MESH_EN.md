# Interface-aligned fuel/cladding mesh — v134

[Русский](THERMAL_MESH_RU.md) · [Neutron refinement](NEUTRON_MESH_EN.md)

`coupling.thermal_grid.aligned_fuel_grid` builds a thermal mesh containing
every neutron-source and coolant-cell interface, then optionally subdivides
each interval. It returns a standard `FuelGrid` for
`LocalModeratorPWRSession`, without remapping state or modifying equations.

Both input meshes must have exactly matching endpoints. Nearby faces are not
merged by tolerance. Subdivisions must be an integer in 1..16, and the final
thermal mesh retains the existing sixteen-cell budget. Unresolved float64
intervals and budget overflow are rejected.

```python
import numpy as np
from atomic_model_station import CoreConfig
from coupling.pwr_channel import FluidGrid
from coupling.pwr_moderator import LocalModeratorPWRSession
from coupling.thermal_grid import aligned_fuel_grid

core = CoreConfig(kind="diffusion", cells=4)
fluid = FluidGrid()
fuel = aligned_fuel_grid(
    np.linspace(0., core.length_m, core.cells + 1),
    fluid.faces_m,
    subdivisions=2,
)
session = LocalModeratorPWRSession(core=core, fluid_grid=fluid, fuel_grid=fuel)
```

## The nonconforming diagnostic

The first series split the previous `(0, 0.7, 2.2, 4)` thermal mesh into
3/6/12 cells, holding four neutron cells and fluid faces `(0, 1, 2, 4)` fixed.
Thermal cells crossing source/fluid interfaces change their source and contact
averages under refinement. Maximum normalized cladding-temperature L2 for
3/6 and 6/12 was about `0.003331` and `0.003541`; distributed power gave
`0.005900` and `0.006161`. Those discrepancies did not contract.

**Monotone mesh convergence is not accepted from this series.** Its three
native trajectories and negative diagnostic are retained. Neither temporal
qualification nor restart was performed for this series. The previous
three-cell thermal mesh is not accepted as mesh independent.
This diagnostic series is outside the scientific JSON repeated from the wheel.
It can be reproduced separately with the library installed:

```python
import json
from pathlib import Path
import numpy as np
from atomic_model_station import CoreConfig
from coupling.pwr_fuel import FuelGrid
from coupling.pwr_moderator import LocalModeratorPWRSession
from scripts.qualify_moderator_transients import protocol

out = Path("artifacts/thermal-nonconforming-local")
out.mkdir(parents=True, exist_ok=False)
original = (0., .7, 2.2, 4.)
for split in (1, 2, 4):
    faces = [float(x) for a, b in zip(original[:-1], original[1:])
             for x in np.linspace(a, b, split + 1)[:-1]] + [4.]
    sim = LocalModeratorPWRSession(
        core=CoreConfig(kind="diffusion", cells=4),
        fuel_grid=FuelGrid(tuple(faces)), max_step_s=.05, rtol=2e-9,
    )
    initial = sim.snapshot()
    rows = [sim.advance(**step) for step in protocol("insertion", float(sim.state[12]))]
    (out / f"{len(faces)-1}.json").write_text(
        json.dumps(dict(initial=initial, snapshots=rows)), encoding="utf-8",
    )
```

## The replication limit

Current capacities and UA are constant per unit length, with no axial
conduction. Within one neutron cell and one fluid cell, source/contact
conditions are uniform. Subdividing an already aligned thermal cell into
identical children should preserve its temperature and reproduce its total
stored energy and power by summing children. This is a property of the
specific local closure.

The qualification uses aligned **4/8/16** thermal cells over 4 m, with four
neutron and three fluid cells fixed. Each mesh has its own hot stationary
solve at the same requested power and initial inventory constraints.
Stationary rod trims are retained, then each trajectory receives `+0.01`
inserted fraction from its own trim. Absolute commands and spatial initial
states need not be identical.

Observations occur at `0`, `0.1`, `0.5`, `1`, `2 s`. All meshes use
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`.
The sixteen-cell fine run performs an exact JSON restart at `0.5 s`, during
rod motion. There are six trajectories and one restart suffix.

## Profiles and energy identities

Nine separately compared fields are fuel/cladding temperatures, three
decay-heat reservoirs, fission power, total nuclear heat, fuel-to-cladding
heat and cladding-to-fluid heat. Exact overlap norms distinguish cell-average
temperatures from cell-integrated energies and powers. Temperature scales
are `1000 K` and `600 K`; powers use initial fission power per length,
and each reservoir uses its own initial inventory per length.

For each 4/8, 8/16 and 4/16 pair, maximum normalized L2 across observations
must be at most `1e-6` for every field. This is a numerical replication
contract, not physical uncertainty. Additional gates check:

- both capacity and both UA totals and per-length densities within `1e-12`;
- energy equations in each fuel/cladding cell, agreement of heat owners,
  and row/column sums of the contact matrix;
- cladding storage as `sum(C[i]*T[i])`, nonnegative entropy generation
  and its consistency with heat flows;
- previous global balances, stationary rank/residuals, frozen references,
  mass/boron mapping, rod arrivals and invalid-command rollback;
- total-power discrepancy at most `1e-6` of initial power, and full-state
  temporal preset discrepancy at most `1e-4`.

Energy identities are recomputed with normalized tolerance `1e-10`.
Matching totals cannot hide altered local capacities or an individual cell's
energy defect. Fuel temperature must have a span greater than 1 K, avoiding
acceptance based only on a uniform-field case.

Accepted delivery: **134 scoped source tests and 75 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Maximum normalized L2 across nine profiles and all three mesh pairs was `4.11075e-11`; full-state temporal discrepancy was `1.75162e-07`. Eight negative controls rejected corrupt local balances, heat pairs, capacity distribution, the third decay reservoir, reference, matrix, restart and entropy.

## Local reproduction and limits

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component thermal-mesh \
  --output artifacts/thermal-mesh-local --phase-timeout-s 1200
```

Use a fresh output directory. Eight phases cover scoped tests, calculations,
sdist → wheel, a fresh installation outside checkout, and repeated scientific
calculations with whole-JSON equality. Local dependencies are reused; no CI
or network is used. Metrics and hashes are in the
[v134 receipt](benchmarks/thermal_mesh_v134.json).

Replication acceptance does not establish generic thermal or full-plant mesh
convergence. Fluid discretization and geometry/inertia, applicable closures,
boron transients, StationEnv and long-duration regimes remain open.
Synthetic cross sections and illustrative feedback remain. No experimental
validation, new hour-long run or full-library suite is claimed.
Root completion remains **58/76, with 18 open tasks**.
