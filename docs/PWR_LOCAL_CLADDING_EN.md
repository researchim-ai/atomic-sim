# Local PWR cladding storage — v127

Update v129: [local coolant connects to a separate PWR session](PWR_CHANNEL_EN.md). The original v127 cladding-session scope is retained below.

Next component v128: [a distributed liquid channel](COOLANT_CHANNEL_EN.md) is qualified separately; it is not yet attached to this PWR session.

[Русский](PWR_LOCAL_CLADDING_RU.md)

`coupling.pwr_cladding.DistributedCladdingSession` extends the reduced
[distributed-fuel PWR](PWR_DISTRIBUTED_FUEL_EN.md) with one cladding temperature
per fuel cell. Fuel, local decay heat, cladding, common coolant and the inherited
steam/electrical plant evolve together. This opt-in class does not change
StationEnv or either existing distributed-fuel session.

For cell length fraction `w[j]`, the capacities and constant conductances are
the inherited totals multiplied by `w[j]`. The cell equations are

```text
q_fc[j] = UA_fc[j] * (T_f[j] - T_c[j])
q_cw[j] = UA_cw[j] * (T_c[j] - T_water)
C_f[j] * dT_f[j]/dt = Q_nuclear[j] - q_fc[j]
C_c[j] * dT_c[j]/dt = q_fc[j] - q_cw[j]
```

The common coolant receives `sum(q_cw)`. The adapter replaces the old thermal
rates and adds the difference between new and inherited coolant heating to its
energy derivative, retaining floating-point differences rather than silently
discarding them. These are internal transfers: the external boundary remains
fission power with the local decay-heat reservoirs included in stored energy.

The old scalar cladding temperature becomes a capacity-weighted alias, excluded
from the independent integration variables. The inherited stored-energy formula
already equals `sum(C_c[j]*T_c[j])`; the new local storage is not added a second
time. The local cladding variables remain active Jacobian columns. Local fuel
temperatures continue to feed the conservative spatial neutron coupling.

Each local wall must remain at least 300 K and strictly below the saturation
temperature at the current common core pressure. An inadmissible local wall is
rejected even when the average remains admissible. These checks retain the
inherited single-phase scope; they do not implement boiling or CHF. Per-cell
storage residuals and conductive entropy generation are exposed in research
snapshots. A failed interval does not commit controls, events or local states.

```python
from coupling.pwr_cladding import DistributedCladdingSession

session = DistributedCladdingSession()
result = session.advance(0.05, {"load_fraction": 0.99})
local_walls = result["distributed_cladding"]["temperature_k"]
restored = DistributedCladdingSession.restore(session.checkpoint())
```

Checkpoints bind both adapter sources, geometry, configurations and history.
They replay through the public interval API and require exact final equality.
Fuel-only and local-cladding checkpoints are deliberately distinct. Snapshots
and checkpoints are privileged research data, not agent observations.

Qualification uses three illustrative cases: uniform, biased with three fuel/
cladding cells, and biased with five cells, all with four neutron cells. Six
native trajectories use two numerical presets over a 0.15 s hold/load-change/
SCRAM sequence. Six restart suffixes and invalid-action rollback are checked.
The uniform trajectory is compared with the preceding distributed-fuel model.
For each case, a separate full-state BDF integration of the same physical RHS
is compared with reduced Radau over an initial 0.02 s hold. This is a numerical
cross-check, not an independent physical model or experimental measurement.

Tests also construct the local thermal capacitance/resistance matrix
independently and compare its derivative at a nonuniform state. They cover
energy ownership, local wall limits, entropy, source/configuration/checkpoint
corruption, and rollback after a late rejection. Raw failed preparation attempts
are retained; accepted results refer to the final frozen source manifest.

The reproducible scoped delivery command builds an sdist, builds its wheel in
a clean directory, audits the wheel contents and installs it outside checkout.
It repeats tests and the full scientific JSON using `python -I`, without the
checkout import path. Scientific dependencies are reused offline.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_pwr_local_cladding_delivery \
  --output artifacts/my-local-cladding-delivery
```

This command executes a declared component test selection; it does not run the
full library profile. The earlier 3580-test library acceptance belongs to
v125–126 and is not reused as a full-profile acceptance of this source epoch.

NPP-12.03 remains open. The core coolant is still one mass/energy/momentum
system, with no local fluid channels or local density/boron feedback. There is
no axial conduction, reconstructed rod geometry, local boiling/CHF closure,
staged-film adapter or StationEnv binding here. The existing constant-UA
parameters and local length weights are illustrative. Long SCRAM, coastdown,
decay-heat removal and experimental validation remain unaccepted. Three/five
cell cases do not establish spatial convergence of a channel model.

Accepted delivery: **141 scoped source tests and 88 installed-wheel tests**, eight successful phases. Maximum preset difference `3.5024e-8`, BDF difference `4.2623e-7`, uniform-limit difference `6.2169e-11`. Component normalization is `abs(a-b)/max(abs(b),1)` in native SI units, not a physical error bound.

See the [sealed v127 receipt](benchmarks/pwr_local_cladding_v127.json).
The plan remains 58/76 with 18 open tasks.
