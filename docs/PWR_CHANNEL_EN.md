# Local coolant connected to PWR — v129

Update v130: [stationary normal-hot initialization](PWR_STATIONARY_EN.md) is available as a separate class. The original lifted-start v129 scope is retained below.

[Русский](PWR_CHANNEL_RU.md)

`coupling.pwr_channel.ChannelPWRSession` connects local coolant inventories to
the existing reduced PWR, distributed fuel, three local decay-heat groups and
local cladding. All states advance in one implicit ODE solve. The new session
is opt-in; StationEnv and the existing session defaults are unchanged.

## State ownership and ports

Each coolant cell owns mass M, internal energy U and dissolved-boron mass B.
The former core M/U/B slots are sums of these local owners, excluded from the
independent integration variables. They provide the legacy plant's aggregate
thermodynamic interface and inventory accounting. Local fluid energy is not
added a second time to the stored-energy ledger.

The original pump and core-outlet flow variables remain the only boundary
momentum owners. Internal faces add their own flow variables. The inlet uses
the cold-leg enthalpy and boron; the outlet uses the last cell's actual state.
Corrections to the hot-leg derivatives replace the former aggregate donor.
Boundary momentum uses the actual first/last-cell pressures, outlet donor
density and the inherited arithmetic-density hydrostatic law.

Contact with cladding uses the
[pairwise conductance operator](COOLANT_CHANNEL_EN.md), including when wall
and fluid meshes differ. The inherited local fuel temperatures continue to
drive spatial Doppler feedback. Moderator-density feedback still uses the
cold leg, and boron feedback the aggregate core concentration. Local
density/boron reactivity feedback is not implemented by this adapter.

## Mechanical energy is part of the connection

The elevated PWR already stores hydraulic kinetic energy using fixed initial
reference densities. Changing its boundary accelerations also changes kinetic
power. The adapter adjusts the compensating thermal exchange in the cold leg,
hot leg and endpoint cells accordingly.

Each internal face adds kinetic storage
`E_k = 0.5*(L/A)*q²/rho_ref`. Its derivative is subtracted equally from the two
adjacent cells' thermal-energy rates. This extends the parent's declared
reference-density closure; it does not promote the standalone channel's
internal-energy approximation into a full compressible-flow total-energy law.
The global ledger also retains fuel/cladding storage, local nuclear heat
reservoirs, gravity, shafts, cooling, feed and electrical accounts. Fission
power remains the external nuclear-energy input.

Local cells lie at the original core elevation, with zero elevation change
inside the channel; the external elevated loop remains. Equivalent area is
the inherited 18 m³ core volume divided by the neutron/fuel span (4 m by
default). Original boundary-link inertances and losses remain. Optional
internal loss is additional and is explicitly disclosed, not silently fitted
or redistributed. This is constructed geometry, not a rod-bundle reconstruction.

## Initial state and admitted domain

The solved lumped PWR initial core is partitioned by cell length, preserving
aggregate M/U/B within floating-point arithmetic. Internal flows interpolate
between the original boundary flows. New internal kinetic energy is included
in the initial energy account. The lift defect is reported.

This is an explicit transient initial condition, **not a stationary heated
channel solution**. A temperature profile develops as cold-leg water enters
the initially uniform core. The old backend's accepted nominal hour is not
reused as acceptance of this model.

Fluid must remain compressed-liquid IF97 region 1, 1–20 MPa and 300–600 K.
Each cladding cell must stay at least 300 K and strictly below saturation in every
overlapping fluid cell. The PWR adapter admits forward flows only, including
internal faces; the standalone component's reverse-flow support does not
qualify a reverse primary-pump quadrant. Constant-UA contact supports either
heat-flow direction within the temperature domain. No boiling/CHF, staged
film closure or mixed-convection qualification has been added.

```python
from coupling.pwr_channel import ChannelPWRSession, FluidGrid

session = ChannelPWRSession(fluid_grid=FluidGrid((0.0, 1.0, 2.0, 4.0)))
result = session.advance(0.005, {"load_fraction": 0.99})
temperatures = result["distributed_fluid"]["temperature_k"]
restored = ChannelPWRSession.restore(session.checkpoint())
```

Snapshots/checkpoints expose privileged physical data, not agent observations.
JSON restart binds both the original PWR initial checkpoint and the new grid,
adapter sources, command history and final snapshot. Restore replays the
commands and requires exact equality. Incorrect commands and late physical
rejections do not commit partial fluid states, controls or events.

## Qualification scope

Four cases cover one, three and five coolant cells and a spatially biased
three-cell case with additional internal loss. Four neutron and three fuel/
cladding cells remain fixed. Two presets give eight native trajectories and
eight exact restart suffixes over 0.06 s: hold, 0.99 load request, SCRAM request.
The one-cell limit is compared with the preceding complete cladding-PWR
session. Refinement compares the complete coupled state, including local
precursors, heat reservoirs, fluid states, internal flows and integral counters.

Tests independently check endpoint momentum, last-cell enthalpy/boron export,
internal kinetic-energy ownership, energy compensation, local domain failures,
checkpoint corruption and late rollback. These checks establish a short
research coupling, not a full shutdown or an industrial operating envelope.

Accepted: **199 scoped source tests and 107 installed-wheel tests**, eight delivery phases. Maximum preset difference `9.7925969e-08`, one-cell limit difference `8.483524e-11`. Normalization is `abs(a-b)/max(abs(b),1)` per component in native SI units, not a physical uncertainty bound.

The declared delivery profile builds an sdist and its clean wheel, checks the
exact installed files, runs tests outside checkout, and compares the entire
scientific JSON with isolated installed execution. No scientific fields are
excluded. Scientific dependencies are reused offline.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_component_delivery \
  --component pwr-channel --output artifacts/my-pwr-channel-delivery \
  --phase-timeout-s 1200
```

NPP-12.03 remains open: stationary distributed initialization, qualified
closures, local moderator/boron feedback, spatial convergence and long-duration
acceptance still require work. Neither real-installation validation nor
StationEnv integration is claimed. The full library profile has not been
rerun for this source epoch. The plan remains 58/76 with 18 open tasks.
See the [v129 receipt](benchmarks/pwr_channel_v129.json).
