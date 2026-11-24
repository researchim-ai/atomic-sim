# Distributed liquid channel — v128

Update v129: [shared channel and contact operators connect to a separate PWR session](PWR_CHANNEL_EN.md). The standalone v128 component scope is retained below.

[Русский](COOLANT_CHANNEL_RU.md)

`coupling.coolant_channel.CoolantChannel` is a separate horizontal channel
component for the next stage of NPP-12.03. Each fluid cell owns its mass,
internal energy and dissolved-boron mass. Each face owns a signed mass flow.
A second, possibly nonmatching mesh owns wall temperatures and heat capacities.
This component is not yet connected to PWR or StationEnv.

The implementation reuses the existing `physics.hydraulics.network` equations:
mass and boron continuity, donor-enthalpy transport, and lumped face momentum
`(L/A)*dq/dt = p_up - p_down - K*q*abs(q)/(2*rho*A²)`.
Cell volume is `A*dx`. Face inertance lengths are distances between cell centres
or between a boundary and its adjacent centre. The declared total loss
coefficient is distributed by those lengths. The channel is horizontal, with
no pump, check valve, gravity term or advective momentum-flux equation.

Pressure and temperature follow from each cell's actual `(M/V,U/M)` through
IF97; they are not averaged from a parent volume. Ports prescribe liquid
pressure, temperature and boron concentration. Incoming enthalpy/boron use the
boundary state, while outgoing values use the adjacent cell. Reversed flow
changes the donor naturally, without clipping the flow to zero.

The supported fluid domain is compressed-liquid region 1, 1–20 MPa and
300–600 K. Every wall must remain at least 300 K and below saturation in every
overlapping coolant cell. Boiling and post-CHF states are rejected. The wall
conductances are declared constant parameters, not boiling or Reynolds-based
correlations. Dissolved boron is a passive dilute tracer; its effect on the
water EOS is omitted.

## Heat transfer across different meshes

For overlap fractions `P[i,j]` from wall cell j to fluid cell i, contact uses
the individual conductances `K[i,j] = P[i,j]*UA[j]` and heat transfers
`Q[i,j] = K[i,j]*(T_wall[j]-T_fluid[i])`.
Each wall loses its column sum and each fluid cell receives its row sum.
External wall heating enters `C_wall*dT_wall/dt` once.

Mapping only the net wall heat would lose simultaneous opposite transfers:
a 550 K wall overlapping equal 540 K and 560 K fluid cells has zero net heat,
but still exchanges heat with both. Pairwise conductances retain both transfers
and nonnegative contact entropy generation
`sum(K[i,j]*(T_wall[j]-T_fluid[i])²/(T_wall[j]*T_fluid[i]))`.
This diagnostic covers conductive contact, not the full channel entropy budget.

The energy ledger includes fluid internal energy and wall storage; mass and
boron have independent boundary ledgers. The inherited low-Mach network energy
approximation excludes flow kinetic-energy storage. Passing its ledger is not
acceptance of a full total-energy compressible-flow model.

## Numerical acceptance

The implicit Radau solver uses a fixed relative finite-difference Jacobian,
with an admitted one-sided probe at a domain boundary. Exact source states are
never clipped or projected. Inverse-EOS initial guesses are fixed by the initial
configuration, so identical inventories produce the same RHS after unrelated
Jacobian probes. The exact-key property cache remains in use. This removed a
resting-state defect without introducing a small-flow cutoff or relaxing EOS.

Each interval commits only after endpoint admission and cumulative balance
checks. A failed power/port request or late physical rejection preserves states,
ports and history. A 30000-RHS-call budget bounds interval work. Intervals are
1 microsecond–0.1 s, with a 10 s/256-interval component budget. These limits are
work bounds, not accepted long-duration operation.

```python
from coupling.coolant_channel import CoolantChannel, ChannelConfig, Port

channel = CoolantChannel(ChannelConfig(initial_wall_temperature_k=552.0))
result = channel.advance(0.005, [10000.0, 20000.0, 5000.0])
restored = CoolantChannel.restore(channel.checkpoint())
```

JSON checkpoints bind source hashes, initial ports, geometry, configurations,
commands and final snapshot. Changed ports are replayed from the original
ports; a resealed but inconsistent final snapshot is refused. Research
snapshots expose physical cell states and are not agent observations.

Six cases cover exact rest, forward/reverse boron transport, heated three/five
fluid cells with three nonmatching wall cells, and pressure-driven reversal.
Two presets give 12 trajectories and 12 exact restart suffixes, each over
0.01 s. Four tracer results are checked against the analytical matrix
exponential of the lossless isothermal compartment equations. Those exact
references qualify tracer transport in that special case, not heated hydraulics.
Three/five-cell cases are not a spatial-convergence proof.

Maximum preset difference is `6.0625e-10` under `max(abs(a-b)/max(abs(b),1))` in native SI units. Exact-tracer error is at most `1.735e-18 kg`; accepted rest-state change is zero. These are numerical metrics, not physical model-error or uncertainty bounds.

Accepted delivery: **98 scoped source tests**, **10 additional EOS/profile
checks**, and **74 tests from an installed wheel**. Eight delivery phases
build an sdist, build its wheel in a clean directory, audit exact contents,
install outside checkout and repeat the complete scientific JSON with
`python -I`. All scientific results match without field exclusions.
Dependencies are reused offline; the full library profile was not rerun.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_component_delivery \
  --component coolant-channel --output artifacts/my-coolant-channel-delivery
```

The shared delivery runner also preserves the previous
`scripts.verify_pwr_local_cladding_delivery` entry point. Failed preparation
attempts are retained separately from the accepted frozen-source delivery.

NPP-12.03 stays open. Next requirements are PWR port/initialization integration,
local density/boron feedback, applicable heat-transfer/friction closures and
long-duration qualification. No real installation, experimental validation,
StationEnv integration or industrial superiority is claimed. The plan remains
58/76, with 18 open tasks. See the
[sealed v128 receipt](benchmarks/coolant_channel_v128.json).
