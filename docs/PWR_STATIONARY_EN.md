# Stationary normal-hot PWR channel start — v130

Update v131: [local density/boron response](PWR_MODERATOR_EN.md) is available in a separate class around a fixed hot reference. The original v130 stationary-start scope is retained below.

[Русский](PWR_STATIONARY_RU.md)

`coupling.pwr_stationary.StationaryChannelPWRSession` constructs an equilibrium
initial state for the [coupled PWR channel](PWR_CHANNEL_EN.md). It solves the
physical stationary equations rather than starting with a uniform coolant
inventory and allowing an artificial inlet-front settling transient. The
preceding lifted-start session remains available unchanged.

This is an opt-in research initialization, with the same illustrative nuclear
coefficients, constant-UA contact, fluid EOS and mechanical closures. It does
not calibrate constitutive parameters or reconstruct a real plant.

## What is solved

Primary liquid states are parameterized by pressure/temperature and converted
to owned M/U/B through IF97. Pressurizer, secondary and condenser inventories
use pressure and equilibrium quality. The solver also varies physical flows,
fuel/cladding temperatures, shaft and rotor states, cooling states, feed
inventory, neutron fluxes and the initial rod position.

Local six-family neutron precursors and three-family nuclear heat reservoirs
are eliminated with their exact stationary equations. Their full derivatives
are checked afterwards. Equilibrium heat reservoirs assume constructed steady
irradiation; they do not represent an independently measured operating history.
Existing aggregate aliases remain derived, not additional independent states.

Constraints retain the seed's primary total mass, secondary+hotwell+feed total
mass and circulating boron total. Uniform stationary circulating concentration
is obtained from the conserved boron and candidate mass, rather than preserving
concentration while silently changing boron mass. Noncirculating boron, inactive
makeup/drain tanks and backup energy retain their initial values.

Requested relative fission power and the existing pressurizer reference
pressure complete the normal-hot equilibrium specification. The initial rod
target is trimmed to the solved position; this control change is reported.
The pressure controller, pump curves, conductances, capacities and other
constitutive settings are not fitted.

The pressure reference replaces an adiabatic entropy anchor. Pressurizer
entropy and initial stored energy may change during construction; both are
reported. No physical preparation transient, heating expenditure or adiabatic
path from the lifted seed is claimed. Energy, mass and boron ledgers start from
the newly constructed initial state. Original `feed_design`/`cooling_design`
describe the retained sizing seed; `stationary_initialization` describes the
actual initialized state.

## Acceptance is independent of optimizer termination

The selected square system replaces dependent mass rows with inventory
anchors and includes the surge-line momentum equation even though its flow is
fixed to zero. Inactive pressurizer heating, spray and steam dumping must remain
inactive under their unchanged control settings.

Bounded least squares uses scaled variables and three-point finite differences.
After it terminates, the initializer checks **every physical derivative**, not
only the selected optimizer rows. Per-state dimensional tolerances and actual
rates are retained. Examples are 1 W for energy inventories, 1e-6 kg/s for mass,
1e-10 kg/s for boron and a 0.01 Pa equivalent momentum defect. Integral counters
may have nonzero rates and are explicitly excluded from physical stationarity.

The row/column-equilibrated constrained Jacobian must have full column rank,
with relative singular-value threshold 1e-9. Domain checks remain unchanged.
A deliberately injected leak in an omitted inactive-tank equation is rejected
by the full physical audit even when the selected optimizer equations converge.
Failure or budget exhaustion does not mutate the supplied seed session.

The normal-hot search is narrower than the dynamic model's entire admitted
domain. Primary trial pressures are 14–17.5 MPa, coolant temperatures 510–600 K,
fuel temperatures 600–1600 K and cladding temperatures 510–605 K. Pressurizer
pressure stays inside its existing deadband; secondary pressure stays below
dump opening. Arbitrary cold starts, active protection branches and every
combination of allowed component parameters are not qualified by this solve.
The configuration admits power requests 0.9–1.05; tested requests are 0.95 and 1.

```python
from coupling.pwr_stationary import StationaryChannelPWRSession, StationaryConfig

session = StationaryChannelPWRSession(stationary=StationaryConfig(relative_power=1.0))
result = session.advance(0.05)
report = result["stationary_initialization"]
restored = StationaryChannelPWRSession.restore(session.checkpoint())
```

`solve(seed, config)` returns a validated state/report without changing `seed`.
It refuses advanced or tripped trajectories. Use the session factory when a
replayable initialized simulator is required. JSON restore validates the
original channel checkpoint, reconstructs equilibrium from the same sources,
replays commands and requires exact final equality. Initialization reports and
checkpoints are privileged research data, not agent observations.

## Qualification and remaining work

Three cases cover one and three coolant cells at full requested power and
three cells at 95%, with two neutron cells and three fuel/cladding cells.
Two numerical presets produce six native trajectories. Each holds equilibrium
for 0.1 s, then applies a 0.99 load request for 0.02 s and SCRAM for 0.02 s.
Three fine-preset restart suffixes are checked exactly. Coarse-preset restart
is not claimed. Nine initializations include the restored fine runs.

Tests also cover the default four-neutron-cell configuration, full rank,
dimensional residuals, inventory constraints, explicit entropy/energy changes,
hold drift, exact JSON restart, malformed metadata, budget failure and refusal
to reinitialize a running trajectory. A small initial residual is numerical
evidence, not a physical uncertainty or experimental-validation bound.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_component_delivery \
  --component pwr-stationary --output artifacts/my-pwr-stationary-delivery \
  --phase-timeout-s 1200
```

Accepted: **158 scoped source tests and 91 installed-wheel tests**, eight delivery phases. Maximum preset difference `1.1199809e-06`, hold drift `2.6035255e-10` under `abs(a-b)/max(abs(b),1)` in native SI units. Maximum initial physical tolerance ratio `0.00041708071` against criterion 1. Entire scientific JSON matches.

The offline component delivery builds a clean sdist/wheel, verifies its exact
contents, installs outside checkout and compares the entire scientific JSON.
Dependencies are reused; this is not a new physical runner or a full-library
test-profile run. The generated stationary matrix has its own finite 8 MiB
reader bound; the existing profiles retain their previous bounds.

Stationary initialization is accepted only for the documented research cases.
NPP-12.03 still needs qualified closures, local moderator/boron feedback,
spatial convergence and long-duration qualification. Full shutdown, coastdown,
DHR, StationEnv integration and real-installation validation remain open.
The plan stays 58/76, with 18 open tasks. See the
[v130 receipt](benchmarks/pwr_stationary_v130.json).
