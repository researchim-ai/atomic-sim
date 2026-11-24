# Finite primary boron exchange — v137

Optional extension: [IF97 liquid-port checking v140](BORON_PORTS_EN.md). Earlier results do not qualify outlet phase states; strict mode rejects unsupported two-phase ports with full rollback.

Optional extension: [regenerative heat recovery v139](BORON_REGENERATOR_EN.md). Default UA is zero; earlier source-bound checkpoints are not automatically rebound.

[Русский](PWR_BORON_RU.md) · [Coupled PWR](COUPLED_MESH_EN.md)

`coupling.pwr_boron.BoronExchangePWRSession` adds separate finite supply and
receiver tanks to the local-moderator PWR session. Supply enters the cold leg
while an equal mass flow leaves it. Primary inventories are never reassigned;
mass, energy and boron move between finite owners. Secondary feed-train tanks
are not used for this primary exchange.

```python
from coupling.pwr_boron import BoronExchangeConfig, BoronExchangePWRSession
from coupling.refinement import coupled_pwr_mesh

session = BoronExchangePWRSession(
    **coupled_pwr_mesh(2),
    boron=BoronExchangeConfig(supply_boron_ppm=0.),
)
result = session.advance(0.1, {"boron_flow_target_kg_s": 50.})
exchange = result["finite_boron_exchange"]
```

The command sets target flow rather than concentration. `supply_boron_ppm`
sets the initial composition of a finite tank; direct runtime ppm assignment
is unsupported. Boron reaches neutron feedback through actual loop transport
and local coolant/boron inventories.

## Inventories and energy

Added states are M/U/B in each tank, flow-actuator state and a pump electrical
energy counter. The counter is diagnostic, not another stored-energy owner.
For q ≥ 0:

```text
dM_supply/dt = -q             dM_receiver/dt = q
dB_supply/dt = -q*c_supply    dB_receiver/dt = q*c_cold
dB_cold/dt += q*(c_supply-c_cold)
dM_cold/dt += 0
```

Whole-system mass and boron include both tanks. The former primary-only
boron ledger is replaced by the combined-owner check in this adapter.

Tank liquid uses a fixed reference density and cp = 4180 J/(kg K); gas
cushions are adiabatic with gamma = 1.4. Reference density/internal energy
come from IF97 at the declared initial state. This linear liquid closure
is not a new boron-solution EOS:

```text
gas_volume = gas_volume_initial + (M_initial-M)/rho_reference
p = p_initial*(gas_volume_initial/gas_volume)**1.4
E_gas = p*gas_volume/(1.4-1)
h = U_water/M + p/rho_reference
dU_water/dt = enthalpy_flow - p*dM/dt/rho_reference
```

Gas energy is counted once. Its derivative restores the appropriate enthalpy
flux when added to water storage.

The pump supplies `q*(p_cold-p_supply)/rho_reference` hydraulic power;
electrical power divides this by efficiency. All electrical input enters
the primary stream as hydraulic work and loss heat. Own needs increase,
grid export decreases, and the global external-energy counter uses the same
boundary change. Letdown retains incoming enthalpy in the receiver.

## Actuation and limits

Flow obeys `dq/dt=(target-q)/tau`. Its exact solution is eliminated from the
numerical state on each constant-command interval. Inventories are not
projected. A zero target leaves a decaying transfer tail, included in budgets.

An active pump currently requires an available external grid. Power loss
at nonzero flow rejects the complete step with rollback. Pump coastdown,
backup power and valve switching are not modeled. Finite reserve, gas
volume, temperature/pressure range, saturation margin and pump head are
checked. Exhaustion/domain failure rejects a step without clipping stocks
or adding hidden makeup.

No line inertia or jet-mixing geometry is included. Stationary initialization
and rank audit describe the base PWR; tanks are separately initialized and
inactive at the start. Their constant states are not presented as a new
full-rank audit. Registered parameters are illustrative with uncertainty=None.

## Numerical qualification

Supply cases at 0 and 2000 ppm have matched initial primary states, temperature,
equipment and flow commands. Each tank starts at 5000 kg with a 5 m³ gas
cushion, 350 K and 0.2 MPa; efficiency is 0.85, target flow 50 kg/s and tau
0.2 s. Initial primary concentration comes from the constructed equilibrium,
without replacing it by a round nominal value.

The scenario lasts 2.5 s: command opening at zero, closing at 2 s, observations
at `0.1`, `0.5`, `1`, `2`, `2.5 s`. Both cases use
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`.
Each fine trajectory has an exact JSON restart at 0.5 s: four trajectories
and two suffixes in total.

Gates require existing global/neutron budgets, new transfer identities,
tank depletion against the exact actuator integral within `1e-5 kg`, correct
primary-boron direction, frozen references, rollback and full-state temporal
agreement within `1e-4`. Chemical-case power difference must exceed `1e-4`
of initial power and twenty times temporal preset sensitivity. This is an
internal comparison using illustrative boron coefficients, not validation
against physical measurements.

Accepted delivery: **93 scoped source tests and 36 installed-wheel tests**, eight successful phases and whole scientific JSON equal. Chemical-case power effect was `0.09827625` of initial power; numerical power sensitivity was `1.5293e-09`, and full-state discrepancy `5.55075e-06`. Primary-boron changes and raw trajectories are retained. Nine negative controls rejected corrupt transfer/energy, wrong direction, reference/restart changes and absent chemical signal.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-exchange \
  --output artifacts/boron-exchange-local --phase-timeout-s 1200
```

Use a fresh output directory. Local tests, sdist → wheel, a fresh installation
outside checkout and whole scientific JSON equality are required. No CI or
network is used. The [v137 receipt](benchmarks/pwr_boron_v137.json) retains
metrics, hashes and preparation attempts.

This is a separate research component. Plant CVCS reconstruction, solubility,
chemical kinetics, detailed mixing, new nuclear data, spatial/long-duration
boron acceptance, StationEnv and experimental validation remain open.
No new hour-long run or full library suite is executed.
Root status remains **58/76, with 18 open tasks**.
