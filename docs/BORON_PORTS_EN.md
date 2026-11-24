# Liquid boron-exchange ports: IF97 and operating limits — v140

Extension: [one-minute operation v141](BORON_DURATION_EN.md) checks continuous 1 kg/s flow to 60 s and JSON restart from 30 s, without physical source changes.

[Русский](BORON_PORTS_RU.md) · [Regenerator](BORON_REGENERATOR_EN.md)

A closed energy budget does not establish that letdown remains liquid.
The v139 exchanger used inlet heat capacities without qualifying outlet
phase states. This package adds an **opt-in IF97 liquid-port policy**;
it does not change heat transfer or inventories to force acceptance.

## Enable the policy

```python
from coupling.pwr_boron import BoronExchangeConfig, BoronExchangePWRSession
from coupling.refinement import coupled_pwr_mesh

session = BoronExchangePWRSession(
    **coupled_pwr_mesh(2),
    boron=BoronExchangeConfig(
        pressure_pa=1e6,
        regenerator_ua_w_k=500000.,
        require_liquid_ports=True,
    ),
    max_step_s=.05,
    rtol=2e-9,
)
result = session.advance(.1, {"boron_flow_target_kg_s": 50.})
ports = result["finite_boron_exchange"]["liquid_ports"]
```

These are research inputs, not plant equipment data. Higher receiver pressure
and stronger heat recovery support liquid outlets in the tested short case.

The default `require_liquid_ports=False` retains the earlier research model
and explicitly reports that phase checking is disabled. It receives no liquid
admissibility claim. The flag is a configuration policy, separate from physical
registry parameters and agent commands. It is available through
`exchange_policy()` and the snapshot field `boron_exchange_policy`; the numeric
registry retains its original schema. It requires an actual boolean.

## Port states

At positive flow, pressure and transported enthalpy determine six water states:
supply, pump outlet, primary return after recovery, primary donor, letdown before
the valve and letdown at receiver pressure. Pump work increases makeup enthalpy;
recovery adds heat to one stream and removes it from the other. The valve is
adiabatic and does no work, so enthalpy is unchanged across it. Flow is prescribed;
valve geometry, momentum and critical flow are not solved.

Every state must be IF97 region 1 with at least **1 K subcooling**. The policy
supports pressures from 0.01 to 16 MPa, with primary pressure above both tanks.
It requires enthalpy reconstruction within **0.01 J/kg**, nonnegative exchanger
endpoint approaches and nonnegative pure-water entropy production for pump,
exchanger and valve. This IF97 port entropy is distinct from the constant-capacity
entropy reported by v139.

Zero flow gives `active=False`, `checked=False` and no outlet states. Nonzero
recovered heat at zero flow is rejected.

## Rejection and restart

An unsupported port rejects the entire requested step. Time, state, controls,
history and energy counters roll back. Enthalpy is not clipped to saturation,
and vapor quality is not replaced with zero. Exact saturated liquid is outside
the positive-subcooling envelope too.

Checks apply to states sampled by the numerical solver. They **do not locate
a phase-onset event** or continue a two-phase branch. The matrix includes an
accepted prefix to 0.1 s followed by a rejected request to 0.5 s with exact
rollback; this is not a resolved flashing event.

The checkpoint binds policy, source and replay history. Changing the saved flag
is not migration: full replay comparison detects the substitution. Earlier
checkpoints are not automatically rebound.

## Numerical checks

Four strict trajectories cover 0/2000 ppm supply and two time settings on
2 neutron / 2 fluid / 4 thermal cells over 2.5 s. Both fine trajectories restart
exactly from 0.5 s. Two non-strict v139 cases and one rejected interval are
also exercised.

At observations, forward `(p,T)` properties check enthalpy states obtained by
inverse `(p,h)` calls. Acceptance also covers mass/energy/boron budgets, both
thermal ports, the independent counterflow boundary problem, time-setting
sensitivity and a frozen initial reference. Eight negative controls alter
copies only. At closing flow with UA 500000 W/K, the boundary reference
needed 2106 nodes. The initial 2000-node attempt stopped and is retained.
This profile now permits 4096 nodes at the unchanged `1e-9` tolerance;
the earlier v139 profile retains its 2000-node default. Physical and temporal
acceptance thresholds are unchanged.

The inverse-state cache holds at most 2048 exact `(p,h)` keys and immutable
thermodynamic states. Reports are fresh; keys are never rounded for reuse.

Accepted delivery: **92 scoped source tests and 72 installed-wheel tests**, eight successful phases and whole scientific JSON equal. All eight negative controls passed. Minimum observed subcooling was `35.074628 K`, endpoint approach `10.890017 K`; maximum enthalpy residual `9.8662e-08 J/kg`. Full-state temporal discrepancy was `2.27911e-06`. Both non-strict cases matched v139 bitwise. Applying the new observer to immutable v139 records finds region 4 at the letdown outlet under its previous inputs; the earlier numerical acceptance did not claim phase admissibility.

## Local reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-ports \
  --output artifacts/boron-ports-local --phase-timeout-s 1200
```

Use a fresh output directory. The profile runs tests, the scientific matrix,
sdist → wheel, installation outside the checkout, installed tests and the
complete scientific replay. Scientific dependencies are reused locally;
no CI or network is needed. [Receipt v140](benchmarks/boron_ports_v140.json).

## Limits

This checks port admissibility, not the entire CVCS. Remaining work includes
variable-property internal profiles, valve-throat cavitation and critical flow,
phase-event continuation, exchanger fluid/wall storage, chemistry, tank-mixing
entropy, long runs and plant data. The result does not cover all parameters
or spatial convergence of the active regenerator. The default StationEnv is
unchanged. The root plan remains **58/76, with 18 tasks open**.
