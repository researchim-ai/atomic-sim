# Closing finite boron exchange — v142

Follow-up: [finite dose planning and atomic execution v143](BORON_DOSING_EN.md).

[Русский](BORON_CLOSURE_RU.md) · [Continuous minute](BORON_DURATION_EN.md)

This package checks the finite closing tail after one minute of boron
exchange. The 1 kg/s command is replaced with zero at 60 s; actual flow
retains the 0.2 s actuator time constant. Physical equations, checkpoint
format and the initial feedback reference are unchanged.

## Protocol

Supply concentrations of 0 and 2000 ppm are run with two integrator settings:
`max_step_s=0.2, rtol=2e-8` and `max_step_s=0.1, rtol=2e-9`. Each initial tank
contains 5000 kg at 350 K and 1 MPa. Regenerator UA is 500000 W/K, liquid
IF97 port checks are enabled, and the joint mesh is 2/2/4.

Minute observations are extended by 60.1, 60.2, 60.5 and 61 s. The 60 s
snapshot precedes the closing command. For `s=t-60 > 0`:

```text
q(60+s) = (1-exp(-300))*exp(-s/0.2) kg/s
m(60+s) = 59.8 + (1-exp(-300))*0.2*(1-exp(-s/0.2)) kg
```

At 61 s, transferred mass is approximately **59.99865241 kg** and flow is
still **0.006737947 kg/s**. The remaining analytic integral to infinity is
0.001347589 kg; it is not assigned to the simulated tanks. The mass reference
rejects times outside 0–61 s.

Both fine runs JSON-roundtrip a checkpoint at 60.2 s during closure. Restore
replays the history and exactly reproduces the two remaining intervals,
snapshots and final checkpoint. An invalid flow command checks full rollback.

## Independent small-flow heat reference

The previous boundary-value reference reaches its mesh limit at high
UA/(q Cp). The new reference integrates the normalized gap in its stable,
decaying direction: `e'=-abs(k)*e`, `J'=e`, initially `(1,0)`, where
`a=UA/Ch` and `k=UA*(1/Cc-1/Ch)`. The resulting profile is rescaled to satisfy
counterflow conditions `theta_h(0)=1`, `theta_h(1)=d(1)`.

Radau uses `rtol=1e-11`, `atol=1e-13`, with an accepted mesh budget of 8192
nodes. Boundary and integral residuals are checked. Failure, nonfinite output
or an exceeded budget rejects the reference. No effectiveness formula or
analytic exponential computes reference heat. The actuator mass reference
uses its own independent analytic solution.

Tests cover both capacity orderings, equal and nearly equal capacities,
reverse heat transfer, temperature translation and degenerate limits.
Where the earlier BVP converges, both independent numerical representations
are compared as well.

## Acceptance and reproduction

All observations check mass, energy, boron, water/gas tank inventories, pump
energy, thermal ports, IF97 states, port entropy, chemical direction and
unchanged feedback reference. Full-state temporal discrepancy must remain
below `1e-4`. The power effect must exceed `1e-4` of initial power and twenty
times temporal noise. A separate gate checks exact actuator flow and commands.
Nine controls mutate copies, including clipped tail flow, a lost closing
command, mass, heat, EOS policy, time, restart and energy failures.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-closure \
  --output artifacts/boron-closure-local --phase-timeout-s 1200
```

Use a fresh directory. The profile runs scoped tests and scientific cases,
builds sdist → wheel, installs outside the checkout, and repeats tests and the
whole scientific JSON. Local scientific dependencies are reused; no CI or
network is required.

Accepted delivery: **132 source and 112 installed-wheel tests**, eight successful phases, whole scientific JSON equal and all nine negative controls detected. Maximum full-state temporal discrepancy: `5.131545e-05`; minimum subcooling: `55.863861 K`. All 24 minute observations match v141; an archived checkpoint restores and continues to the exact final state and checkpoint. [Machine receipt](benchmarks/boron_closure_v142.json).

## Scope

Acceptance ends five actuator time constants after the closing command.
Exact zero flow, arbitrarily small flows, hour-long operation, depleted tanks,
power loss, phase-event localization and two-phase continuation remain open.
The regenerator has no dynamic wall/fluid inventories and uses inlet heat
capacities. IF97 checks ports, not the internal profile or valve throat.
There is no experimental validation, spatial qualification of this protocol,
or StationEnv connection. Root status remains **58/76, 18 open**.
