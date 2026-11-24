# Finite dosing through flow commands — v143

v146 update: [common PWR solver and checkpoint identities](PWR_SPARSE_EN.md#checkpoint-consequences). The protocol below retains its historical scope; source-bound checkpoints are not automatically migrated.

[Русский](BORON_DOSING_RU.md) · [Closing tail](BORON_CLOSURE_EN.md)

The new API requests a finite mass of **makeup water** from the existing
supply tank. Tank concentration determines its boron content; the request
is not a mass of pure boron. Planning includes current flow, actuator lag
and a finite closing tail. No stock, concentration or energy is assigned
directly.

```python
from coupling.boron_dosing import plan_boron_dose, execute_boron_dose

# session is an existing BoronExchangePWRSession.
plan = plan_boron_dose(session, mass_kg=0.3, flow_kg_s=1.0)
# Inspect plan.steps() before execution if desired.
result = execute_boron_dose(session, plan)
```

## Planning and atomic execution

Planning does not advance time. Execution copies the session, applies normal
`boron_flow_target_kg_s` commands, calls the physical integrator, and verifies
both tank mass changes and residual flow. The original session is committed
only after the entire sequence succeeds. A late failure, including an
unsupported phase regime, leaves its checkpoint unchanged.

Requests range from 0.001 to 100000 kg, further limited by available stock,
receiver pressure, physical time and replay history. Positive flow is bounded
by model configuration. Closing lasts 5–20 actuator time constants, default
five. Long phases are split into native intervals of at most 60 seconds.

For initial flow `q0`, target `q`, time constant `tau`, open duration `T` and
finite closing duration `S`:

```text
q_close = q + (q0-q)*exp(-T/tau)
m_finite = q*T + (q0-q)*tau*(1-exp(-T/tau))
           + q_close*tau*(1-exp(-S/tau))
q_end = q_close*exp(-S/tau)
m_future = tau*q_end
```

A scalar brentq solve finds `T`. Supply reserve is assessed against
**m_finite + m_future**, retaining the entire eventual tail. An already
flowing actuator has an unavoidable minimum transfer; smaller requests are
refused. A closing-only plan is supported when the request equals the finite
tail integral of current flow.

The receiver forecast checks gas volume and its 2 MPa pressure limit after
the entire eventual tail. It predicts inventories and gas pressure, not
thermal admissibility. Temperature, IF97 ports, pump head and all physical
domains remain native solver checks during execution.

Each plan binds the full origin checkpoint SHA256 and planner file SHA256.
A changed session or altered computed field requires replanning. Plan values
and step generation are bounded. Results retain explicit native commands,
snapshots and actual transferred mass.

## Qualification

Two successive 0.3 and 0.2 kg water doses at a 1 kg/s target preserve residual
flow between requests. Supply concentrations of 0/2000 ppm are compared using
`max_step_s=0.05, rtol=2e-8` and `max_step_s=0.025, rtol=2e-9`. Initial tanks
contain 5000 kg each at 350 K and 1 MPa, UA is 500000 W/K, liquid IF97 checks
are enabled, and the joint mesh is 2/2/4.

Independent quadrature integrates the exact actuator rate without calling
the planner mass formula. It verifies both phases and accumulated transfer.
Checks cover mass/energy/boron ledgers, independent thermal reference, EOS,
entropy and a boron effect resolved above time-discretization noise. Two JSON
restarts occur inside the second dose through native history. Atomic execution
itself is not pausable; manually executing `plan.steps()` only provides native
per-step atomicity.

Tests cover impossible requests, future-tail reserve, a target below current
flow, closing-only plans, stale/altered plans, budgets, late failure,
corrupted final inventory and a genuine unsupported IF97 regime. Six negative
controls mutate copies of qualification results.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-dosing \
  --output artifacts/boron-dosing-local --phase-timeout-s 1200
```

Use a fresh directory. Scoped tests, scientific qualification, sdist → wheel,
installation outside the checkout, installed tests and the complete scientific
JSON are checked offline using local scientific dependencies. No CI is needed.

Accepted delivery: **134 source and 114 installed-wheel tests**, eight successful phases, whole scientific JSON equal, all six negative controls detected. Maximum full-state temporal discrepancy: `2.8274612e-06`; minimum observed subcooling: `56.003726 K`. An archived v142 checkpoint at 61 s restores exactly and continues with a 0.3 kg dose and verified ledgers. [Machine receipt](benchmarks/boron_dosing_v143.json).

## Scope

This is a dosing command for an existing research component, not plant CVCS,
solubility modeling, a safety guarantee or experimental validation. There is
no StationEnv connection. Internal thermal profiles, dynamic exchanger
inventories, phase-event continuation and long-duration plant qualification
remain open. Existing physical equations and checkpoint format are unchanged.
Root status remains **58/76, 18 open**.
