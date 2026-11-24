# One-minute finite boron exchange — v141

Follow-up: [finite closing tail v142](BORON_CLOSURE_EN.md). The minute package below retains its original scope.

[Русский](BORON_DURATION_RU.md) · [IF97 ports](BORON_PORTS_EN.md)

This package extends a component's numerically checked interval from 2.5 to
**60 seconds**, with continuous makeup requested at 1 kg/s. Finite supply and
receiver inventories evolve, heat accumulates, and the strict IF97 port policy
remains active. Physical equations, coefficients and checkpoint format are
unchanged.

This is a one-minute research case, not hour-long plant qualification,
equipment testing or experimental validation.

## Protocol

Both cases use 5000 kg in each tank, initially 350 K and 1 MPa,
UA = 500000 W/K, strict liquid ports, and 2 neutron / 2 fluid / 4 thermal cells.
Only supply boron differs: 0 or 2000 ppm. Initial primary states match;
feedback references remain frozen.

A 1 kg/s command is issued initially. Observations are at 0, 0.1, 1, 10, 30
and 60 s. The actuator time constant is 0.2 s:

```text
q(t) = 1 * (1 - exp(-t/0.2))
m_transferred(t) = 1 * (t - 0.2*(1 - exp(-t/0.2)))
```

Exactly **59.8 kg** should transfer by 60 s. This protocol supplies its mass
reference explicitly; it does not reuse the old 50 kg/s, 2 s closure reference.
Flow is still enabled at the endpoint. Closing, depletion, power loss and
further continuation are outside this acceptance.

Each case uses `(max_step_s, rtol)` of `(0.2, 2e-8)` and `(0.1, 2e-9)`.
Both fine cases round-trip a checkpoint through JSON at 30 s, restore their
verified base and history, and repeat the last 30 s exactly, including the
final checkpoint.

## Stable temperature reference

Small flow makes UA/capacity large. Subtracting nearly equal temperatures
inside the earlier boundary problem loses precision; mesh refinement alone
cannot resolve that conditioning issue. The same differential equations are
written in normalized hot temperature `theta` and temperature difference `d`:

```text
theta' = -(UA/Ch) * d
d' = UA*(1/Cc - 1/Ch) * d
theta(0) = 1
theta(1) - d(1) = 0
```

The coordinate follows the hot stream; the cold stream runs oppositely.
The effectiveness–NTU formula is not used by this reference. `solve_bvp` has
normalized tolerance `1e-10` and a finite 8192-node limit. Nonconvergence fails
acceptance. Tests cover small flow, unequal and nearly equal capacities,
zero limits, temperature translation and reverse transfer. The earlier
reference and its defaults are retained.

## Acceptance

Checks cover global mass/energy/boron, exact tank usage, pump work and thermal
ports, frozen feedback references, IF97 states and port entropy. Full-state
time-setting discrepancy must be at most `1e-4`. Primary boron changes must
have the correct sign. The two supply cases must produce a power difference
of at least `1e-4` initial power and twenty times temporal noise.

Observed heat and capacity-model outlet temperatures are compared with the
new boundary solution; the relative heat criterion is `1e-8`. These are
numerical thresholds, not physical uncertainty estimates. Seven negative
controls alter copies, including inventories, energy, time and late restart.

Accepted delivery: **94 scoped source tests and 74 installed-wheel tests**, eight successful phases and whole scientific JSON equal. All seven negative controls passed. Minimum observed subcooling was `55.868074 K`, maximum receiver temperature `350.2077156 K`; transferred mass `59.8 kg`. Maximum full-state temporal discrepancy was `5.13155e-05`. An archived strict v140 checkpoint at 2.5 s replayed exactly, with unchanged physical model source identity.

## Local reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-duration \
  --output artifacts/boron-duration-local --phase-timeout-s 1200
```

Use a fresh output directory. The profile runs tests, scientific cases,
sdist → wheel, installation outside the checkout, installed tests and the
complete scientific replay. Scientific dependencies are reused locally;
no CI or network is required. [Receipt v141](benchmarks/boron_duration_v141.json).

## Limits

The result covers this continuous 1 kg/s protocol, conductance, pressures and
meshes. It does not transfer automatically to larger flows, other hardware,
closing flow or other spatial resolutions. The regenerator remains steady
with inlet-frozen capacities; IF97 qualifies ports, not internal profiles
or valve-throat conditions. Phase-event continuation, critical flow,
exchanger storage, chemistry, plant data, hour-long regimes and StationEnv
integration remain open. The full library test profile is not run.
The root plan remains **58/76, with 18 tasks open**.
