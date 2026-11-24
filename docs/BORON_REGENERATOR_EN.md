# Regenerative heat recovery in finite boron exchange — v139

Optional extension: [IF97 liquid-port checking v140](BORON_PORTS_EN.md). Earlier results do not qualify outlet phase states; strict mode rejects unsupported two-phase ports with full rollback.

[Русский](BORON_REGENERATOR_RU.md) · [Finite boron exchange](PWR_BORON_EN.md)

Hot primary letdown can return part of its heat to cold makeup. This package
adds an **optional, steady counterflow heat exchanger** to the research boron
exchange component. It reduces the modeled cooling imposed by makeup and
the heat entering the receiving water. It does not reconstruct plant CVCS
hardware.

## Model

`BoronExchangeConfig.regenerator_ua_w_k` specifies total conductance UA,
from 0 to 10⁷ W/K. The default **0** disables heat recovery. There are no new
dynamic states: the exchanger has no fluid inventory or wall heat capacity.
Whether this steady approximation is appropriate for a physical transient
has not been established by measurements.

The two streams have equal mass flow. Their heat capacities may differ:
the hot inlet uses its current IF97 value, held fixed within each steady
exchanger calculation; the cold inlet uses 4180 J/(kg K), consistent with
the existing linear tank closure. Inlet values are reevaluated as the plant
state changes.

The effectiveness–NTU relationship follows the
[EnergyPlus Engineering Reference, Heat Exchangers](https://energyplus.readthedocs.io/en/stable/guides/engineering-reference/16.8-heat-exchangers.html).
For r = Cmin/Cmax and NTU = UA/Cmin:

```text
epsilon = (1 - exp(-NTU*(1-r))) / (1 - r*exp(-NTU*(1-r)))
Q = epsilon * Cmin * (T_hot_in - T_cold_in)
```

Equal capacity rates use the direct limit `epsilon = UA / (UA + Cmin)`;
near equality uses `expm1` to avoid cancellation. Zero flow or UA gives zero
heat transfer. Reversed inlet temperatures reverse heat transfer. Reported
outlet temperatures and entropy belong to this constant-capacity model;
**they are not IF97 outlet flash calculations**.

## Pump work and energy ownership

Makeup passes through the pump before the regenerator. Specific pump work is
`w = head / (rho_ref * eta)`. In the reference-density liquid closure,
ideal pressure work raises p/rho in enthalpy; only the excess raises sensible
temperature, by `(w - head/rho_ref) / 4180`. Electrical work still enters
plant own needs and its energy counter exactly once.

```text
primary_rate = q*(h_supply - h_primary_donor) + pump_electric + Q
receiver_water_rate = q*h_primary_donor - Q - p_receiver*q/rho_ref
supply_water_rate = -q*h_supply + p_supply*q/rho_ref
```

Gas-cushion work remains separate. Q is an internal transfer, not external
heat or electrical generation. Boron transport, equal makeup/letdown flows
and finite tank inventories remain in place. UA is illustrative, without
measurements, calibration or a physical uncertainty estimate.

## Qualification

Eight native trajectories use 2 neutron / 2 fluid / 4 thermal cells:
0/2000 ppm supply × UA 0/200000 W/K × two time settings. Observations are
0.1, 0.5, 1, 2 and 2.5 s. A 50 kg/s target is requested initially; closure
starts at 2 s and retains the exponential flow tail. Both fine recovery
cases restart from 0.5 s with exact continuation.

At each observation, a separate differential boundary problem solves the
two counterflow temperatures with `solve_bvp`, without the effectiveness
formula. Unit tests cover equal, unequal and nearly equal capacities,
zero limits and reverse transfer. Acceptance also requires global budgets,
individual energy ports, tank depletion, boron, a frozen initial reference,
time-setting refinement, pump accounting and capacity-model entropy.
Seven negative controls alter copies only; they are not physical experiments.

Accepted delivery: **75 scoped source tests and 66 installed-wheel tests**, eight successful phases and whole scientific JSON equal. All seven negative controls passed. Peak recovered heat was `22.792407 MW`; final receiver cooling was `2.2606245 K` / `2.2606193 K` for dilution/boration. Maximum full-state temporal discrepancy was `5.55075e-06`. Both bypass physical trajectories matched v138 bitwise at six observations each.

## Local reproduction

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component boron-regenerator \
  --output artifacts/boron-regenerator-local --phase-timeout-s 1200
```

Use a fresh output directory. The profile checks source tests, sdist → wheel,
installation outside the checkout, installed tests and the entire scientific
replay. Whole scientific JSON must match. Scientific dependencies are reused
from the local environment; no CI or network is required for this run.
[Acceptance receipt v139](benchmarks/boron_regenerator_v139.json).

## Limits

This package does not qualify long runs or spatial convergence of the active
regenerator. The v138 matrix belongs to the preceding source and disabled
heat recovery. Updated adapter signatures do not rebind earlier checkpoints.
The default StationEnv does not use this component.

Remaining work includes fluid/wall storage, pressure loss and sizing,
variable heat capacity along the exchanger, outlet/letdown EOS and phase
checks, auxiliary cooling, chemistry and plant CVCS data. Capacity-model
entropy is not the complete IF97 entropy balance. The root plan remains
**58/76, with 18 tasks open**.
