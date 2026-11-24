# Open station benchmark — v117

[Русский](STATION_OPEN_BENCHMARK_RU.md)

`scripts.station_benchmark` defines a source-bound public contract for the
existing synthetic R0 thermal/shaft subsystem. All parameters, initial states,
boundaries, commands, SI outputs, error scales and work budgets are identical
across methods. This is not a full PWR/WWER comparison or experimental validation.

The three ten-second scenarios are nominal operation, a 1→0.8→1 MW source step
at 2/6 s, and generator disconnection/reconnection at 2/6 s. Commands change
exactly at segment boundaries without resetting states or integrals. The output
grid has 41 samples at 0.25 s spacing. Four temperatures, rotor energy, source
energy, electrical export and cooling rejection are compared at every sample.
Normalized absolute error uses 1 K / 1 MJ scales and must not exceed 1e-4.
Relative integrated energy defects must not exceed 1e-8.

A native port-connected exact linear stepper is compared with separately coded
component power balances integrated by Radau, BDF and DOP853. The latter
implementation does not call the port assembly matrix. Both share physical
parameters and SciPy. The reference is a centered affine matrix exponential
with boundary integrals, not an experimental observation. Algorithms use the
[official SciPy IVP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

Every IVP method uses rtol=1e-8, normalized atol=1e-9 and max_step=0.25 s.
Each attempt has a 20,000 actual RHS-call budget and a 60 s wall budget.
Failures remain in the denominator. Existing sampled R0 physical-domain checks
are applied at output epochs; continuous-time domain validity is not proven.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.station_benchmark \
  --output artifacts/my-station-benchmark
```

After installation, run `python -I -m scripts.station_benchmark --output
<fresh-directory>` outside the checkout. Outputs include the contract,
progressively retained attempts, runtime report, scientific results without
wall-time fields, and a capability/gap matrix in JSON. Timing covers advancement
and its setup, excluding reference generation and result assessment. A shared
host and single observations do not establish universal speed superiority.
RHS counts are not comparable to matrix-exponential work for the exact stepper.

External results use exactly these JSON fields: `format` with value
`station-r0-submission/v1`, `contract_sha256`, `case`, `simulator`, `times_s`,
and `values`. Each value row contains all eight SI quantities in contract order.

```bash
python -I -m scripts.station_benchmark --submission external-result.json \
  --output assessed-result
```

The assessor recomputes metrics, rejects a different contract, missing/reordered
samples, time shifts, NaN and unknown fields. Prefixes are scored without
passing the horizon. Claimed success or execution cost cannot be injected;
external simulator identity and timing are explicitly unverified.

NPP-17.01 is accepted for this public R0 benchmark contract. Full-station and
external-model comparisons, real measurements, agent utility and complete R1
acceptance remain open. No superiority over RELAP, TRACE, Modelica or another
product follows from these subsystem runs. See the
[sealed v117–119 receipt](benchmarks/station_open_suite_v117_v119.json).
