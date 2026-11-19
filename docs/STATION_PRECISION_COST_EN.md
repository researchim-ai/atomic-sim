# Station precision and computational cost

[Русский](STATION_PRECISION_COST_RU.md). The local v111 matrix ties numerical discrepancies to actual solver work and the outcome of every attempt. Its scope is short StationEnv transitions; full R1 acceptance and experimental validation remain separate.

## Reproduce locally

From the repository root, with dependencies installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.qualify_station_precision \
  --output artifacts/my-station-precision --repeats 2
```

Use a fresh output directory and two to five repeats. An installed wheel exposes the same module; use its environment's Python without PYTHONPATH. No CI is required.

Eight fixed cases cover hold, load change, feed-pump cycling, condensate-pump cycling, DHR switching, rod motion, SCRAM and a deliberately invalid load command. All use manual controls, seed 0 and a 0.05 s control tick. Operational windows last 0.10–0.15 s. They exercise local transitions, not complete pump coastdown or residual-heat-removal qualification.

Only two numerical controls differ: `fast_candidate` uses `rtol=1e-7`, `max_step_s=0.1`; `detailed` uses `2e-8`, `0.05`; `reference` uses `5e-9`, `0.025`. Physical configuration is identical and environment defaults are unchanged.

## Acceptance

The full physical state is compared initially and after every tick. The maximum component/sample discrepancy is `abs(candidate-reference)/max(abs(reference),1)`, bounded by `1e-3`. This is an explicitly mixed absolute/relative metric in native state units, not an absolute physical error estimate. The tighter numerical reference can also be wrong. Monotonic improvement near roundoff is not required.

Energy, mass and boron residuals must satisfy `1e-6`, `1e-10` and `1e-10` for both trajectories. Trip reasons, modes, targets, termination flags and event order/type must match; event time tolerance is `1e-3` s. Every attempt includes a real checkpoint, JSON serialization and restoration with an identical checkpoint. Instrumented and ordinary scientific payloads must match exactly. Unexpected failures are retained and fail acceptance.

Two ordinary repeats and one diagnostic run per case/preset produce 72 attempts. All nine invalid-command attempts must return `invalid_action`. Their costs remain recorded, while incomplete attempts receive no simulated-time speed or speed ratio. The expected 63 completions and nine input refusals cannot hide accidental failures.

## Measurement contract

Initialization, advancement, checkpoint creation, JSON serialization and restoration with native replay are measured separately. Advancement includes native agent steps plus identical privileged physical-trace capture. These timings describe the evaluation protocol, not an isolated integration kernel.

Speed comparisons use only ordinary repeats: median wall time, wall time per simulated second and detailed-to-fast-candidate time ratio. Preset order reverses on alternate repeats. Exact water and saturation caches are cleared before each execution; no warmup is discarded. Two repeats do not establish reliable confidence intervals. CPU, affinity, OS, Python/NumPy/SciPy versions and thread limits are recorded; shared-machine load is uncontrolled. A fast candidate may be slower for a particular case.

A separate cProfile run records exclusive own time for EOS, Jacobians, linear algebra, nonlinear solving, events, serialization and other functions. Groups classify functions: EOS calls nested inside Jacobian work remain EOS time. Per-function cumulative time is retained but must not be summed. Diagnostic timings do not contribute to speed rankings.

Actual returned Radau iteration counts include rejected collocation attempts. Linear-solve calls and native `nfev/njev/nlu` are separate counters: `nlu` counts factorizations, not solves. SciPy `nfev` excludes some RHS calls from the custom Jacobian; those remain visible in the profile. Internal inverse-EOS iterations are not counted; EOS function cost is profiled. An interrupted collocation marks iteration counts incomplete and prevents acceptance.

The probe temporarily wraps a private installed SciPy function and the station's integrator alias. Its signature/result schema are checked and the Radau file hash is pinned. Original functions are restored on all exits, including exceptions; nested probes are refused. Run the benchmark in an exclusive process without concurrent integrations in that process. A SciPy upgrade requires a new qualification.

## Delivery evidence

The compact receipt is [station_precision_cost_v111.json](benchmarks/station_precision_cost_v111.json). Full local evidence includes `protocol.json`, eight `*-science.json`, eight `*-cost.json` and `report.json`. Scientific JSON and scientific report fields are repeated exactly from the installed wheel. Wall time and profiles remain separate measurements for each execution.

All 105 checks passed in the source run. Maximum state discrepancies from the reference were `8.716609e-6` for the fast candidate and `1.886416e-6` for the detailed preset, both during feed-pump switching. Detailed-to-fast-candidate time ratios ranged from 0.811 to 1.218; speed gains were not universal. These are short-window observations on one shared machine. A new execution must be assessed using its own measurements, failures and machine configuration.

Local verification covers 82 probe, cost-matrix, environment and capability tests. All 17 new tests are repeated from the installed wheel in a temporary directory outside the checkout, using `python -I` without PYTHONPATH. Dependencies are reused via `system_site_packages`; this does not test different SciPy/NumPy versions or the full library suite. Equality covers nine scientific JSON files and the report's `source/checks/passed/comparisons/summary/attempts` fields. Timing, profiles and machine metadata are independent measurements, not equality requirements.

NPP-15.01 and NPP-15.04 are accepted for this declared short matrix. Sparse Jacobians/factorization reuse under NPP-15.02, long transients under NPP-11 and industrial qualification remain open. New cases or backend changes require requalification.
