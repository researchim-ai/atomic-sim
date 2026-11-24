# Four separate station-agent objectives — v118

Update v120–122: [changing dispatch](STATION_DISPATCH_EN.md) and a fresh repeat of this suite accept NPP-14.03 for research task contracts. The original v118 scope is retained below.

[Русский](STATION_AGENT_OBJECTIVES_RU.md)

`scripts.station_agent_suite` connects existing StationEnv task contracts in one
reproducible run without blending their metrics into a universal reward.
Policies receive copies of observations, public action schemas and declared
objectives; privileged labels and model meters remain evaluator material.

- Power holding fixes the initial publicly measured fission-power target for
  three 0.01 s ticks. Tracking and passive policies retain error integrals,
  time outside the band, missing measurements and trip status. This does not
  qualify tracking a changing electrical dispatch schedule.
- Electrical budgets require at least 20 MJ exported and at most 3 MJ own-needs
  energy over the same short horizon, with power and conservation gates.
  Integrated backend counters avoid endpoint quadrature and double subtraction.
  A DHR-request baseline must retain its added consumption. This is a budget
  contract, not tariff optimization, wear modeling or measured plant efficiency.
- Diagnosis compares redundant pressure channels over 0.04 s with a 150 kPa
  bias from tick 1. False positives, false negatives and delays remain separate.
  The silent baseline must miss the fault; private truth is never sent to it.
- Recovery starts after four 0.05 s feed-pump-off ticks. Candidate and held
  baseline run from the same checkpoint for one second. Success requires
  full horizons, valid measurements, energy gates, an initial excursion, final
  in-band dwell and a relative IAE gain. No-op and invalid-action candidates
  remain negative controls; the original checkpoint must remain exact.

All four objectives retain invalid-action attempts. The suite contains 12
contract attempts; each of the three recovery attempts has two actual arms,
so contract counts are not a count of physical trajectories. Deterministic
matched-model improvement is not real-plant causal or population evidence.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.station_agent_suite \
  --output artifacts/my-station-agent-suite
```

After wheel installation run `python -I -m scripts.station_agent_suite
--output <fresh-directory>` outside the checkout. The recipe pins objectives,
policies and sources. Results retain all metric namespaces, failures, budgets
and forks; recovery outputs are also written individually. Scientific replay
removes only measured `wall_s` fields, keeping every scientific field.

NPP-14.03 remains open for changing-load and more substantial operational tasks.
The [configuration-derived holdout audit](STATION_PHYSICAL_HOLDOUTS_EN.md)
does not close unseen-object or independent operating-history requirements.
See the [v117–119 receipt](benchmarks/station_open_suite_v117_v119.json).
