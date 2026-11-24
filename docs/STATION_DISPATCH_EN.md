# Changing electrical dispatch — v120–122

[Русский](STATION_DISPATCH_RU.md)

`atomic_model_station_energy_eval` now exports `DispatchContract`,
`score_dispatch`, `compare_dispatch`, `schedule_policy` and `hold_dispatch_policy`.
Immutable paired electrical targets and generator-load fractions define a
schedule before any policy is created. Policies receive only copies of operator
observations, action schemas and the declared current objective. Current ideal
power/energy meters remain privileged evaluator material.

The transparent feed-forward reference sends the public fraction through the
normal `load_fraction` command. A held baseline sends no new command. Separate
gates require full horizons, connected grid, no trips, valid energy accounting,
endpoint power tracking and agreement of integrated delivered energy with the
piecewise-constant schedule. Signed energy errors and incomplete prefixes are
retained. Endpoint quadrature is not substituted for backend ODE energy counters.

Matched comparisons run both policies from copies of the same native state and
leave the caller checkpoint exact. Failed policy initialization, invalid actions
and accepted prefixes remain in results. A failed, tripped, disconnected or
incomplete baseline cannot generate a gain; neither can a near-zero baseline
error. Comparative success requires candidate task success, a baseline that
misses the power band, different commands and a declared minimum IAE improvement.

The qualified short configuration starts at hot equilibrium, uses six 0.1 s
intervals and load fractions 1 / 0.98 / 0.98 / 1.02 / 1.02 / 1. Targets are fixed
from initial generation and own-needs power, without later hidden-meter updates.
Both power and delivered-energy bands are 0.2%. Reference, held, invalid-candidate,
invalid-baseline and tracking-baseline controls are retained. A finer 0.05 s
schedule preserves switching times; restart occurs after the second interval.
Six comparisons plus two restart comparisons each contain two actual arms.

In this case the reference reduces sampled IAE by 99.6214%; its peak endpoint
error is approximately 0.0158%. Holding misses the band for 0.4 s, despite nearly
matching total scheduled energy. The requested/delivered energies are
522.666551/522.693831 MJ. This demonstrates why both tracking and energy gates
are necessary. Common-grid coarse/fine differences are below 6.85e-16 for power
and 1.15e-16 for export energy. These observations establish neither a general
convergence order nor accuracy over other or longer schedules.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.qualify_station_dispatch \
  --output artifacts/my-station-dispatch
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.station_agent_suite \
  --output artifacts/my-station-objectives
```

After installation use the same modules with `python -I -m ...` outside checkout.
Outputs must be fresh. Each comparison is retained; scientific replay includes
both restart arms and removes only measured `wall_s` fields.

Together with the freshly repeated [four objectives](STATION_AGENT_OBJECTIVES_EN.md),
this accepts NPP-14.03 for research task definitions, metrics and baseline
policies. Industrial utility, day-long dispatch, economic optimality, unseen
physical-object transfer, full R1 and experimental validation remain unaccepted.
See the [sealed v120–122 receipt](benchmarks/station_dispatch_v120_v122.json).
