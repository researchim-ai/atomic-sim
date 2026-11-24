# Qualified reduced core resolution — v147

[Русский](CORE_FIDELITY_RU.md)

Two neutron resolutions of the coupled research PWR are compared: 2 and 4
diffusion cells, with unchanged thermal/fluid grids, coefficients, requested
initial power and inventory anchors. Four cells are a numerical reference
within the same synthetic model, not measured truth or a continuum solution.

`QualifiedCoreRun` permits reduced resolution only for a verified protocol,
current code identity, reassessed numerical evidence and a favorable local
cost comparison. Both default initial anchors are reconstructed and matched
exactly to evidence before execution. Failed reports or incomplete negative
controls are refused. If the two-cell median is not lower, four cells are selected.
No running state is remapped between grids.

## Domain and budgets

Rod insertion by 0.01 stroke and insertion/return run to 2/3 s. Observations
are at 0, 0.1, 0.5, 1, 2 s, plus 3 s for return. Both grids use
`max_step_s=0.1, rtol=2e-8` and `max_step_s=0.05, rtol=2e-9`: eight trajectories
and four exact JSON restarts from 0.5 s.

Each grid has a separately constructed stationary start at the same requested
power. Computed critical rod trims may differ; the relative perturbation is
identical. Trims and absolute commands are retained. Spatial initial fields
are not asserted to be identical.

Predeclared limits are 0.05 normalized spatial L2 discrepancy for fission and
each of six precursor families; 0.001 discrepancy in total fission/net export
power normalized by initial reference fission power; 5 K for fuel, cladding
and fluid temperatures; 0.001 relative local pressure/face-flow discrepancy;
and 1e-4 temporal full-state discrepancy. Exact overlap norms avoid hiding
local errors by averaging first. L2 budgets are not pointwise error or
physical uncertainty bounds. Energy, mass, boron, local stocks, stationary
rank, frozen feedback reference and rollback are checked too.

## Measured cost and executable boundary

Three complete episodes per scenario/resolution include construction and
stationary initialization. Lower measured median cost is required for two
cells; a smaller state count alone is insufficient. Timings are separate
from deterministic scientific JSON. Timing compares native PWR sessions and
excludes evidence parsing and the wrapper's two-anchor verification. It provides
no universal speed/deadline
guarantee.

```python
from atomic_model_training.spec import read_json
from coupling.core_fidelity import QualifiedCoreRun, protocol

science = read_json('artifacts/core-fidelity-local/source/scientific.json',
                    limit=16 * 1024 * 1024)
cost = read_json('artifacts/core-fidelity-local/source/performance.json')
run = QualifiedCoreRun(science, cost, case='insertion')
position = run.observation()['snapshot']['actuator_status']['rod_position']
for step in protocol('insertion', position):
    observation = run.advance(**step)
```

Criteria are recomputed from retained trajectories before selection. Unknown
scenarios, altered intervals/commands, stale sources or continuation beyond
the horizon are rejected. This is a bounded protocol runner, not unrestricted
agent control on the reduced model. These restrictions apply to QualifiedCoreRun;
the low-level LocalModeratorPWRSession remains a research API without this
qualification. Another history requires a fresh reference
run or new qualification. The wrapper retains its protocol boundary even
when selecting four cells.

Canonical native commands ensure equivalent `1`/`1.0` inputs do not change
restart hashes. Checkpoints bind science, timings, scenario and source. Restore
reassesses eligibility, reconstructs the stationary start and replays the
allowed prefix to exact checkpoint equality, without clipping or projection.

Separate source/wheel `policy_proof.json` files test off-protocol rejection,
exact wrapper restart and refusal after completion. They are separate from
scientific JSON because measured timing signatures can differ. Six controls
mutate power, precursor family six, temperature, energy, restart and rollback
in copied data.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component core-fidelity \
  --output artifacts/core-fidelity-local --phase-timeout-s 1800
```

Use a fresh directory. Tests, numerical cases, cost/dispatch proof, sdist →
wheel, installation outside the checkout and complete replay run offline.
Whole scientific JSON must match; timings need not. No CI is required.

Accepted: **60 source and 47 installed-wheel tests**, eight delivery phases, whole scientific JSON equality, six negative controls and both source/wheel dispatch proofs. Common cases where two cells are measured cheaper: `[]`. Reference/two-cell time ratios were 0.946/0.946 in source and 0.944/0.944 in wheel; below one means the two-cell model was slower. NPP-12.04 accepted: `True` within this bounded research scope. Plan at v147 acceptance: **59/76, 17 open**. [Machine receipt](benchmarks/core_fidelity_v147.json).

## Limits

This does not validate physical accuracy, SCRAM, other amplitudes, arbitrary
agent actions, long horizons, other parameters/geometries or phase changes.
R2 remains separate. Existing PWR physical sources and identities are unchanged.
