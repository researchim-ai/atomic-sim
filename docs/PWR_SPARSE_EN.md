# Sparse Radau path for the coupled PWR — v146

[Русский](PWR_SPARSE_RU.md) · [Steady LU cache](SPARSE_STEADY_EN.md)

`SparseLocalModeratorPWRSession` passes a CSC Jacobian to the implicit solver
of the existing monolithic research PWR: neutronics, fuel, cladding, local
coolant inventories and the surrounding loop.

```python
from coupling.pwr_sparse import SparseLocalModeratorPWRSession
from coupling.refinement import coupled_pwr_mesh

model = SparseLocalModeratorPWRSession(**coupled_pwr_mesh(2),
                                      max_step_s=0.05, rtol=2e-9)
position = model.controls['rod_target']
row = model.advance(0.1, {'rod_target': position + 0.01})
restored = SparseLocalModeratorPWRSession.restore(model.checkpoint())
```

## Numerical policy

A narrow `_solver_jacobian` hook returns the original dense array by default.
The optional adapter converts it to CSC, removing only exact zeros. There is
no drop threshold: every computed nonzero is retained, including 1e-300 in a
dedicated test. Finite-difference probes, physical-domain checks, one-sided
fallback, exact known modes, actuator law, tolerances and work limits remain
unchanged. No approximate structural pattern or frozen Jacobian is introduced.

SciPy Radau selects SuperLU and applies its native factor refresh/reuse policy.
The v145 user LU cache is not used here. Matrix dimensions, nonzero counts and
native nlu/njev/RHS statistics are retained. This does not reduce the cost of
finite-difference derivative evaluations or guarantee a speed improvement.

## Qualification

Dense and CSC backends use a common 2/2/4 neutron/fluid/thermal mesh and the
same illustrative physics. Two presets are compared:
`max_step_s=0.05, rtol=2e-9` and `max_step_s=0.025, rtol=2e-10`.
Scenarios are rod insertion to 2 s, insertion/return to 3 s and explicit SCRAM
to 4 s, requiring arrival at the rod endpoint.

Twelve trajectories check matching initial physical states, stationary rank,
energy/mass/boron, local inventories, precursors, frozen feedback reference,
commands and physical events. Maximum full-state backend discrepancy must
remain below 1e-5, and temporal-preset discrepancy below 1e-4. Six JSON restarts
at 0.5 s exactly reproduce their fine-run suffixes. Invalid commands check
whole-interval rollback. Six negative controls mutate result copies.

SCRAM acceptance covers only this four-second actuator/reactor response.
Full plant shutdown, pump coastdown and long-term decay-heat removal remain
unqualified.

## Checkpoint consequences

The new adapter has its own `pwr-sparse-local-moderator/v1` format and numerical
policy source identity. Dense and sparse checkpoints cannot be silently
exchanged. Restore recomputes the stationary reference and replays history;
no saved matrix or factorization is trusted.

Changing the common solver file also changes existing PWR source-identity
chains. Historical source-bound checkpoints are refused by current code;
they require their matching source version or an explicitly qualified migration.
Hashes are not rewritten. A separate regression reruns archived v143 fine
dilution commands through the current dense path. Matching physical values
does not establish checkpoint compatibility or transfer historical hour-long
acceptance. Existing machine receipts remain immutable.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component pwr-sparse \
  --output artifacts/pwr-sparse-local --phase-timeout-s 1200
```

Use a fresh directory. Scoped and regression tests, scientific cases, sdist →
wheel, installation outside the checkout, installed tests and complete scientific
replay run offline. Timings are separate from deterministic JSON, exclude
construction/restart, and use one observation per combination. Snapshot/checkpoint
preparation inside the measured segment is included; these are not isolated
linear-solve timings. They do not
provide a statistically established speed advantage. No CI is needed.

Accepted delivery: **125 prior source scoped/regression tests**, followed by **17 final scoped and 17 installed-wheel tests** on tightened qualification presets, eight phases, whole scientific JSON equality and six detected negative controls. Maximum full-state backend difference: `2.7758986e-06`; temporal-preset difference: `1.1413578e-05`. Observed Jacobian nonzero fraction: `0.0516581–0.0557537`. Five archived dense boron observations match v143 bitwise across ten fields; its old checkpoint is refused by source identity. [Machine receipt](benchmarks/pwr_sparse_v146.json).

## Remaining work

Structural sparse Jacobian assembly without all dense probes, cheaper
derivatives, long-duration operation, other meshes/scenarios, StationEnv routing
and applicable experimental data remain open. Existing research assumptions
remain in effect. Broader NPP-11.03/04, NPP-12.03 and NPP-15.02 are not closed.
Root status remains **58/76, 18 open**.
