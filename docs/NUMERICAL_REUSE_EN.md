# Aggregate acceptance of numerical reuse — v148

[Русский](NUMERICAL_REUSE_RU.md)

This package combines the original NPP-15.02 requirements: correct caches,
sparse solves and factorization reuse, with measured discrepancies and explicit
applicability. It does not promise universal plant speedup or fully sparse
nonlinear Jacobian assembly.

## Included implementations

1. [Transition cache](EXCHANGER_CACHE_EN.md): exact dt/matrix/capacity keys,
   bounded LRU, independent energy integrals, six paired trajectories and
   six exact restarts, including boundary changes.
2. [Sparse steady LU](SPARSE_STEADY_EN.md): exact CSC keys, memory limits,
   rank checks on misses and residual checks for every RHS; 27 steady models
   per backend and nine transient/restart pairs.
3. [Coupled PWR CSC path](PWR_SPARSE_EN.md): every computed Jacobian nonzero
   retained, native Radau/SuperLU. Its full v146 qualification has twelve
   trajectories and six restarts. Exact source fingerprint matching is
   required; changed physics/runtime needs a new full qualification. Two
   additional short matched PWR trajectories and one exact sparse restart
   are executed in the aggregate run.

The first two matrices are rerun in the current source and installed wheel.
Their complete scientific results and the new PWR check enter the aggregate
JSON. Existing receipts are immutable; the prior full PWR source fingerprint
is embedded in the versioned executable protocol.

Tests cover resource limits, exact keys, changed matrices, singular systems,
corrupt integrals/solutions, overflowing residual norms, platforms without
extended longdouble range, restart and refusal to reuse stale qualifications.
Small systems may be slower with caching; these paths remain opt-in.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component numerical-reuse \
  --output artifacts/numerical-reuse-local --phase-timeout-s 1200
```

Use a fresh directory. Tests, aggregate numerical cases, sdist → wheel,
installation outside the checkout, installed tests and whole scientific replay
run offline. Child benchmark clocks remain separate from deterministic JSON.
No CI is required.

A failed historical source fingerprint check means its old matrix cannot be
reused. Fully requalify the changed model/runtime before updating the reference
in a new versioned protocol. Rewriting identities is not checkpoint migration.

Accepted: **75 source and 75 installed-wheel tests**, eight successful phases and whole scientific JSON equality. NPP-15.02 is closed within the stated domains. Plan at v148 acceptance: **60/76, 16 open**. [Machine receipt](benchmarks/numerical_reuse_v148.json).

## Remaining work

Structural Jacobian assembly without all dense probes, cheaper derivatives,
other phase regimes, arbitrary grids and long plant transients remain future
work. They are not claimed complete. The original NPP-15.02 asks for qualified
numerical optimizations, not exhaustion of every possible optimization.
Experimental validation and the complete integral plant matrix remain separate.
