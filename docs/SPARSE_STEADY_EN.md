# Sparse steady initialization — v145

[Русский](SPARSE_STEADY_RU.md) · [Transition operator cache](EXCHANGER_CACHE_EN.md)

`SparseFactorCache` reuses a real matrix LU factorization for different right-hand
sides. The optional `SparseSteadyExchanger` applies it to existing exchanger
steady initialization. Physical equations, subsequent integration and the
default implementation remain unchanged.

```python
from core.sparse_linear import SparseFactorCache
from physics.thermal.sparse_steady_exchanger import SparseSteadyExchanger

factors = SparseFactorCache(entries=8, max_bytes=8 * 1024 * 1024)
model = SparseSteadyExchanger(configuration, boundary_controls,
                             {'mode': 'steady'}, factor_cache=factors)
# Reuse factors for another inlet-temperature pair.
row = model.advance(0.2)
info = factors.cache_info()
```

Inlet temperatures change the right-hand side of `A*T=-b`; unchanged flows and
properties preserve the matrix. Changed coefficients require a new exact key.
Given-temperature initialization does not use LU.

## Numerical contract

Inputs become real float64 CSC with summed duplicates, removed explicit zeros
and sorted indices. Keys retain dimensions, indices and exact canonical values,
without tolerance-based quantization. NumPy and SciPy sparse matrices of 1–256
rows and at most 65536 stored input entries are supported, with one finite real
right-hand side. Boolean, complex, malformed and nonfinite inputs are refused.

A **dense numerical rank guard** runs on matrix misses before SuperLU
factorization. Unanchored or numerically singular steady systems remain rejected.
This is deliberately a bounded small-system implementation, not a fully sparse
large-model pipeline.

Every solution, including cache hits, must satisfy the current original matrix:

```text
||A*x-b||inf / (||A||inf*||x||inf + ||b||inf) <= 1e-12
```

Norm products use the extended range of `longdouble`; nonfinite results fail.
The guard must not falsely accept a corrupt solve because its denominator
overflowed. Failed solves or residual checks evict the associated factor.
Tests include deliberately corrupted solutions with coefficients near 1e308.

The exchanger separately requires positive temperatures and a dimensional
steady residual below **1e-6 W per cell**. A normalized linear residual does
not replace physical acceptance.

## Memory and restart

LRU limits are 1–32 entries and 1024–67108864 accounted bytes. A factor larger
than the budget can solve without being retained. Accounting includes CSC/LU
arrays, permutations and byte keys, excluding Python/SuperLU overhead and
transient dense rank-check arrays; it is not a process peak-RAM bound.

Inputs are copied, solutions and statistics are fresh values, and sharing is
explicit between sequential models. There is no global cache or qualified
concurrent access to one instance. Clearing removes factors and counters.

Factors are not serialized. JSON restart builds a fresh cache, recomputes the
initial steady anchor, restores physical inventories and checks ledgers.
Own snapshots and exact suffix replay are tested. Dense/sparse anchors can
differ by roundoff: universal legacy migration, especially at zero time, is
not claimed. One positive-time legacy snapshot example is tested separately.

## Qualification and cost

Counterflow, reversed streams and zero flow with a nonzero ambient thermal
anchor use 1/4/16 cells and three inlet pairs each: 27 steady models per backend,
nine transient pairs and nine exact JSON restarts. Unanchored zero flow is
rejected. Temperature, dimensional residuals, energy integrals, port power,
commands and times are compared. Nine families require nine factorizations and
18 hits. Six negative controls mutate copies.

Timing is separate: three repeats of 32 full model initializations on 8/32 cells,
including construction and first factorization. Dense, retained sparse factors
and clearing before every solve are measured. Sparse handling can cost more
than a dense solve for small systems; no universal speedup is assumed.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component sparse-steady \
  --output artifacts/sparse-steady-local --phase-timeout-s 1200
```

Use a fresh directory. Source tests, scientific cases, sdist → wheel, installation
outside the checkout, installed tests and full replay run offline. Timings are
separate in source/installed `performance.json`; no CI is required.

Accepted delivery: **73 source and 43 installed-wheel tests**, eight phases and whole scientific JSON equality. Maximum temperature difference: `5.7980287e-12 K`; dimensional steady residual: `2.6107455e-09 W`. All nine restarts were exact. Local median dense/reuse time ratios: **0.55 / 1.49** at 8/32 cells; a value below one means slower execution. [Machine receipt including both timing runs](benchmarks/sparse_steady_v145.json).

## Remaining work

This optimizes existing small linear steady families. Sparse nonlinear PWR
Jacobian assembly, refresh criteria, nonlinear factorization reuse and
long-duration plant qualification remain open. No experimental validation or
full-library run is claimed. The broader NPP-15.02 remains open: **58/76, 18 open**.
