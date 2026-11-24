# Reusing exchanger transition operators — v144

Follow-up: [sparse steady initialization and LU reuse v145](SPARSE_STEADY_EN.md).

[Русский](EXCHANGER_CACHE_RU.md)

The optional `CachedTwoStreamExchanger` reuses transition operators for the
existing two-stream exchanger with finite fluid and wall heat capacities.
`TwoStreamExchanger` remains the default. Physical equations, inventories,
conductances, flow directions and independently integrated cell energies are
preserved.

```python
from physics.thermal.cached_exchanger import CachedTwoStreamExchanger

model = CachedTwoStreamExchanger(configuration, boundary_controls,
                                initialization, cache_entries=8)
row = model.advance(0.1)
info = model.cache_info()
model.clear_cache()  # Physical state is unchanged.
```

## Cached operator

For `T'=A*T+b`, the temperature increment and independent energy integrals
are propagated using the current derivative `v=A*T0+b`:

```text
dT' = A*dT + v
I'  = C*A*dT + C*v
v'  = 0
```

Only the response of `dT,I` to `v` is cached. Keys contain exact `dt`, matrix
bytes and capacity bytes, with no rounded operating points or physical states.
Flow or coefficient changes require a different operator. Inlet temperature
changes can reuse an unchanged operator while recomputing the driving vector.
Energy integrals are not replaced with differences of stored states.

Physical temperatures and integrals commit only after native ledger checks.
A failed interval leaves the physical snapshot unchanged; a computed operator
and cache counters may remain. Corrupting the integral block is explicitly
tested to fail the energy check.

## Memory and restart

Each instance owns an LRU cache of 1–32 entries with 1–32 cells. Maximum
retained array and byte-key payload is 7102464 bytes. `cache_info()['bytes']`
excludes Python overhead and temporary exponential workspaces. Operators are
read-only arrays; statistics are returned as a fresh dictionary.

Caches are not serialized or trusted from snapshots. JSON restart starts
with an empty cache and exactly reproduces the suffix. Legacy exchanger
snapshots are accepted. Clearing the cache leaves physical state unchanged.
An instance assumes sequential advancement; independent instances do not
share caches.

## Qualification and local cost

Six paired cached/default trajectories use 1/4/8 cells and twenty steps each.
They cover counterflow, inlet changes, reversal, return to previous flow,
zero flow and two time steps, with six exact restarts. Tests additionally
cover 16/32 cells, an isolated analytic mode, exact keys, eviction, different
capacities and corrupt operators. Six negative controls mutate result copies.

Timing is separate from deterministic scientific JSON: 64 steps of 0.1 s,
three repeats on 4/16 cells. The default solver is compared with retained
cache reuse and clearing before every step. Construction is excluded; first
fill is included. Computing the larger operator without reuse can cost more
than a default step. This short local benchmark provides no universal speed
guarantee.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m scripts.verify_component_delivery --component exchanger-cache \
  --output artifacts/exchanger-cache-local --phase-timeout-s 1200
```

Use a fresh directory. Source tests, sdist → wheel, installation outside the
checkout, installed tests and the whole scientific replay are checked offline.
`performance.json` is separate in source/installed; measured timings need not
match. Local scientific dependencies are reused; no CI is needed.

Accepted delivery: **57 source and 27 installed-wheel tests**, eight phases and whole scientific JSON equality. Maximum temperature difference: `5.6843419e-14 K`; cell-energy integral difference: `3.4924597e-10 J`. Six trajectories yielded 105 hits and 15 fills; six restarts were exact. Local median repeated-step speed ratios versus the default solver: **1.77× / 6.30×** at 4/16 cells. [Machine receipt including both timing runs](benchmarks/exchanger_cache_v144.json).

## Scope

Constant properties, fixed fluid masses, prescribed flows and piecewise
constant controls only. No new correlations, phase changes or hydraulic
solution. This optimization is not connected to PWR/StationEnv. Sparse
Jacobians, factorization reuse and the broader NPP-15.02 remain open. No
experimental validation or full-library test run is claimed. **58/76, 18 open**.
