# Conservative neutron/thermal-cell exchange — v123–124

Update v125–126: [local fuel and decay heat now connect to opt-in core/staged PWR sessions](PWR_DISTRIBUTED_FUEL_EN.md). The original v123–124 standalone scope is retained below.

[Русский](CONSERVATIVE_CORE_CHANNELS_RU.md)

`coupling.ConservativeMap` supplies one immutable overlap operator to the
existing steady multiphysics solver and a new transient test problem. The old
`conserve_integral` now rejects NaN, invalid widths/shapes, unrepresentable
integrals and mismatched physical lengths instead of silently returning a
corrupt field. Only summation-roundoff endpoint reconciliation remains in the
width-based legacy adapter; interior boundaries and state values are not scaled.

For cell-integrated power, `P[j,i] = overlap[j,i]/source_width[i]`, followed by
`power_th = P @ power_n`. Density/average transfer instead divides overlaps by
the target width. The feedback temperature uses `P.T`, preserving the algebraic
power/temperature pairing across meshes. That identity is not a new physical
energy law. The geometry is one-dimensional; this is not 3D volume intersection
or a variable-cross-section model. Mesh arrays are owned, immutable and safe to
share across deep copies. The supported mesh budget is 1–512 cells per side,
with exactly matching endpoints and adequately resolved float64 intervals.

Existing steady wrappers retain their `ValueError` interface, use the common
operator, and build geometry once per fixed-point solve. Their provenance now
pins the shared operator SHA, preventing unnoticed reuse of old checkpoints
when the exchange implementation changes. Prior sealed evidence is retained.

`coupling.core_channels` adds `ChannelConfig`, `CoreChannels` and `ChannelSession`.
A synthetic one-group reflecting `DiffusionCore` owns six neutron-precursor
families locally in each of 2–8 neutron cells. Another mesh owns lumped thermal
capacities with linear conductance to a fixed sink. Mapped fission heating and
returned temperature enter a monolithic system with explicitly synthetic linear
temperature reactivity feedback. No evaluated plant cross sections are inferred.

Native integration uses backward Euler with analytic Newton matrices and
admissible damping. Populations are not clipped or renormalized. Energy,
neutron/precursor inventory and transfer identities are checked. A late failure
rolls back the entire requested interval. JSON checkpoints bind sources,
configuration and history, replay exactly and reject resealed state corruption.

**All fission heat is prompt in this test problem.** Neutron precursors are not
a decay-heat model. There are no fluid mass/momentum inventories, advection,
EOS/phase changes, rod banks or StationEnv integration. This problem is unsuitable
for qualifying SCRAM or long decay-heat removal.

The numerical matrix has three configurations: uniform forcing without thermal
feedback, local forcing with feedback, and local forcing on a five-cell thermal
mesh instead of three. Each has four neutron cells and a 0.2 s horizon.
Steps of 0.02/0.01/0.005 s are compared against Radau and BDF on the same RHS;
the uniform case also matches the independent point-kinetics matrix exponential.
Nine native trajectories, six IVP reference trajectories and three restart pairs
are retained. The minimum observed terminal-state order is 0.9884; finest
normalized error is below 1.91e-5. Temperature scale is 1 K and heat-integral
scale is nominal power times 1 s. These finite checks establish neither global
trajectory bounds nor experimental validation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.qualify_core_channels \
  --output artifacts/my-core-channels
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m pytest -q tests/test_conservative_exchange.py \
  tests/test_core_channel_transient.py tests/test_multiphysics.py
```

After wheel installation run `python -I -m scripts.qualify_core_channels
--output <fresh-directory>` outside checkout. Also rerun the existing
`python -I -m atomic_model_multiphysics --output <fresh-file>` qualification.
Installed reports are compared in full, including checkpoints and source hashes.
The [v123–124 receipt](benchmarks/core_channel_exchange_v123_v124.json) records
acceptance. An earlier test epoch was rejected after adding the provenance
dependency; its partial executions are not mixed into the final profile. Two final-epoch
processes ended with SIGTERM before producing receipts; both raw attempts were
retained and rerun against unchanged sources.

NPP-12.03 remains open: this is a conservative exchange foundation and a separate
transient problem, awaiting integration with PWR inventory owners, complete
heat-source accounting, applicable feedback data and coupled-transient
qualification. The plan remains 58/76 with 18 open tasks; complete R1/R2 is not
claimed.
