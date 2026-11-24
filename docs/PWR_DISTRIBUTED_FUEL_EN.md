# Distributed PWR fuel and decay heat — v125–126

Update v127: [local cladding storage connects to a separate reduced-PWR session](PWR_LOCAL_CLADDING_EN.md). The original v125–126 scope is retained below.

[Русский](PWR_DISTRIBUTED_FUEL_RU.md)

`coupling.pwr_fuel` connects spatial fuel temperatures and local decay-heat
reservoirs to existing PWR inventories, extending the earlier
[mesh-exchange foundation](CONSERVATIVE_CORE_CHANNELS_EN.md).
`DistributedFuelSession` uses `PWRCoreSession`; `StagedDistributedFuelSession`
also retains the inherited primary loop, film closure, finite steam header,
separator and reheat. These are separate opt-in classes. StationEnv and its
configuration have not been changed to select them.

The spatial neutron model still owns six precursor families locally.
A separate mesh owns M fuel temperatures and 3×M stored decay-heat energies.
Legacy slots 7–10 are derived aggregate aliases, excluded from independent
integration. Local capacities and fuel/clad conductances preserve inherited
totals through length weights. Fission heating uses the conservative overlap
operator; temperature feedback uses its transpose. The inherited square-root
Doppler law replaces the bulk contribution locally without adding it twice.
Rod, density and boron contributions retain their previous approximations.

Each heat reservoir obeys `dH[g,j]/dt = f[g]*P[j] - lambda[g]*H[g,j]`.
Thermal power is `Q[j] = (1-sum(f))*P[j] + sum(lambda[g]*H[g,j])`.
Local fuel temperatures exchange heat with the existing common cladding.
Initial H is explicitly distributed from inherited aggregate inventories by
cell length, without claiming a measured spatial irradiation history.
Reservoirs subsequently evolve locally rather than being reset to a new
instantaneous equilibrium. Neutron precursors and heat reservoirs are distinct.

The full stored energy includes H once. The inherited thermal-plant ledger used
Q as its nuclear input; the new full ledger uses fission P, adding P−Q to the
external-energy integral. An independently integrated storage-exchange counter
checks the change in H. Fuel temperature energy already exists through the
bulk alias and is not added again. Absolute neutron-balance defects and a scale
based on absolute operator contributions are retained, avoiding division by a
near-zero stationary net rate. An injected neutron leak must still be detected.

Qualification covers four cases: uniform reduced PWR, local perturbations with
three/five thermal cells, and uniform staged PWR. Each uses four neutron cells
and three 0.05 s intervals: hold, load fraction 0.99, SCRAM request. Two numerical
presets are compared. Eight trajectories, eight restart suffixes and two
uniform legacy comparisons are retained. The maximum normalized preset
difference is approximately 3.16e-6 against a 1e-4 criterion, using
`max(abs(reference),1)` per component in its native SI units. This is not a
probabilistic uncertainty or a general model-accuracy claim.

Heat remains above instantaneous fission power after the short SCRAM segment,
with H decreasing. Full shutdown and long decay-heat removal are not thereby
qualified. JSON restart after the load change is exact; invalid commands and
late rejections do not commit a partial interval. Negative local inventories
and excessive local temperatures cannot hide behind valid aggregates.

```python
from coupling.pwr_fuel import StagedDistributedFuelSession

session = StagedDistributedFuelSession()
result = session.advance(0.05, {"load_fraction": 0.99})
restored = StagedDistributedFuelSession.restore(session.checkpoint())
```

Snapshots/checkpoints are privileged research data, not agent observations.
Checkpoints bind the backend, mesh, sources and history; core/staged formats
cannot be interchanged. Mesh endpoints must match, with 1–16 fuel cells;
staged heated length must also match the neutron/fuel coordinate.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.qualify_pwr_distributed_fuel \
  --output artifacts/my-pwr-distributed-fuel
```

After wheel installation use `python -I -m scripts.qualify_pwr_distributed_fuel
--output <fresh-directory>` outside checkout. Scientific results and checkpoints
are compared in full between source and installed execution. One long library
batch ended with SIGTERM before producing a receipt; its raw attempt was
retained and the batch passed when rerun against unchanged sources.

NPP-12.03 remains open: cladding and core coolant are still common inventories,
without distributed fluid mass/momentum, channel flows or local density/boron
feedback. Nuclear data remain illustrative, StationEnv integration is absent,
and the old backend's accepted hour is not reused as acceptance of this new
variant. Long coastdown, full SCRAM/DHR, channel-grid convergence and experimental
validation remain unaccepted. Existing closure limits have not been relaxed.
The plan remains 58/76, with 18 open tasks. See the
[v125–126 receipt](benchmarks/pwr_distributed_fuel_v125_v126.json).
