# Local moderator response in the PWR channel — v131

v146 update: [common PWR solver and checkpoint identities](PWR_SPARSE_EN.md#checkpoint-consequences). The protocol below retains its historical scope; source-bound checkpoints are not automatically migrated.

Follow-up: [seconds-scale transients v132](MODERATOR_TRANSIENTS_EN.md). The short receipt below describes v131. Shared event-registration source changed in v132, so the earlier unchanged station-signature statement does not apply to v132.

[Русский](PWR_MODERATOR_RU.md)

`coupling.pwr_moderator.LocalModeratorPWRSession` adds local density and boron
response to the [stationary PWR channel](PWR_STATIONARY_EN.md). Actual local
fluid inventories now affect the spatial neutron derivative, in addition to
the existing local fuel Doppler response. This separate opt-in class does not
switch StationEnv or the preceding PWR session defaults.
Actuator and chemical-source APIs are inherited unchanged; this block supplies
feedback from the transported moderator fields.

The model uses inherited **illustrative incremental reactivity coefficients**.
It does not introduce evaluated state-dependent cross sections, measured
moderator coefficients or a reconstructed plant. Absolute local nuclear data
cannot be inferred from this incremental model.

## Mapping the actual inventories

The neutron and fluid meshes may differ. For fluid cell j, density is `M[j]/V[j]`
and boron density is `B[j]/V[j]`. If `P[j,i]` is overlap length divided by neutron
cell i's width, the projected quantities are

```text
water_density[i] = sum(P[j,i] * M[j]/V[j])
boron_density[i] = sum(P[j,i] * B[j]/V[j])
boron_ppm[i] = 1e6 * boron_density[i] / water_density[i]
```

Thus density is a volume average; ppm is water-mass weighted. A plain average
of ppm would be wrong when densities differ. With the declared constant flow
area, integrating the projected water and boron densities over neutron cells
reproduces the fluid inventories. Both defects are audited. These calculated
repartitions are not extra mass or boron state owners.

Zero boron is supported without a positive floor. Invalid inventories,
nonfinite values, mismatched shapes and nonuniform flow area are rejected.
The standalone mapping tests do not qualify every possible composition as an
operating point of the full PWR.

## A fixed hot reference

The session first constructs the existing stationary hot baseline, then freezes
its projected local densities/ppm, cold-leg density and bulk-core ppm before
any commands. The local reactivity law is

```text
R[i] = R_previous[i]
     + k_density * ((density[i]-density_ref[i]) - (cold_density-cold_ref))
     + k_boron   * ((ppm[i]-ppm_ref[i]) - (bulk_ppm-bulk_ref))
```

`R_previous` contains the inherited local Doppler, rod, bias and bulk moderator
terms. Subtracting the bulk **changes** replaces their time-varying response
with the local changes, retaining the baseline offset. Moderator feedback is
not added twice. At the reference state the correction is exactly zero.
Density and boron references do not follow the transient or reset after a
command, trip or restart.

This reference is calculated from the reduced model, not measured or fitted
to experimental data. Anchoring preserves its already constructed equilibrium;
it does not establish that its absolute spatial reactivity field represents a
real reactor. The existing water EOS still treats boron as a dilute passive
tracer. Large composition changes and a physical applicability range for these
constant coefficients remain unqualified.

The coefficient and reference entries in `parameter_registry()` carry family
`illustrative`, explicit units and `uncertainty=None`. The source fingerprint
binds the mapping, inherited coefficients, baseline initializer and new adapter.

## Numerical and restart checks

The neutron reactivity calculation now has an overridable hook. Its default
retains the previous law. The local model supplies the new reactivity through
that same owner, so legacy neutron aliases and the neutron/precursor balance
audit use the actual local law. There is no second derivative update or
duplicate nuclear/thermal state. Fluid transport, thermal contact, kinetics,
heat reservoirs and energy ledgers remain coupled in one ODE system.

The entire physical RHS is compared before and after enabling feedback at the
reference state. In addition to the baseline initializer's rank, a separate
finite-difference stationary Jacobian is audited with the new local law and
references held fixed. It must have full column rank and satisfy the existing
constraint tolerances. `stationary_initialization` describes the baseline;
`local_moderator_stationarity` describes this new fixed-reference audit.

Tests independently check the mapped inventories, mass-weighted ppm, local
boron redistribution at unchanged bulk concentration, density response to a
manufactured local warming, and removal of a cold-leg-only density change from
neutron feedback. These manufactured perturbations test instantaneous response;
they are not mass-conserving operational commands. Actual trajectories retain
the existing global mass/energy/boron and neutron/precursor budget checks.

```python
from coupling.pwr_moderator import LocalModeratorPWRSession

session = LocalModeratorPWRSession()
result = session.advance(0.02, {"scram": True})
local_fields = result["local_moderator"]
restored = LocalModeratorPWRSession.restore(session.checkpoint())
```

JSON restore recomputes the stationary reference from the same sources, replays
commands and requires exact final equality. Saved projected references are
not accepted as arbitrary calibration inputs: resealed reference corruption
is refused. Snapshots/checkpoints are privileged research data, not agent
observations.

Three cases cover one/three fluid cells at full requested power and three cells
at 95%, with two neutron cells. Six trajectories use two numerical presets:
0.1 s hold, 0.02 s load request and 0.02 s SCRAM request. Three fine-preset
restart suffixes are checked; coarse restart is not claimed. Each case must
show nonzero moderator correction, fixed references, conservative mapped
inventories, numerical refinement and valid budgets. Unit tests also cover the
default four-neutron-cell configuration and corruption/boundary cases.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  .venv-layout/bin/python -m scripts.verify_component_delivery \
  --component pwr-moderator --output artifacts/my-pwr-moderator-delivery \
  --phase-timeout-s 1200
```

Accepted: **209 scoped source tests and 127 installed-wheel tests**, eight delivery phases. Maximum preset difference `1.6035354e-06`, hold drift `2.6035233e-10` under `abs(a-b)/max(abs(b),1)` in native SI units. Maximum inventory-map relative defect `1.4832767e-16`. These are numerical metrics, not physical uncertainty. Entire scientific JSON matches.

The offline delivery builds a clean sdist/wheel, verifies exact package contents,
installs outside checkout and compares the complete scientific JSON. Scientific
dependencies are reused. The full library test profile is not rerun by this
component command. Existing closure/domain limits are retained.

The largest moderator reactivity correction in these 0.14 s native runs is
approximately `1.30e-11`. Their moderator excursion is therefore very small;
they do not qualify a pronounced density or boron transient. Response sign
and magnitude are checked with the explicitly manufactured fields in unit
tests. Longer, appreciably perturbed native trajectories remain necessary.

This accepts synthetic local response coupling in the documented short cases.
NPP-12.02 still needs traceable physical state-dependent nuclear data;
NPP-12.03 still needs qualified closures, spatial convergence and long-duration
qualification. Real-installation validation, full shutdown/coastdown/DHR and
StationEnv integration remain open. The plan stays 58/76 with 18 open tasks.
See the [v131 receipt](benchmarks/pwr_moderator_v131.json).
