# HEM network rest preservation with unequal cell volumes

[Русский](HEM_NETWORK_REST_RU.md). A general library check found mass drift in a synthetic quiescent branched network. Despite identical initial pressure and temperature, one cell's mass changed from `1.91445947658913` to `1.9144594765891298` kg after 0.001 s: one ULP. The exact rest-preservation test exposed this difference; its tolerance was not relaxed.

## Cause

Initial inventory is formed as `m₀ = ρ₀ V`. Dividing the rounded inventory by volume need not recover the same binary `ρ₀`. Plenum and channel volumes produced an artificial density difference of approximately `1.14e-13` kg/m³. The Rusanov stabilization term converted it into transport. The reproduced maximum initial mass rate was approximately `6.70e-13` kg/s. EOS pressure and temperature were identical across all cells.

## Change

The integrator already stores deviations from its initial state. The dissipative jump now uses canonical initial intensive states and those integrated deviations:

```text
ΔW_ab = (W₀,b − W₀,a) + (δW_b − δW_a)
δW_i = (δm_i, δP_i, δC_i) / V_i
W₀,i = (ρ₀,i, ρ₀,i v₀,i, ρ₀,i (u₀,i − e_ref + v₀,i²/2))
```

`C` is the existing energy coordinate with a shifted reference, including kinetic energy. Physical flux terms, EOS, admission guards, friction/heat-transfer closures and step bounds are unchanged. Each shared face flux still enters neighboring cells with opposite signs and is retained in the ledger.

No jump tolerance, inventory projection or manual restoration of the initial state is introduced. A separate test applies a nonzero integrated `δm=1e-16` kg and checks the nonzero analytical dissipative transport and total-mass conservation. Branched, loop and parallel topologies with unequal node volumes must preserve mass, energy and zero momentum exactly at rest.

## Impact checks

The change is confined to `atomic_model_hem_network`; mixture and qualified phase-model subclasses are also checked. 297 targeted HEM and delivery tests passed. Existing qualification protocols were repeated without changing criteria: 12 graph cases with three temporal caps and two IVP references, 10 mixture cases using the same pattern and 9 phase-model cases with their additional checks. The reference solvers share IF97 EOS with production and do not provide independent experimental validation.

Earlier receipts remain evidence for their own source versions; the changed graph requires new source-bound evidence. The PWR station backend does not depend on this graph flux, and its accepted nominal-hour recipe still matches current sources. This fix does not extend physical applicability or close additional physical-validation tasks.


[v114 receipt](benchmarks/hem_network_rest_v114.json): all three qualification JSON files were repeated exactly from the wheel, with no excluded fields. The installed repeat passed 90 HEM/delivery tests, excluding one inapplicable checkout import-guard test.
