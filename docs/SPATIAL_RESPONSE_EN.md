# Forward/adjoint core response and sensitivities — v109

[Русский](SPATIAL_RESPONSE_RU.md)

`atomic_model_spatial_response` adds a static discrete adjoint and local k sensitivities to the existing 1D multigroup finite-volume model. It preserves the original operator assembly and station integrator.

The equations are `A φ = F φ / k` and `Aᵀ ψ = Fᵀ ψ / k`, with `ψᵀ F φ = 1`. Cell widths are already present in the assembled operators and must not be counted again in the inner product. Differentiating this generalized eigenproblem gives:

`dk = ψᵀ (k dF − k² dA) φ / (ψᵀ F φ)`.

For multigroup adjoints and small-perturbation theory, see Bell and Glasstone, *Nuclear Reactor Theory*, [sections 6.2–6.3](https://digital.library.unt.edu/ark:/67531/metadc870144/m1/291/ocr/). The implemented local formulas differentiate this repository's specific FV matrices.

Gradients cover neutron-birth rate, absorption, off-diagonal group transfer and diffusion coefficients. A scattering change includes both removal from its origin group and addition to its destination. The convention is `[to, from, cell]`; diagonal placeholders are not model parameters. Diffusion derivatives include both sides of harmonic interior-face conductance and physical zero-flux boundary leakage. Mesh, boundary type, spectra, precursor fractions/decay and neutron speeds remain fixed.

## Normalization and station scope

Optional `power_weights_w` and `power_w` define an explicit watt-valued projection. The forward/adjoint vectors are scaled inversely, preserving pairing and k gradients. Energy per fission or neutron yield is not inferred from `nu_fission_per_m`.

`station_core_response` uses the existing `DiffusionCore` power projection and leaves the station checkpoint unchanged. It is a static reference at zero bulk reactivity and zero axial bias. Nonzero configured axial bias is rejected. It does not diagnose the live core state or add local thermal feedback/channel coupling.

```python
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab
from atomic_model_spatial_response import sensitivities

result = sensitivities(Model(slab(7, heterogeneous=True)), tolerance=1e-12)
```

The coupled neutron operator must be irreducible and the fundamental forward/adjoint pair positive. Disconnected modes, exhausted iteration budgets and invalid normalization are refused rather than assigned an arbitrary derivative.

## Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. .venv-layout/bin/python -m scripts.qualify_spatial_response --output artifacts/my-spatial-response
```

Four frozen synthetic cases cover homogeneous, nonuniform heterogeneous, reflecting and single-cell multigroup configurations. Checks use independent dense assembly, local central differences, production/loss scaling identities, generalized left eigenvectors and second-order Taylor remainders. These are numerical checks of synthetic coefficient models, not experimental validation of a real core or the complete R2 configuration.

## Acceptance v109

Four frozen cases and the existing station static adapter passed. All 128 local derivatives passed independent checks; the minimum observed Taylor-remainder order was 1.99804. Pair normalization, power projection and scaling identities passed, with the station checkpoint unchanged.

The shared targeted suite passed 153 tests. An installed wheel repeated 44 new tests, all six response JSON documents and the candidate-registry JSON exactly, without excluding fields. The full library suite was not run. NPP-12.01 is accepted for the declared static discrete synthetic-data scope. Real nuclear data, local feedback, bank mapping, channel dynamics and full R2 remain separate. [Receipt](benchmarks/spatial_response_v109.json).
