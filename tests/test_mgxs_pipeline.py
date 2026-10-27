"""Двухгрупповая пластина, таблица сечений и контракт OpenMC."""

import numpy as np
import pytest

from core.mesh import Mesh1D
from core.status import DomainError
from coupling.openmc_adapter import (
    OpenMCAdapter,
    compare_k,
    interpret_mgxs_result,
    slab_specification,
    specification_manifest,
    validate_specification,
)
from data.open_library import open_two_group_provider
from physics.neutronics.diffusion import (
    analytical_two_group_slab_k,
    removal_with_scatter,
    two_group_slab_k,
)


def test_two_group_slab_converges_to_buckling():
    diffusion = np.array([0.02, 0.005])
    sigma_a = np.array([0.01, 0.08])
    scatter = 0.04
    nu_sigma_f = np.array([0.012, 0.10])
    height = 1.2
    expected = analytical_two_group_slab_k(diffusion, sigma_a, scatter, nu_sigma_f, height)
    errors = []
    for cells in (20, 40, 80):
        mesh = Mesh1D.uniform(height, cells)
        k_value, _phi, converged = two_group_slab_k(
            mesh, diffusion, sigma_a, scatter, nu_sigma_f, tol=1e-9
        )
        assert converged
        errors.append(abs(k_value - expected))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 5.0e-3


def test_library_scales_density_and_refuses_burnup():
    provider = open_two_group_provider()
    base = provider.at_state(900.0)
    dense = provider.at_state(900.0, density_ratio=1.25)
    assert dense.sigma_a_per_m == pytest.approx(base.sigma_a_per_m * 1.25)
    assert dense.scatter_to == pytest.approx(base.scatter_to * 1.25)
    assert dense.diffusion_m == pytest.approx(base.diffusion_m / 1.25)
    scatter = base.scatter_to[:, :, None]
    removal = removal_with_scatter(base.sigma_a_per_m[:, None], scatter)
    assert removal[0, 0] == pytest.approx(base.sigma_a_per_m[0] + base.scatter_to[1, 0])
    assert removal[1, 0] == pytest.approx(base.sigma_a_per_m[1])
    with pytest.raises(DomainError):
        provider.at_state(900.0, burnup_mwd_per_kg=10.0)
    with pytest.raises(DomainError):
        provider.at_state(900.0, density_ratio=0.0)


def test_openmc_specification_has_no_invented_k():
    spec = slab_specification()
    manifest = specification_manifest(spec)
    assert manifest["groups"] == 2
    assert specification_manifest(spec)["sha256"] == manifest["sha256"]
    broken = slab_specification()
    broken["energy_bounds_ev"] = [1.0, 2.0, 3.0]
    with pytest.raises(DomainError):
        validate_specification(broken)
    result = OpenMCAdapter().run_mgxs(spec)
    assert result["applicability"] in ("unsupported", "out_of_domain")
    assert "k_combined_mean" not in result
    assert result["manifest"]["sha256"] == manifest["sha256"]
    payload = {
        "k_combined_mean": 1.01,
        "k_combined_std": 0.002,
        "macroscopic_unit": "1/cm",
        "absorption_per_cm": [0.01, 0.2],
        "flux_std": [0.001, 0.002],
        "openmc_version": "test-report",
    }
    interpreted = interpret_mgxs_result(payload)
    assert interpreted["absorption_per_m"][1] == pytest.approx(20.0)
    report = compare_k(1.01, payload)
    assert report["delta_k"] == pytest.approx(0.0)
    assert report["delta_over_std"] == pytest.approx(0.0)
    del payload["k_combined_std"]
    with pytest.raises(DomainError):
        interpret_mgxs_result(payload)
