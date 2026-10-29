"""Разреженная 2D/3D диффузия против плотной сборки и аналитического баклинга."""

import numpy as np
import pytest

from physics.neutronics.transport import analytical_rectangle_k, diffusion_k_nd, diffusion_k_sparse


def test_sparse_homogeneous_matches_dense_and_improves_on_three_grids():
    diffusion = 0.012
    sigma = 0.2
    nu_sigma = 0.25
    lengths = (1.2, 1.2)
    dense, _phi, dense_ok = diffusion_k_nd((8, 8), lengths, diffusion, sigma, nu_sigma)
    sparse, _flux, sparse_ok = diffusion_k_sparse((8, 8), lengths, diffusion, sigma, nu_sigma)
    assert dense_ok and sparse_ok
    assert sparse == pytest.approx(dense, rel=0.0, abs=1.0e-12)
    expected = analytical_rectangle_k(diffusion, sigma, nu_sigma, lengths)
    errors = []
    for cells in (8, 16, 32):
        k_value, _flux, ok = diffusion_k_sparse((cells, cells), lengths, diffusion, sigma, nu_sigma)
        assert ok
        errors.append(abs(k_value - expected))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    cube = (1.0, 1.0, 1.0)
    dense3, _phi3, ok3 = diffusion_k_nd((4, 4, 4), cube, diffusion, sigma, nu_sigma)
    sparse3, _flux3, ok_sparse3 = diffusion_k_sparse((4, 4, 4), cube, diffusion, sigma, nu_sigma)
    assert ok3 and ok_sparse3
    assert sparse3 == pytest.approx(dense3, rel=0.0, abs=1.0e-12)


def test_extra_absorber_lowers_k_and_cellwise_d_matches_scalar():
    diffusion = 0.012
    sigma = np.full((8, 8), 0.2)
    nu_sigma = 0.25
    lengths = (1.2, 1.2)
    base, _flux, ok = diffusion_k_sparse(sigma.shape, lengths, diffusion, sigma, nu_sigma)
    sigma[:4, :] = 0.35
    shielded, _flux2, ok2 = diffusion_k_sparse(sigma.shape, lengths, diffusion, sigma, nu_sigma)
    assert ok and ok2
    assert shielded < base
    varying = np.full((6, 6), diffusion)
    k_array, _flux3, ok_array = diffusion_k_sparse((6, 6), lengths, varying, 0.2, nu_sigma)
    k_scalar, _flux4, ok_scalar = diffusion_k_sparse((6, 6), lengths, diffusion, 0.2, nu_sigma)
    assert ok_array and ok_scalar
    assert k_array == pytest.approx(k_scalar, rel=0.0, abs=1.0e-12)
