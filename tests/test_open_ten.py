"""Поток-вес, заморозка прямой, факторизация, паросодержание и LMTD."""

import numpy as np
import pytest

from components.heat_exchanger import lmtd_heat_w, log_mean_temperature_difference
from core.status import DomainError
from physics.depletion.chains import stable_daughter_inventory
from physics.hydraulics.two_phase_ext import (
    boiling_onset_length_m,
    equilibrium_quality,
    wall_force_split,
)
from physics.neutronics.diffusion import buckling_k, buckling_k_slope, flux_weighted_sigma
from physics.neutronics.transport import FactorizedDiffusion, _sparse_system, apply_cmfd, cmfd_ratio
from validation.uq import FrozenLinear


def test_flux_weight_buckling_slope_and_factorization():
    sigma = flux_weighted_sigma(np.array([0.2, 0.8]), np.array([1.0, 3.0]), np.array([2.0, 2.0]))
    volume_mean = (0.2 * 2.0 + 0.8 * 2.0) / 4.0
    assert sigma == pytest.approx((0.2 * 2.0 + 0.8 * 6.0) / 8.0)
    assert sigma != pytest.approx(volume_mean)
    flat = flux_weighted_sigma(np.array([0.2, 0.8]), np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert flat == pytest.approx(volume_mean)

    buckling = 4.0
    slope = buckling_k_slope(0.02, 0.3, 0.4, buckling)
    step = 1.0e-4
    numerical = (
        buckling_k(0.02, 0.3, 0.4, buckling + step) - buckling_k(0.02, 0.3, 0.4, buckling - step)
    ) / (2.0 * step)
    assert numerical == pytest.approx(slope, rel=1.0e-6)

    shape = (3, 4)
    diffusion = 0.01 + 0.005 * np.arange(12).reshape(shape)
    operator = FactorizedDiffusion(shape, (1.0, 0.8), diffusion, 0.25)
    first = np.arange(12, dtype=float).reshape(shape) + 1.0
    second = np.ones(shape)
    solved_first = operator.solve(first)
    solved_second = operator.solve(second)
    assert operator.factorizations == 1
    assert operator.solves == 2
    again = operator.solve(first)
    assert again == pytest.approx(solved_first, abs=1.0e-10)
    matrix, _unused = _sparse_system(shape, (1.0, 0.8), diffusion, 0.25, 1.0)
    phi = np.linspace(0.2, 1.4, 12).reshape(shape)
    recovered = operator.solve((matrix @ phi.reshape(-1)).reshape(shape))
    assert recovered == pytest.approx(phi, abs=1.0e-9)
    assert not np.allclose(solved_first, solved_second)


def test_quality_lmtd_cmfd_and_wall_force():
    onset = boiling_onset_length_m(2.0, 4200.0, 500.0, 530.0, 2.0e5)
    assert onset == pytest.approx(2.0 * 4200.0 * 30.0 / 2.0e5)
    assert boiling_onset_length_m(2.0, 4200.0, 530.0, 530.0, 2.0e5) == pytest.approx(0.0)
    quality = equilibrium_quality(1.0e6, 2.0, 2.0e6, inlet_quality=0.1)
    assert quality == pytest.approx(0.1 + 1.0e6 / (2.0 * 2.0e6))
    assert 2.0 * 2.0e6 * (quality - 0.1) == pytest.approx(1.0e6)
    with pytest.raises(DomainError):
        equilibrium_quality(5.0e6, 2.0, 2.0e6, inlet_quality=0.0)

    assert log_mean_temperature_difference(10.0, 5.0) == pytest.approx(5.0 / np.log(2.0))
    assert log_mean_temperature_difference(7.0, 7.0) == pytest.approx(7.0)
    assert lmtd_heat_w(1000.0, 10.0, 5.0) == pytest.approx(1000.0 * 5.0 / np.log(2.0))
    with pytest.raises(DomainError):
        log_mean_temperature_difference(10.0, 0.0)

    coarse = np.array([1.0, 2.0, 4.0])
    reference = np.array([1.5, 1.0, 4.0])
    ratio = cmfd_ratio(coarse, reference)
    assert apply_cmfd(coarse, ratio) == pytest.approx(reference)
    with pytest.raises(DomainError):
        cmfd_ratio(np.array([0.0, 1.0]), np.array([1.0, 1.0]))

    liquid, vapor = wall_force_split(-30.0, 0.25)
    assert liquid + vapor == pytest.approx(-30.0)
    assert liquid == pytest.approx(-7.5)


def test_frozen_line_is_not_refit_and_daughter_conserves_atoms():
    line = FrozenLinear()
    abscissa = np.array([0.0, 1.0, 2.0])
    line.fit(abscissa, 2.0 * abscissa + 1.0)
    holdout_x = np.array([3.0, 4.0])
    assert line.holdout_abs_error(holdout_x, 2.0 * holdout_x + 1.0) == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert line.slope == pytest.approx(2.0)
    with pytest.raises(DomainError):
        line.fit(abscissa, abscissa)

    parent, daughter = stable_daughter_inventory(5.0, 0.2, 4.0)
    assert parent + daughter == pytest.approx(5.0)
    assert parent == pytest.approx(5.0 * np.exp(-0.8))
