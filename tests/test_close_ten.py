"""Трёхмерное отражение, подобие насоса, объём твэла и выход канала."""

import math

import numpy as np
import pytest

from components.equipment import affinity_ratios, orifice_mass_flow, parallel_orifice_flow
from core.status import DomainError
from physics.depletion.chains import DecayHeat
from physics.hydraulics.two_phase_ext import channel_exit_quality, equilibrium_quality
from physics.materials.iapws import (
    homogeneous_momentum_flux,
    mixture_specific_volume,
    separated_momentum_flux,
    void_fraction,
)
from physics.neutronics.transport import analytical_reflective_k, diffusion_k_sparse
from physics.thermal.fuel_rod import FuelRod
from plants.reactor_types import SFRLoop


def test_reflective_cube_improves_and_affinity_orifices_add():
    diffusion = 0.012
    absorption = 0.2
    fission = 0.28
    lengths = (1.0, 0.9, 0.7)
    reflected = (True, False, False)
    expected = analytical_reflective_k(diffusion, absorption, fission, lengths, reflected)
    coarse, _phi, ok = diffusion_k_sparse(
        (4, 4, 4), lengths, diffusion, absorption, fission, tol=1.0e-8, lower_reflective=reflected
    )
    fine, _phi2, ok2 = diffusion_k_sparse(
        (6, 6, 6), lengths, diffusion, absorption, fission, tol=1.0e-8, lower_reflective=reflected
    )
    assert ok and ok2
    assert abs(fine - expected) < abs(coarse - expected)

    assert affinity_ratios(2.0) == pytest.approx((2.0, 4.0, 8.0))
    assert affinity_ratios(0.0) == pytest.approx((0.0, 0.0, 0.0))
    one = orifice_mass_flow(0.62, 0.01, 800.0, 2.0e5)
    both = parallel_orifice_flow([(0.62, 0.01), (0.62, 0.01)], 800.0, 2.0e5)
    assert both == pytest.approx(2.0 * one)


def test_mixture_momentum_channel_exit_and_linear_decay_heat():
    quality, rho_l, rho_v = 0.3, 800.0, 20.0
    specific = mixture_specific_volume(quality, rho_l, rho_v)
    void = void_fraction(quality, rho_l, rho_v)
    assert void == pytest.approx((quality / rho_v) / specific)
    flux = 1200.0
    assert separated_momentum_flux(flux, quality, void, rho_l, rho_v) == pytest.approx(
        homogeneous_momentum_flux(flux, quality, rho_l, rho_v)
    )

    onset_heat = 2.0 * 4200.0 * 30.0
    onset = onset_heat / 2.0e5
    assert channel_exit_quality(onset, 2.0, 4200.0, 500.0, 530.0, 2.0e5, 1.5e6) == pytest.approx(0.0)
    past = onset + 4.0
    produced = channel_exit_quality(past, 2.0, 4200.0, 500.0, 530.0, 2.0e5, 1.5e6)
    assert produced == pytest.approx(equilibrium_quality(2.0e5 * 4.0, 2.0, 1.5e6, 0.0))

    heat = DecayHeat.illustrative()
    stored = np.array([1.0e6, 2.0e6, 4.0e6])
    assert heat.advance_linear_fission(stored, 2.0e6, 5.0e4, 0.0) == pytest.approx(stored)
    step = 1.0e-6
    rate = (heat.advance_linear_fission(stored, 2.0e6, 5.0e4, step) - stored) / step
    assert rate == pytest.approx(heat.fractions * 2.0e6 - heat.decay_constants * stored, rel=1.0e-4)
    with pytest.raises(DomainError):
        heat.advance_linear_fission(stored, 1.0, -1.0, 2.0)


def test_pin_volumes_match_cylinders_and_sodium_step_matches_its_derivative():
    rod = FuelRod(
        r_fuel_m=0.004,
        r_clad_inner_m=0.0041,
        r_clad_outer_m=0.00475,
        n_fuel=6,
        n_clad=2,
        length_m=2.0,
        n_rods=3,
        rho_fuel=10000.0,
        cp_fuel=300.0,
        rho_clad=6500.0,
        cp_clad=300.0,
        k_fuel=3.0,
        k_clad=18.0,
        h_gap=1.0e4,
        h_coolant=2.0e4,
    )
    fuel, clad = rod.cylinder_volumes_m3()
    parts = rod.volumes()
    assert parts[: rod.n_fuel].sum() == pytest.approx(fuel)
    assert parts[rod.n_fuel :].sum() == pytest.approx(clad)
    gap = math.pi * (rod.r_clad_inner_m**2 - rod.r_fuel_m**2) * rod.length_m * rod.n_rods
    assert parts.sum() == pytest.approx(fuel + clad)
    assert gap > 0.0

    sodium = SFRLoop()
    sodium.temperature_k = 650.0
    slope = sodium.temperature_derivative_k_s()
    start = sodium.temperature_k
    sodium.advance_frozen_cp(1.0e-4)
    assert sodium.temperature_k == pytest.approx(start + slope * 1.0e-4, rel=1.0e-6)
    sodium.temperature_k = sodium.steady_temperature()
    held = sodium.temperature_k
    sodium.advance_frozen_cp(5.0)
    assert sodium.temperature_k == pytest.approx(held)
