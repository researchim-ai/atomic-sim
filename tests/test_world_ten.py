"""Токи 3D, cusping, продувка, index-1, спектр деления и идентифицируемость."""

import math

import numpy as np
import pytest

from core.mesh import Mesh1D
from core.residual import DuplicateConstraint, IndexOneDrive, ResidualSystem, consistent_initial
from core.status import DomainError
from coupling.maps import conserve_integral
from physics.depletion.chains import DecayHeat
from physics.hydraulics.network import charge_letdown_inventory, diluted_boron_kg
from physics.hydraulics.two_phase_ext import isenthalpic_quality
from physics.materials.iapws import saturation_properties
from physics.neutronics.diffusion import (
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    rod_absorption,
    rod_absorption_staircase,
)
from physics.neutronics.point_kinetics import PointKinetics, batch_derivatives
from physics.neutronics.transport import divergence_from_currents, divergence_nd, operator_leakage
from physics.neutronics.xs import fission_power_w, fission_source_per_group
from validation.uq import linear_normal_moments, require_identifiable


def _rod_k(cells: int, absorption: np.ndarray) -> float:
    mesh = Mesh1D.uniform(1.0, cells)
    removal = diffusion_operator(mesh, np.full(cells, 0.015), absorption)
    fission = fission_operator(mesh, np.full(cells, 0.45))
    value, _flux, ok = k_eigenvalue(removal, fission, tol=1.0e-11)
    assert ok
    return value


def test_three_dimensional_currents_match_leakage_and_cusping_shrinks():
    shape = (4, 3, 3)
    lengths = (1.2, 0.8, 0.6)
    rng = np.random.default_rng(7)
    diffusion = 0.01 + 0.02 * rng.random(shape)
    flux = rng.random(shape)
    leakage = operator_leakage(shape, lengths, diffusion, np.full(shape, 0.2), flux)
    assert divergence_nd(shape, lengths, diffusion, flux).reshape(-1) == pytest.approx(
        leakage, abs=1.0e-12
    )
    flat = (6, 5)
    field = rng.random(flat)
    conductivity = 0.008 + 0.01 * rng.random(flat)
    assert divergence_nd(flat, (1.0, 0.7), conductivity, field) == pytest.approx(
        divergence_from_currents(flat, (1.0, 0.7), conductivity, field)
    )

    errors = []
    for cells in (8, 16, 32):
        base = np.full(cells, 0.2)
        partial = rod_absorption(base, 0.37, 0.8)
        stairs = rod_absorption_staircase(base, 0.37, 0.8, snap="floor")
        errors.append(abs(_rod_k(cells, stairs) - _rod_k(cells, partial)))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] > 0.0


def test_flash_letdown_and_index_one_start():
    pressure = 1.0e6
    liquid, vapor = saturation_properties(pressure)
    enthalpy = 0.5 * (liquid.h_j_kg + vapor.h_j_kg)
    assert isenthalpic_quality(enthalpy, pressure) == pytest.approx(0.5, abs=1.0e-10)
    lower = isenthalpic_quality(enthalpy, 0.5e6)
    assert lower > 0.5
    with pytest.raises(DomainError):
        isenthalpic_quality(liquid.h_j_kg, pressure)

    mass, boron = charge_letdown_inventory(100.0, 2.0, 5.0, 0.0, 0.0, 4.0)
    assert mass == pytest.approx(80.0)
    assert boron / mass == pytest.approx(2.0 / 100.0)
    filled_mass, filled_boron = charge_letdown_inventory(100.0, 2.0, 0.0, 5.0, 0.01, 4.0)
    assert filled_mass == pytest.approx(120.0)
    assert filled_boron == pytest.approx(2.0 + 0.2)
    same_mass, same_boron = charge_letdown_inventory(50.0, 1.0, 3.0, 3.0, 0.0, 10.0)
    assert same_mass == pytest.approx(50.0)
    assert same_boron == pytest.approx(diluted_boron_kg(50.0, 1.0, 3.0, 0.0, 10.0))
    with pytest.raises(DomainError):
        charge_letdown_inventory(10.0, 1.0, 5.0, 0.0, 0.0, 2.0)

    state_mass = 10.0
    state_boron = 1.0
    step = 2.0 / 4000
    for _ in range(4000):
        concentration = state_boron / state_mass
        state_boron += (4.0 * 0.05 - 1.0 * concentration) * step
        state_mass += (4.0 - 1.0) * step
    exact_mass, exact_boron = charge_letdown_inventory(10.0, 1.0, 1.0, 4.0, 0.05, 2.0)
    assert exact_mass == pytest.approx(state_mass, rel=1.0e-6)
    assert exact_boron == pytest.approx(state_boron, rel=1.0e-4)

    drive = ResidualSystem([IndexOneDrive(1.5)])
    solved, rate = consistent_initial(drive, 0.0, np.array([0.2, 5.0]), np.array([True, False]))
    assert solved == pytest.approx(np.array([0.2, 1.5]))
    assert rate == pytest.approx(np.array([1.3, 0.0]))
    drive.require_consistent(0.0, solved, rate)
    with pytest.raises(DomainError):
        consistent_initial(
            ResidualSystem([DuplicateConstraint()]),
            0.0,
            np.array([0.0, 0.0]),
            np.array([False, False]),
        )


def test_spectrum_power_decay_source_and_identifiability():
    chi = np.array([0.7, 0.3])
    nu = np.array([0.2, 1.4])
    flux = np.array([3.0, 5.0])
    source = fission_source_per_group(chi, nu, flux)
    assert np.sum(source) == pytest.approx(np.sum(nu * flux))
    with pytest.raises(DomainError):
        fission_source_per_group(np.array([0.7, 0.2]), nu, flux)
    kappa = np.array([[1.0e-11, 2.0e-11], [3.0e-11, 4.0e-11]])
    field = np.array([[1.0e17, 2.0e17], [4.0e17, 5.0e17]])
    volume = np.array([0.1, 0.2])
    power = fission_power_w(kappa, field, volume)
    assert power == pytest.approx(float(np.sum(kappa * field * volume)))
    density = np.sum(kappa * field, axis=0)
    mapped = conserve_integral(density, np.array([1.0, 1.0]), np.array([0.5, 1.5]))
    assert np.sum(mapped * np.array([0.5, 1.5])) == pytest.approx(np.sum(density))

    heat = DecayHeat.illustrative()
    stored = np.array([1.0e8, 2.0e8, 4.0e8])
    released, remaining = heat.shutdown_release_j(stored, 25.0)
    assert released + float(np.sum(remaining)) == pytest.approx(float(np.sum(stored)))
    assert remaining == pytest.approx(stored * np.exp(-heat.decay_constants * 25.0))

    kinetics = PointKinetics.thermal_u235(source=2.5)
    amplitudes = np.array([1.0, 0.4])
    precursors = np.vstack(
        [kinetics.equilibrium_precursors(1.0), kinetics.equilibrium_precursors(0.4)]
    )
    reactivity = np.array([-0.002, 0.0])
    package_source = np.array([2.5, 7.0])
    dn, _dc = batch_derivatives(amplitudes, precursors, reactivity, kinetics, package_source)
    scalar = kinetics.derivatives(0.4, precursors[1], 0.0, source=7.0)
    assert dn[1] == pytest.approx(scalar[0])
    default_dn, _default_dc = batch_derivatives(amplitudes, precursors, reactivity, kinetics)
    assert default_dn[0] == pytest.approx(
        kinetics.derivatives(1.0, precursors[0], -0.002, source=2.5)[0]
    )

    generation = 5.0e-5
    external = 1.0e6
    rho = -0.003
    amplitude_row = np.array([-generation / rho, -external / rho])
    with pytest.raises(DomainError):
        require_identifiable(amplitude_row)
    assert require_identifiable(np.vstack([amplitude_row, [0.0, 1.0]])) == 2
    expected, variance = linear_normal_moments(
        np.array([1.0, 2.0]), np.array([[1.0, 0.5], [0.5, 2.0]]), np.array([3.0, -1.0])
    )
    assert expected == pytest.approx(1.0)
    assert variance == pytest.approx(8.0)
    draws = np.random.default_rng(4).multivariate_normal(
        [1.0, 2.0], [[1.0, 0.5], [0.5, 2.0]], size=20000
    ) @ np.array([3.0, -1.0])
    assert abs(float(np.mean(draws)) - expected) < 4.0 * math.sqrt(variance / draws.size)
    with pytest.raises(DomainError):
        linear_normal_moments(
            np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, -0.2]]), np.array([1.0, 1.0])
        )
