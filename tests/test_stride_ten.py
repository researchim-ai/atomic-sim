"""Пуазейль, излучение зазора, стационар с источником, выбег и шаблон диффузии."""

import math

import numpy as np
import pytest

from components.equipment import coastdown_omega, rested_coastdown
from components.separator import SteamDrum
from control.regulation import RodBank, RodBanks
from core.residual import AlgebraicOrifice, ResidualSystem, consistent_initial
from core.solvers import integrate
from core.status import DomainError
from physics.hydraulics.closures import (
    friction_darcy,
    heat_transfer_coefficient,
    poiseuille_drop_pa,
)
from physics.materials.iapws import saturation_properties
from physics.neutronics.point_kinetics import PointKinetics
from physics.neutronics.transport import _sparse_system, couplings_are_orthogonal
from physics.thermal.fuel_rod import gap_conductance, radiation_gap_h
from plants.vver import WORTH, worth_between


def test_poiseuille_gap_radiation_and_source_equilibrium():
    flow, diameter, viscosity, length, density = 0.005, 0.01, 1.0e-3, 2.0, 800.0
    reynolds = abs(flow) * diameter / (viscosity * math.pi * diameter**2 / 4.0)
    friction_factor = 64.0 / reynolds
    dynamic = flow**2 / (density * (math.pi * diameter**2 / 4.0) ** 2)
    hand = friction_factor * (length / diameter) * dynamic / 2.0
    drop = friction_darcy(flow, diameter, viscosity, 1.0e-5, length, density)
    assert drop == pytest.approx(hand)
    assert drop == pytest.approx(poiseuille_drop_pa(flow, diameter, viscosity, length, density))
    turbulent = friction_darcy(0.05, 0.01, 1.0e-3, 1.0e-5, 2.0, 800.0)
    assert turbulent != pytest.approx(poiseuille_drop_pa(0.05, 0.01, 1.0e-3, 2.0, 800.0))
    with pytest.raises(DomainError):
        heat_transfer_coefficient(0.0393, 0.01, 1.0e-3, 4000.0, 0.6)

    fuel = 900.0
    clad = 620.0
    quotient = (fuel**4 - clad**4) / (fuel - clad)
    polynomial = (fuel + clad) * (fuel**2 + clad**2)
    assert quotient == pytest.approx(polynomial)
    emitted = radiation_gap_h(0.6, fuel, clad)
    assert emitted == pytest.approx(0.6 * 5.670374419e-8 * polynomial)
    assert gap_conductance(0.3, 1.0e-4, 0.6, fuel, clad) == pytest.approx(0.3 / 1.0e-4 + emitted)
    assert radiation_gap_h(0.6, 800.0, 800.0) == pytest.approx(
        0.6 * 5.670374419e-8 * 4.0 * 800.0**3
    )
    with pytest.raises(DomainError):
        gap_conductance(0.3, 0.0, 0.6, fuel, clad)

    kinetics = PointKinetics.thermal_u235()
    state = kinetics.equilibrium_with_source(-0.01, 2.0)
    rate = kinetics.derivatives(state[0], state[1:], -0.01, source=2.0)
    assert rate == pytest.approx(0.0, abs=1.0e-8)
    with pytest.raises(DomainError):
        kinetics.equilibrium_with_source(0.001, 2.0)


def test_coastdown_stops_stencil_is_orthogonal_and_orifice_initializes():
    assert rested_coastdown(20.0, 4.0, 2.0, 5.0) == pytest.approx(
        coastdown_omega(20.0, 4.0, 2.0, 5.0)
    )
    assert rested_coastdown(20.0, 4.0, 2.0, 20.0) == pytest.approx(0.0)
    assert rested_coastdown(20.0, 4.0, 2.0, 10.0) == pytest.approx(0.0)

    shape = (3, 4)
    matrix, _fission = _sparse_system(
        shape, (1.2, 0.8), 0.01 + 0.02 * np.linspace(0.0, 1.0, 12).reshape(shape), 0.2, 0.3
    )
    assert couplings_are_orthogonal(matrix, shape)
    strides = {1, shape[1]}
    coo = matrix.tocoo()
    corner = 1 + shape[1]
    assert corner not in {abs(int(row) - int(col)) for row, col in zip(coo.row, coo.col)}
    assert strides <= {abs(int(row) - int(col)) for row, col in zip(coo.row, coo.col) if row != col}

    orifice = ResidualSystem([AlgebraicOrifice(0.62, 0.01, 800.0, 2.0e5)])
    solved, rate = consistent_initial(orifice, 0.0, np.array([0.2]), np.array([False]))
    scale = (0.62 * 0.01) ** 2 * 2.0 * 800.0 * 2.0e5
    assert solved[0] == pytest.approx(math.sqrt(scale), rel=1.0e-8)
    assert rate[0] == pytest.approx(0.0)
    orifice.require_consistent(0.0, solved, rate)


def test_bdf_beats_euler_drum_level_and_rod_banks():
    exact = math.exp(-1.0)
    steps = 10
    euler = 1.0
    dt = 1.0 / steps
    for _ in range(steps):
        euler += dt * (-euler)
    numerical = integrate(lambda _t, y: -y, np.array([1.0]), (0.0, 1.0), rtol=1e-8, atol=1e-10)
    assert abs(float(numerical.final[0]) - exact) < abs(euler - exact) / 10.0

    pressure = 6.0e6
    liquid, vapor = saturation_properties(pressure)
    area = 2.0
    volume = 10.0
    drum = SteamDrum(
        area,
        volume,
        pressure,
        0.25 * volume * liquid.rho_kg_m3,
        0.75 * volume * vapor.rho_kg_m3,
    )
    assert drum.level_m() == pytest.approx(0.25 * volume / area)
    with pytest.raises(DomainError):
        SteamDrum(area, volume, pressure, 1.2 * volume * liquid.rho_kg_m3, 1.0).level_m()

    banks = RodBanks(
        [
            RodBank("shim", 0.2, 0.01, 0.003, 0.5),
            RodBank("regulating", 0.8, 0.01, 0.005, 0.1),
        ]
    )
    expected = -(0.003 * worth_between(0.5, 0.2) + 0.005 * worth_between(0.1, 0.8))
    assert banks.reactivity() == pytest.approx(expected)
    assert WORTH > 0.0
