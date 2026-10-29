"""Аккумулятор, запаздывание привода, приоритет АЗ и стержень между узлами."""

import numpy as np
import pytest

from components.accumulator import GasAccumulator
from core.mesh import Mesh1D
from core.status import DomainError
from instrumentation.channels import DelayedActuator
from physics.neutronics.diffusion import diffusion_operator, fission_operator, k_eigenvalue, rod_absorption
from plants.vver import VVERPlant


def _k_with_rod(n_cells: int, insertion: float) -> float:
    mesh = Mesh1D.uniform(1.0, n_cells)
    sigma = rod_absorption(np.full(n_cells, 1.0), insertion, 4.0)
    removal = diffusion_operator(mesh, np.full(n_cells, 0.012), sigma)
    fission = fission_operator(mesh, np.full(n_cells, 1.2))
    value, _flux, converged = k_eigenvalue(removal, fission, tol=1.0e-11)
    assert converged
    return value


def test_accumulator_follows_polytrope_and_stops_when_a_phase_vanishes():
    tank = GasAccumulator(2.0, 0.5, 4.0e6, 1.0, 800.0)
    mass = tank.liquid_mass_kg() - 80.0
    pressure = tank.set_liquid_mass_kg(mass)
    assert pressure * tank.gas_volume_m3 == pytest.approx(4.0e6 * 0.5)
    adiabatic = GasAccumulator(2.0, 0.5, 4.0e6, 1.4, 800.0)
    adiabatic.set_liquid_mass_kg(adiabatic.liquid_mass_kg() - 80.0)
    expected = 4.0e6 * (0.5 / adiabatic.gas_volume_m3) ** 1.4
    assert adiabatic.pressure_pa == pytest.approx(expected, rel=1.0e-12)
    with pytest.raises(DomainError):
        tank.set_liquid_mass_kg(0.0)
    with pytest.raises(DomainError):
        tank.set_liquid_mass_kg(tank.liquid_density_kg_m3 * tank.total_volume_m3)


def test_actuator_delay_scram_priority_and_rod_grid():
    drive = DelayedActuator(0.0, speed_limit=0.5, delay_s=0.5)
    assert drive.step(1.0, 0.4) == pytest.approx(0.0)
    assert drive.step(1.0, 0.2) == pytest.approx(0.1)

    plant = VVERPlant()
    plant.scram_latched = True
    plant.rod_command = 1.0
    derivative = plant.derivatives(0.0, plant.pack())
    assert derivative[12] == pytest.approx(plant.scram_speed)

    for n_cells in (7, 20, 21):
        sigma = rod_absorption(np.ones(n_cells), 0.37, 4.0)
        added = float(np.sum(sigma - 1.0) / n_cells)
        assert added == pytest.approx(4.0 * 0.37, rel=1.0e-12)
    assert _k_with_rod(40, 0.2) > _k_with_rod(40, 0.8)
