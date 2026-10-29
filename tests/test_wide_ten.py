"""Параллельный Пуазейль, поперечный ток, Клапейрон, натрий и генератор."""

import math

import numpy as np
import pytest

from components.grid import (
    electrical_power_w,
    grid_electrical_torque_nm,
    speed_after_constant_torques,
)
from components.heat_exchanger import (
    crossflow_cmin_unmixed_asymptote,
    crossflow_cmin_unmixed_effectiveness,
)
from physics.hydraulics.closures import (
    parallel_poiseuille_flow_kg_s,
    poiseuille_drop_pa,
    poiseuille_flow_kg_s,
)
from physics.hydraulics.two_phase_ext import drift_void_fraction, quality_from_drift_void
from physics.materials.coolants import (
    sodium_cp_j_kg_k,
    sodium_density_kg_m3,
    sodium_internal_energy_j_kg,
)
from physics.materials.iapws import clapeyron_slope_pa_per_k, saturation_pressure_mpa


def test_parallel_pipes_crossflow_and_clapeyron():
    drop = 2.0e4
    one = poiseuille_flow_kg_s(drop, 0.01, 1.0e-3, 2.0, 800.0)
    assert poiseuille_drop_pa(one, 0.01, 1.0e-3, 2.0, 800.0) == pytest.approx(drop, rel=1.0e-12)
    both = parallel_poiseuille_flow_kg_s(drop, [0.01, 0.01], 1.0e-3, 2.0, 800.0)
    assert both == pytest.approx(2.0 * one)

    assert crossflow_cmin_unmixed_effectiveness(0.0, 0.4) == pytest.approx(0.0)
    assert crossflow_cmin_unmixed_effectiveness(2.0, 0.0) == pytest.approx(1.0 - math.exp(-2.0))
    assert crossflow_cmin_unmixed_effectiveness(40.0, 0.5) == pytest.approx(
        crossflow_cmin_unmixed_asymptote(0.5), abs=1.0e-12
    )
    small = crossflow_cmin_unmixed_effectiveness(1.0e-4, 0.3)
    assert small == pytest.approx(1.0e-4, rel=1.0e-3)

    temperature = 450.0
    thermo = clapeyron_slope_pa_per_k(temperature)
    step = 0.02
    numerical = (
        saturation_pressure_mpa(temperature + step) - saturation_pressure_mpa(temperature - step)
    ) / (2.0 * step) * 1.0e6
    assert numerical == pytest.approx(thermo, rel=1.0e-3)


def test_sodium_generator_and_distributed_vapor():
    assert sodium_density_kg_m3(500.0) == pytest.approx(1016.0 - 0.235 * 500.0)
    assert (sodium_density_kg_m3(501.0) - sodium_density_kg_m3(499.0)) / 2.0 == pytest.approx(-0.235)
    center = 700.0
    delta = 1.0e-3
    slope = (
        sodium_internal_energy_j_kg(center + delta) - sodium_internal_energy_j_kg(center - delta)
    ) / (2.0 * delta)
    assert slope == pytest.approx(sodium_cp_j_kg_k(center), rel=1.0e-6)

    omega = 100.0
    power = 2.0e6
    torque = grid_electrical_torque_nm(True, power, omega)
    assert electrical_power_w(torque, omega) == pytest.approx(power)
    assert grid_electrical_torque_nm(False, power, omega) == pytest.approx(0.0)
    assert speed_after_constant_torques(100.0, 20.0, 30.0, 10.0, 2.0) == pytest.approx(102.0)

    rho_l, rho_v, flux = 800.0, 20.0, 1000.0
    void = drift_void_fraction(1.0, rho_l, rho_v, flux, distribution=1.2, drift_velocity_m_s=0.2)
    assert void < 1.0
    assert quality_from_drift_void(
        void, rho_l, rho_v, flux, distribution=1.2, drift_velocity_m_s=0.2
    ) == pytest.approx(1.0)
    assert np.isfinite(void)
