"""Двухгрупповой баклинг, изоэнтропа, вал, cp воды и энергия фаз."""

import math

import numpy as np
import pytest

from components.grid import shaft_work_j, speed_after_constant_torques
from core.status import DomainError
from physics.hydraulics.two_phase_ext import (
    TwoFluidVolume,
    critical_temperature_ratio,
    ideal_gas_cp_j_kg_k,
    ideal_gas_sound_speed_m_s,
    interface_energy_split_j,
    isentropic_speed_m_s,
)
from physics.materials.iapws import isobaric_enthalpy_slope_j_kg_k, properties_pt
from physics.neutronics.diffusion import two_group_buckling_k, two_group_critical_buckling


def test_two_group_critical_buckling_and_subcritical_refusal():
    absorption = np.array([0.01, 0.08])
    fission = np.array([0.012, 0.10])
    diffusion = np.array([0.02, 0.005])
    scatter = 0.04
    chi = np.array([1.0, 0.0])
    buckling = two_group_critical_buckling(absorption, scatter, fission, diffusion)
    assert buckling > 0.0
    assert two_group_buckling_k(
        absorption, scatter, fission, chi, diffusion, buckling
    ) == pytest.approx(1.0, rel=1.0e-12)
    assert two_group_buckling_k(absorption, scatter, fission, chi, diffusion, buckling * 1.2) < 1.0
    with pytest.raises(DomainError):
        two_group_critical_buckling(absorption, scatter, np.array([0.001, 0.001]), diffusion)


def test_isentropic_throat_shaft_and_water_cp():
    gamma, gas_r, stagnation = 1.4, 287.0, 300.0
    static = stagnation * critical_temperature_ratio(gamma)
    speed = isentropic_speed_m_s(gamma, gas_r, stagnation, static)
    assert speed == pytest.approx(ideal_gas_sound_speed_m_s(gamma, gas_r, static), rel=1.0e-12)
    capacity = ideal_gas_cp_j_kg_k(gamma, gas_r)
    assert 0.5 * speed**2 == pytest.approx(capacity * (stagnation - static), rel=1.0e-12)

    omega0, inertia, mechanical, electrical, time_s = 100.0, 20.0, 30.0, 10.0, 2.0
    omega1 = speed_after_constant_torques(omega0, inertia, mechanical, electrical, time_s)
    assert shaft_work_j(omega0, inertia, mechanical, electrical, time_s) == pytest.approx(
        0.5 * inertia * (omega1**2 - omega0**2)
    )

    for pressure, temperature in ((5.0e6, 400.0), (0.5e6, 600.0)):
        state = properties_pt(pressure, temperature)
        slope = isobaric_enthalpy_slope_j_kg_k(pressure, temperature, 0.05)
        assert slope == pytest.approx(state.cp_j_kg_k, rel=1.0e-6)


def test_interface_heat_enters_the_phase_sum():
    liquid_h, vapor_h = 4.0e5, 2.7e6
    volume = TwoFluidVolume(10.0, 1.0, 8.0e6, 3.0e6)
    moved_target = 0.1
    step = 0.2
    heat_w = moved_target / step * (vapor_h - liquid_h)
    before_liquid, before_vapor = volume.liquid_energy_j, volume.vapor_energy_j
    before = volume.total_energy_j
    moved = volume.transfer(moved_target / step, step, liquid_h, vapor_h, interface_heat_w=heat_w)
    liquid_delta, vapor_delta = interface_energy_split_j(moved, liquid_h, vapor_h, heat_w * step)
    assert volume.liquid_energy_j - before_liquid == pytest.approx(liquid_delta, rel=1.0e-12)
    assert volume.vapor_energy_j - before_vapor == pytest.approx(vapor_delta, rel=1.0e-12)
    assert volume.total_energy_j - before == pytest.approx(heat_w * step, rel=1.0e-12)
    assert liquid_delta + vapor_delta == pytest.approx(heat_w * step)

    adiabatic = TwoFluidVolume(10.0, 1.0, 8.0e6, 3.0e6)
    stored = adiabatic.total_energy_j
    adiabatic.transfer(0.4, 0.25, liquid_h, vapor_h, interface_heat_w=0.0)
    assert adiabatic.total_energy_j == pytest.approx(stored, rel=1.0e-12)
    assert math.isfinite(adiabatic.total_energy_j)
