"""Групповой beta_eff, баланс твэла, смешение, критическое отношение и лаг рампы."""

import math

import numpy as np
import pytest

from components.equipment import orifice_flow_derivative, orifice_mass_flow
from components.heat_exchanger import HeatExchangerWall
from core.status import DomainError
from instrumentation.channels import Sensor, ramp_lag
from physics.depletion.chains import XenonIodine
from physics.hydraulics.network import boiloff_inventory
from physics.hydraulics.two_phase_ext import choked_mass_flow, critical_pressure_ratio
from physics.materials.iapws import mix_isobaric, properties_pt
from physics.neutronics.diffusion import (
    adjoint_delayed_fraction,
    adjoint_delayed_group_fractions,
)
from physics.neutronics.point_kinetics import PointKinetics
from physics.thermal.fuel_rod import FuelRod


def _rod() -> FuelRod:
    return FuelRod(
        r_fuel_m=0.004,
        r_clad_inner_m=0.0041,
        r_clad_outer_m=0.00475,
        n_fuel=8,
        n_clad=2,
        length_m=1.0,
        n_rods=1,
        rho_fuel=10000.0,
        cp_fuel=300.0,
        rho_clad=6500.0,
        cp_clad=300.0,
        k_fuel=3.0,
        k_clad=18.0,
        h_gap=1.0e5,
        h_coolant=2.0e4,
    )


def _sensor_ramp(steps: int) -> float:
    sensor = Sensor(tau_s=2.0, value=0.0)
    dt = 4.0 / steps
    rng = np.random.default_rng(0)
    value = 0.0
    for index in range(steps):
        value = sensor.step(1.5 * (index + 1) * dt, dt, rng)
    return value


def test_group_beta_sums_and_the_pin_rejects_its_heat_to_the_coolant():
    flux = np.array([1.0, 2.0, 0.5])
    weight = np.array([0.3, 1.0, 0.8])
    fission = np.array([1.0, 1.2, 0.9])
    widths = np.full(3, 0.2)
    groups = np.array(
        [
            [0.001, 0.003, 0.002],
            [0.004, 0.001, 0.0015],
        ]
    )
    fractions = adjoint_delayed_group_fractions(flux, weight, groups, fission, widths)
    combined = adjoint_delayed_fraction(flux, weight, groups.sum(axis=0), fission, widths)
    assert fractions.sum() == pytest.approx(combined)
    assert fractions[0] == pytest.approx(
        adjoint_delayed_fraction(flux, weight, groups[0], fission, widths)
    )
    flat = adjoint_delayed_group_fractions(flux, weight, np.array([0.002, 0.004]), fission, widths)
    assert flat == pytest.approx([0.002, 0.004])

    rod = _rod()
    power = 4.0e8
    temperature = rod.steady_state(power, 580.0)
    assert rod.boundary_heat_w(temperature, 580.0) == pytest.approx(
        float(np.sum(rod.generation(power)))
    )
    assert np.all(temperature > 580.0)


def test_mix_choke_ratio_boiloff_and_ramp_lag():
    cold = properties_pt(1.0e6, 350.0)
    warm = properties_pt(1.0e6, 450.0)
    mass, enthalpy, state = mix_isobaric(2.0, cold.h_j_kg, 3.0, warm.h_j_kg, 1.0e6)
    assert mass == pytest.approx(5.0)
    assert mass * enthalpy == pytest.approx(2.0 * cold.h_j_kg + 3.0 * warm.h_j_kg)
    assert state.h_j_kg == pytest.approx(enthalpy, rel=1.0e-6)

    ratio = critical_pressure_ratio(1.4)
    assert ratio == pytest.approx((2.0 / 2.4) ** (1.4 / 0.4))
    upstream = 1.0e6
    critical = 12.0
    assert choked_mass_flow(
        upstream, 0.2 * ratio * upstream, ratio, critical, 1.0e-3
    ) == pytest.approx(critical)
    assert choked_mass_flow(
        upstream, 0.8 * ratio * upstream, ratio, critical, 1.0e-3
    ) == pytest.approx(critical)
    assert choked_mass_flow(upstream, 0.95 * upstream, ratio, critical, 1.0e-3) < critical
    with pytest.raises(DomainError):
        critical_pressure_ratio(1.0)

    mass_left, boron = boiloff_inventory(100.0, 0.2, 5.0, 4.0)
    assert mass_left == pytest.approx(80.0)
    assert boron == pytest.approx(0.2)
    assert boron / mass_left == pytest.approx(0.2 / 80.0)
    with pytest.raises(DomainError):
        boiloff_inventory(10.0, 0.1, 5.0, 2.0)

    exact = ramp_lag(0.0, 1.5, 2.0, 4.0)
    coarse = _sensor_ramp(8)
    fine = _sensor_ramp(40)
    assert abs(fine - exact) < abs(coarse - exact)


def test_wall_split_observation_orifice_slope_and_xenon_field():
    wall = HeatExchangerWall(2.0e6, 1.0e6, 8.0e5, 500.0)
    primary, stored, secondary = wall.power_split_w(560.0, 450.0)
    assert primary + stored + secondary == pytest.approx(0.0, abs=1.0e-8)
    equilibrium = wall.equilibrium_temperature_k(560.0, 450.0)
    wall.temperature_k = equilibrium
    _primary, stored_eq, _secondary = wall.power_split_w(560.0, 450.0)
    assert stored_eq == pytest.approx(0.0, abs=1.0e-6)

    kinetics = PointKinetics.thermal_u235()
    assert kinetics.visible_amplitude_modes(-0.01) == 7
    with pytest.raises(DomainError):
        kinetics.visible_amplitude_modes(-0.01, tolerance=1.0e-2)

    drop = 2.0e5
    slope = orifice_flow_derivative(0.62, 0.01, 800.0, drop)
    numerical = (
        orifice_mass_flow(0.62, 0.01, 800.0, drop + 1.0)
        - orifice_mass_flow(0.62, 0.01, 800.0, drop - 1.0)
    ) / 2.0
    assert numerical == pytest.approx(slope, rel=1.0e-6)
    with pytest.raises(DomainError):
        orifice_flow_derivative(0.62, 0.01, 800.0, 0.0)

    xenon = XenonIodine()
    low = xenon.equilibrium(1.0e18, 1.0e16)
    high = xenon.equilibrium(1.0e20, 1.0e18)
    iodine, concentration = xenon.shutdown_field(
        np.array([low[0], high[0]]), np.array([low[1], high[1]]), 3.0e4
    )
    scalar = xenon.shutdown_xenon(high[0], high[1], 3.0e4)
    assert iodine[1] == pytest.approx(scalar[0])
    assert concentration[1] == pytest.approx(scalar[1])
    low_peak = xenon.shutdown_peak_time_s(low[0], low[1])
    high_peak = xenon.shutdown_peak_time_s(high[0], high[1])
    assert high_peak > low_peak
    assert math.isfinite(high_peak)
