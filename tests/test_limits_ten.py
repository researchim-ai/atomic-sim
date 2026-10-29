"""Пределы NTU, насыщение, останов насоса, inhour и сброс защиты."""

import math

import numpy as np
import pytest

from components.equipment import rested_coastdown, time_to_rest_s
from components.heat_exchanger import (
    counterflow_asymptote,
    counterflow_effectiveness,
    parallelflow_asymptote,
    parallelflow_effectiveness,
)
from control.regulation import ProtectionLogic
from core.status import DomainError
from physics.hydraulics.two_phase_ext import channel_exit_quality, critical_pressure_ratio
from physics.materials.coolants import flibe_density_kg_m3, lead_density_kg_m3
from physics.materials.iapws import saturation_pressure_mpa, saturation_temperature_k
from physics.neutronics.point_kinetics import PointKinetics, inhour_root
from plants.vver import differential_worth, rod_integral_worth


def test_ntu_limits_critical_ratio_and_saturation_round_trip():
    assert counterflow_asymptote(0.4) == pytest.approx(1.0)
    assert counterflow_effectiveness(80.0, 0.4) == pytest.approx(1.0, abs=1.0e-12)
    assert parallelflow_asymptote(0.5) == pytest.approx(1.0 / 1.5)
    assert parallelflow_effectiveness(80.0, 0.5) == pytest.approx(parallelflow_asymptote(0.5), abs=1.0e-12)
    with pytest.raises(DomainError):
        parallelflow_asymptote(1.2)

    assert critical_pressure_ratio(1.000001) == pytest.approx(math.exp(-0.5), rel=1.0e-4)

    for temperature in (300.0, 450.0, 600.0):
        pressure = saturation_pressure_mpa(temperature)
        assert saturation_temperature_k(pressure) == pytest.approx(temperature, rel=1.0e-8)


def test_stop_time_rod_ends_inhour_and_channel_heat():
    assert time_to_rest_s(20.0, 4.0, 2.0) == pytest.approx(10.0)
    assert rested_coastdown(20.0, 4.0, 2.0, 10.0) == pytest.approx(0.0)
    assert rested_coastdown(20.0, 4.0, 2.0, 9.9) > 0.0

    assert rod_integral_worth(0.0) == pytest.approx(0.0)
    assert rod_integral_worth(1.0) == pytest.approx(1.0)
    assert differential_worth(0.0) == pytest.approx(0.0)
    assert differential_worth(1.0) == pytest.approx(0.0)

    kinetics = PointKinetics.thermal_u235()
    assert 0.0 < kinetics.beta_total < 0.01
    root = inhour_root(kinetics, 0.001)
    assert 0.0 < root < 0.001 / kinetics.generation_time_s

    def residual(omega: float) -> float:
        return 0.001 - omega * kinetics.generation_time_s - float(
            np.sum(omega * kinetics.beta / (omega + kinetics.decay))
        )

    assert residual(root) == pytest.approx(0.0, abs=1.0e-12)

    mass, heat_capacity, cold, saturation, power, latent = 2.0, 4200.0, 500.0, 530.0, 2.0e5, 1.5e6
    distance = 7.0
    quality = channel_exit_quality(distance, mass, heat_capacity, cold, saturation, power, latent)
    sensible = mass * heat_capacity * (saturation - cold)
    assert sensible + mass * quality * latent == pytest.approx(power * distance)


def test_protection_reset_and_coolant_density_slopes():
    logic = ProtectionLogic(setpoint=1.0, delay_s=0.0)
    assert logic.step([2.0, 2.0, 0.0], 0.1) is True
    assert logic.step([2.0, 2.0, 0.0], 0.1, reset=True) is True
    assert logic.step([0.0, 0.0, 0.0], 0.1, reset=True) is False
    assert logic.latched is False
    assert logic.timer_s == pytest.approx(0.0)

    assert lead_density_kg_m3(700.0) > lead_density_kg_m3(1000.0)
    assert flibe_density_kg_m3(850.0) > flibe_density_kg_m3(1050.0)
