"""Проскальзывание, отверстие, inhour, покрытие по seed и баланс РБМК."""

import numpy as np
import pytest

from components.equipment import (
    flow_from_quadratic_head,
    orifice_mass_flow,
    orifice_pressure_drop,
    quadratic_shutoff_head,
)
from core.status import DomainError
from physics.hydraulics.two_phase_ext import equilibrium_quality, isenthalpic_quality
from physics.materials.iapws import (
    quality_from_slip_void,
    saturation_properties,
    slip_void_fraction,
    void_fraction,
)
from physics.neutronics.diffusion import outscatter, removal_with_scatter
from physics.neutronics.point_kinetics import PointKinetics, inhour_root
from plants.rbmk import RBMKChannel
from scenarios.session import commands_due
from validation.uq import coverage_across_seeds


def test_slip_round_trip_channel_enthalpy_and_orifice_inverse():
    quality = 0.2
    rho_l, rho_v, slip = 800.0, 20.0, 1.7
    void = slip_void_fraction(quality, rho_l, rho_v, slip)
    assert quality_from_slip_void(void, rho_l, rho_v, slip) == pytest.approx(quality)
    assert slip_void_fraction(quality, rho_l, rho_v, 1.0) == pytest.approx(
        void_fraction(quality, rho_l, rho_v)
    )

    liquid, vapor = saturation_properties(1.0e6)
    latent = vapor.h_j_kg - liquid.h_j_kg
    produced = equilibrium_quality(5.0e5, 1.5, latent, inlet_quality=0.15)
    enthalpy = liquid.h_j_kg + produced * latent
    assert isenthalpic_quality(enthalpy, 1.0e6) == pytest.approx(produced)

    flow = 3.0
    first = orifice_pressure_drop(flow, 0.62, 0.01, 800.0)
    second = orifice_pressure_drop(flow, 0.7, 0.02, 800.0)
    assert orifice_mass_flow(0.62, 0.01, 800.0, first) == pytest.approx(flow)
    assert orifice_mass_flow(0.7, 0.02, 800.0, second) == pytest.approx(flow)
    assert first > 0.0 and second > 0.0
    assert first + second > first


def test_series_head_inhour_and_reactivity_slope():
    assert quadratic_shutoff_head(0.0, 10.0, 1.0e6) == pytest.approx(1.0e6)
    assert quadratic_shutoff_head(10.0, 10.0, 1.0e6) == pytest.approx(0.0)
    head = quadratic_shutoff_head(6.0, 10.0, 1.0e6)
    assert flow_from_quadratic_head(head, 10.0, 1.0e6) == pytest.approx(6.0)
    with pytest.raises(DomainError):
        quadratic_shutoff_head(11.0, 10.0, 1.0e6)

    kinetics = PointKinetics.thermal_u235()
    root = inhour_root(kinetics, -0.01)

    def residual(omega: float) -> float:
        return -0.01 - omega * kinetics.generation_time_s - float(
            np.sum(omega * kinetics.beta / (omega + kinetics.decay))
        )

    assert -float(np.min(kinetics.decay)) < root < 0.0
    assert residual(root) == pytest.approx(0.0, abs=1.0e-12)

    amplitude = 1.2
    precursors = kinetics.equilibrium_precursors(amplitude)
    step = 1.0e-6
    numerical = (
        kinetics.derivatives(amplitude, precursors, 0.001 + step)[0]
        - kinetics.derivatives(amplitude, precursors, 0.001 - step)[0]
    ) / (2.0 * step)
    assert numerical == pytest.approx(kinetics.amplitude_reactivity_slope(amplitude))


def test_seed_coverage_rbmk_balance_scatter_and_unknown_command():
    covered = coverage_across_seeds(0.0, 1.0, 0.9, 8000, np.arange(8))
    assert covered.min() > 0.87
    assert covered.max() < 0.93
    with pytest.raises(DomainError):
        coverage_across_seeds(0.0, 1.0, 0.9, 100, np.array([1, 1]))

    channel = RBMKChannel()
    assert channel.structure_power_residual_w() == pytest.approx(0.0, abs=1.0e-6)
    assert channel.void_reactivity(0.4) > channel.void_reactivity(0.2)
    channel.void_coefficient = -0.02
    assert channel.void_reactivity(0.4) < 0.0

    absorption = np.array([0.1, 0.2])
    scatter = np.array([[0.5, 0.05], [0.3, 0.4]])
    assert removal_with_scatter(absorption, scatter) == pytest.approx(absorption + outscatter(scatter))

    with pytest.raises(DomainError):
        commands_due([{"t": 0.0, "note": "x"}], 0.0)
    assert commands_due([{"t": 0.0, "throttle": 0.5}], 0.0)["throttle"] == pytest.approx(0.5)
