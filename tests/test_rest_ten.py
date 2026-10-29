"""Выбег, сосредоточенное тело, критическая пластина, лаг, ксенон и сопло."""

import math

import numpy as np
import pytest

from components.equipment import coastdown_dissipated_j, rested_coastdown
from control.regulation import PID, pid_integrator_output
from core.events import EventSpec, collect_sign_changes
from instrumentation.channels import Actuator, Sensor, rate_limited_position, step_lag
from physics.depletion.chains import XenonIodine, xenon_macroscopic_per_m
from physics.hydraulics.two_phase_ext import critical_pressure_ratio, critical_temperature_ratio
from physics.materials.coolants import sodium_cp_j_kg_k
from physics.neutronics.diffusion import (
    absorption_reactivity_delta,
    analytical_slab_k,
    critical_buckling_m2,
    critical_slab_height_m,
    diffusion_length_m,
)
from physics.thermal.fuel_rod import analytical_cylinder_temperature, cylinder_surface_flux_w_m2
from plants.reactor_types import SFRLoop, lumped_temperature_k


def test_coastdown_work_lumped_sodium_and_critical_slab():
    omega0, torque, inertia = 20.0, 4.0, 2.0
    kinetic = 0.5 * inertia * omega0**2
    assert coastdown_dissipated_j(omega0, torque, inertia, 10.0) == pytest.approx(kinetic)
    assert rested_coastdown(omega0, torque, inertia, 10.0) == pytest.approx(0.0)
    elapsed = 5.0
    speed = omega0 - torque / inertia * elapsed
    lost = 0.5 * inertia * (omega0**2 - speed**2)
    assert coastdown_dissipated_j(omega0, torque, inertia, elapsed) == pytest.approx(lost)

    loop = SFRLoop(temperature_k=650.0)
    capacity = loop.mass_kg * sodium_cp_j_kg_k(loop.temperature_k)
    expected = lumped_temperature_k(
        loop.temperature_k,
        loop.sink_temperature_k,
        loop.fission_power_w,
        loop.ua_w_k,
        capacity,
        2.0,
    )
    assert loop.advance_frozen_cp(2.0) == pytest.approx(expected, rel=1.0e-12)

    diffusion, absorption, fission = 0.02, 0.08, 0.1
    length = diffusion_length_m(diffusion, absorption)
    buckling = (fission / absorption - 1.0) / length**2
    assert critical_buckling_m2(diffusion, absorption, fission) == pytest.approx(buckling)
    height = critical_slab_height_m(diffusion, absorption, fission)
    assert analytical_slab_k(diffusion, absorption, fission, height) == pytest.approx(
        1.0, rel=1.0e-12
    )


def test_sensor_integrator_actuator_and_cylinder():
    assert step_lag(0.0, 10.0, 2.0, 2.0) == pytest.approx(10.0 * (1.0 - math.exp(-1.0)))
    sensor = Sensor(tau_s=2.0, value=0.0)
    assert sensor.step(10.0, 2.0, np.random.default_rng(0)) == pytest.approx(
        step_lag(0.0, 10.0, 2.0, 2.0)
    )

    regulator = PID(kp=0.0, ki=0.4, kd=0.0, output_limit=10.0)
    for _ in range(4):
        output = regulator.step(0.5, 0.25)
    assert output == pytest.approx(pid_integrator_output(0.5, 0.4, 1.0))
    saturated = PID(kp=10.0, ki=1.0, kd=0.0, output_limit=1.0)
    assert saturated.step(1.0, 0.1) == pytest.approx(1.0)
    assert saturated.integral == pytest.approx(0.0)
    saturated.step(1.0, 0.1)
    assert saturated.integral == pytest.approx(0.0)

    actuator = Actuator(position=0.2, speed_limit=0.1)
    assert actuator.step(1.0, 0.5) == pytest.approx(
        rate_limited_position(0.2, 1.0, 0.1, 0.5, 0.0, 1.0)
    )
    stopped = Actuator(position=0.0, speed_limit=1.0, maximum=0.3)
    assert stopped.step(1.0, 1.0) == pytest.approx(0.3)

    power, radius, conductivity, surface = 1.0e8, 0.004, 3.0, 600.0
    flux = cylinder_surface_flux_w_m2(power, radius)
    assert flux * 2.0 * math.pi * radius == pytest.approx(power * math.pi * radius**2)

    def temperature(radius_m: float) -> float:
        return analytical_cylinder_temperature(radius_m, radius, power, conductivity, surface)

    step = 1.0e-6
    gradient = (temperature(radius + step) - temperature(radius - step)) / (2.0 * step)
    assert -conductivity * gradient == pytest.approx(flux, rel=1.0e-9)


def test_xenon_nozzle_and_event_priority():
    xenon = XenonIodine()
    fission, flux = 1.0e18, 1.0e18
    iodine, number = xenon.equilibrium(fission, flux)
    slope = xenon.derivatives(iodine, number, fission, flux)
    assert slope == pytest.approx(0.0, abs=1.0e-6 * xenon.gamma_i * fission)
    macro = xenon_macroscopic_per_m(number, xenon.sigma_xe_m2)
    assert macro == pytest.approx(number * xenon.sigma_xe_m2)
    assert absorption_reactivity_delta(macro, 0.2) == pytest.approx(-macro / 0.2)

    gamma = 1.4
    temperature_ratio = critical_temperature_ratio(gamma)
    assert temperature_ratio == pytest.approx(2.0 / (gamma + 1.0))
    assert critical_pressure_ratio(gamma) == pytest.approx(
        temperature_ratio ** (gamma / (gamma - 1.0))
    )

    def level(_time, state):
        return state[0] - 0.4

    found = collect_sign_changes(
        [EventSpec("low", level, priority=1), EventSpec("high", level, priority=5)],
        0.0,
        np.array([0.0]),
        1.0,
        np.array([1.0]),
    )
    assert [item.name for item in found] == ["high", "low"]
    assert found[0].time_s == pytest.approx(0.4)
