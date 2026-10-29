"""Группы запаздывающих нейтронов, самоэкранирование, низкая уставка и MSR."""

import numpy as np
import pytest

from control.regulation import PID, ProtectionLogic, directed_trip_time
from core.assembly import matrix_bandwidth
from core.mesh import Mesh1D
from core.status import DomainError
from physics.hydraulics.network import FlowNetwork, Junction, Volume
from physics.neutronics.diffusion import diffusion_operator, uniform_multigroup_delayed_step
from physics.neutronics.point_kinetics import PointKinetics
from physics.neutronics.xs import SelfShieldingTable
from plants.reactor_types import MSRPrecursors
from plants.vver import (
    differential_worth,
    doppler_reactivity,
    doppler_slope_per_k,
    rod_integral_worth,
    worth_between,
)
from validation.uq import normal_central_interval, sample_coverage


def test_two_delayed_groups_match_point_kinetics_and_shielding_stays_inside_the_table():
    speed = 2200.0
    nu_sigma = 1.02
    absorption = 1.0
    beta = np.array([0.002, 0.0045])
    decay = np.array([0.08, 0.3])
    flux = 2.0
    precursors = beta * nu_sigma * flux / decay
    got = uniform_multigroup_delayed_step(
        flux, precursors, absorption, nu_sigma, speed, beta, decay, 0.04
    )
    kinetics = PointKinetics(beta, decay, generation_time_s=1.0 / (speed * nu_sigma))
    rho = (nu_sigma - absorption) / nu_sigma
    reference = kinetics.reference_step(np.concatenate([[flux], speed * precursors]), rho, 0.04)
    assert got[0] == pytest.approx(reference[0], rel=1.0e-10)
    assert speed * got[1:] == pytest.approx(reference[1:], rel=1.0e-10)

    table = SelfShieldingTable(np.array([1.0, 10.0, 100.0]), np.array([0.4, 0.7, 1.0]))
    assert table.factor(1.0) == pytest.approx(0.4)
    assert table.factor(55.0) == pytest.approx(0.85)
    assert table.shielded_macro(2.0, 10.0) == pytest.approx(1.4)
    with pytest.raises(DomainError):
        table.factor(0.5)
    with pytest.raises(DomainError):
        table.factor(101.0)


def test_low_trip_ignores_the_output_step_and_a_short_dip_resets():
    def falling(time_s: float) -> float:
        return 10.0 - 8.0 * time_s

    coarse = ProtectionLogic(setpoint=6.0, delay_s=0.25, direction="low")
    fine = ProtectionLogic(setpoint=6.0, delay_s=0.25, direction="low")
    assert directed_trip_time(coarse, falling, 0.0, 1.0, 0.4) == pytest.approx(0.75)
    assert directed_trip_time(fine, falling, 0.0, 1.0, 0.05) == pytest.approx(0.75)

    def dip(time_s: float) -> float:
        if time_s <= 0.25:
            return 10.0 - 40.0 * time_s
        return 40.0 * (time_s - 0.25)

    brief = ProtectionLogic(setpoint=4.0, delay_s=0.3, direction="low")
    assert directed_trip_time(brief, dip, 0.0, 0.5, 0.05) is None
    assert brief.latched is False
    assert brief.timer_s == pytest.approx(0.0)


def test_bumpless_transfer_worth_integral_and_tridiagonal_operator():
    regulator = PID(kp=0.5, ki=0.2, kd=0.0, output_limit=1.0)
    assert regulator.engage(0.2, 0.35) == pytest.approx(0.35)
    assert regulator.step(0.2, 1.0e-6) == pytest.approx(0.35, abs=1.0e-4)
    assert regulator.engage(0.0, 5.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        PID(kp=1.0, ki=0.0, kd=0.0).engage(0.0, 0.2)

    insertion = 0.37
    assert worth_between(0.0, insertion) == pytest.approx(rod_integral_worth(insertion))
    step = 1.0e-6
    for point in (0.2, 0.37, 0.8):
        slope = (
            rod_integral_worth(point + step) - rod_integral_worth(point - step)
        ) / (2.0 * step)
        assert slope == pytest.approx(differential_worth(point), rel=1.0e-8)
    slope = doppler_slope_per_k(900.0, -1.44e-3)
    numerical = (
        doppler_reactivity(900.0 + 1.0e-3, 800.0, -1.44e-3)
        - doppler_reactivity(900.0 - 1.0e-3, 800.0, -1.44e-3)
    ) / 2.0e-3
    assert numerical == pytest.approx(slope, rel=1.0e-6)

    mesh = Mesh1D.uniform(1.0, 5)
    operator = diffusion_operator(mesh, np.array([0.01, 0.02, 0.015, 0.03, 0.012]), np.full(5, 0.2))
    assert matrix_bandwidth(operator) == 1


def test_interval_coverage_msr_balance_and_closed_check_valve():
    low, high = normal_central_interval(2.0, 0.5, 0.9)
    draws = np.random.default_rng(11).normal(2.0, 0.5, size=30000)
    assert sample_coverage(draws, low, high) == pytest.approx(0.9, abs=0.02)
    shifted = sample_coverage(draws, low + 1.0, high + 1.0)
    assert shifted < 0.7

    circulating = MSRPrecursors()
    production = 3.0
    core, loop = circulating.equilibrium_precursors(production)
    assert core + loop == pytest.approx(production / circulating.decay_constant)
    circulating.core_precursors = core
    circulating.loop_precursors = loop
    derivatives = circulating.derivatives(production)
    assert derivatives[0] == pytest.approx(0.0, abs=1.0e-8)
    assert derivatives[1] == pytest.approx(0.0, abs=1.0e-8)
    circulating.core_precursors = 0.0
    circulating.loop_precursors = 0.0
    circulating.advance(200.0, production)
    assert circulating.core_precursors == pytest.approx(core, rel=1.0e-8)
    assert circulating.loop_precursors == pytest.approx(loop, rel=1.0e-8)

    left = Volume("left", 2.0, 0.0)
    right = Volume("right", 2.0, 0.0)
    left.set_liquid(10.0e6, 560.0, boron_ppm=500.0)
    right.set_liquid(15.0e6, 560.0, boron_ppm=500.0)
    blocked = Junction(
        "check",
        "left",
        "right",
        0.05,
        1.0,
        0.0,
        mass_flow_kg_s=0.0,
        flow_law="momentum",
        check_valve=True,
    )
    network = FlowNetwork({"left": left, "right": right}, {"check": blocked})
    held = network.derivatives(network.pack())
    assert held[0] == pytest.approx(0.0, abs=1.0e-12)
    assert held[2] == pytest.approx(0.0, abs=1.0e-12)
    assert held[6] == pytest.approx(0.0, abs=1.0e-8)
    blocked.check_valve = False
    opening = network.derivatives(network.pack())
    assert opening[6] < 0.0
