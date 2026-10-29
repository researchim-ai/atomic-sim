"""Мгновенный период, гомогенизация, NTU, петли, синус и команды сценария."""

import math

import numpy as np
import pytest

from components.heat_exchanger import counterflow_effectiveness, counterflow_heat_w
from core.status import DomainError
from physics.hydraulics.closures import friction_darcy, haaland_friction_factor
from physics.neutronics.diffusion import (
    homogenize_parallel,
    homogenize_series,
    sine_slab_peak_to_average,
)
from physics.neutronics.point_kinetics import PointKinetics
from plants.reactor_types import (
    HTGRBrayton,
    ideal_brayton_efficiency,
    isentropic_temperature_ratio,
    split_loop_rises_k,
)
from scenarios.session import commands_due


def test_prompt_period_homogenization_and_ntu_limits():
    kinetics = PointKinetics.thermal_u235()
    state = kinetics.initial_state(1.0)
    period = kinetics.inverse_period(state[0], state[1:], 0.002)
    assert period == pytest.approx(0.002 / kinetics.generation_time_s)

    sigma, diffusion = homogenize_parallel([1.0, 3.0], [0.2, 0.6], [0.01, 0.03])
    assert sigma == pytest.approx((0.2 + 3.0 * 0.6) / 4.0)
    assert diffusion == pytest.approx((0.01 + 3.0 * 0.03) / 4.0)
    series_sigma, series_d = homogenize_series([1.0, 1.0], [0.2, 0.4], [0.01, 0.04])
    assert series_sigma == pytest.approx(0.3)
    assert series_d == pytest.approx(2.0 / (1.0 / 0.01 + 1.0 / 0.04))
    assert series_d < 0.5 * (0.01 + 0.04)

    assert counterflow_effectiveness(0.0, 0.5) == pytest.approx(0.0)
    assert counterflow_effectiveness(2.0, 0.0) == pytest.approx(1.0 - math.exp(-2.0))
    assert counterflow_effectiveness(2.0, 1.0) == pytest.approx(2.0 / 3.0)
    heat = counterflow_heat_w(1.0e5, 500.0, 400.0, 3.0, 0.8)
    assert 0.0 < heat <= 1.0e5 * 100.0
    with pytest.raises(DomainError):
        counterflow_effectiveness(1.0, 1.2)


def test_loops_sine_peak_haaland_and_brayton():
    rises = split_loop_rises_k([6.0e6, 4.0e6], [50.0, 25.0], 4200.0)
    heats = np.array([50.0, 25.0]) * 4200.0 * rises
    assert heats.sum() == pytest.approx(1.0e7)
    assert rises[1] > rises[0]
    assert "vver-lumped-loop" != "flow-split-loops"

    coarse = sine_slab_peak_to_average(9)
    mid = sine_slab_peak_to_average(17)
    fine = sine_slab_peak_to_average(33)
    target = math.pi / 2.0
    assert abs(fine - target) < abs(mid - target) < abs(coarse - target)
    with pytest.raises(DomainError):
        sine_slab_peak_to_average(8)

    smooth = haaland_friction_factor(1.0e5, 0.0, 0.01)
    rough = haaland_friction_factor(1.0e5, 1.0e-4, 0.01)
    assert rough > smooth > 0.0
    turbulent = friction_darcy(0.05, 0.01, 1.0e-3, 1.0e-4, 2.0, 800.0)
    assert turbulent > 0.0

    gamma = 5.0 / 3.0
    ratio = 2.0
    efficiency = ideal_brayton_efficiency(ratio, gamma)
    temperature_ratio = isentropic_temperature_ratio(ratio, gamma)
    assert efficiency == pytest.approx(1.0 - 1.0 / temperature_ratio)
    turbine_inlet = 1000.0
    compressor_inlet = turbine_inlet / temperature_ratio / 1.5
    carnot = 1.0 - compressor_inlet / turbine_inlet
    assert 0.0 < efficiency < carnot
    with pytest.raises(DomainError):
        ideal_brayton_efficiency(1.0, gamma)


def test_commands_at_one_time_and_graphite_relaxes():
    timeline = [
        {"t": 0.0, "throttle": 0.8, "boron_ppm": 900.0},
        {"t": 0.0, "throttle": 0.4, "scram": True},
        {"t": 2.0, "rod_command": -1.0},
    ]
    now = commands_due(timeline, 0.0)
    assert now["throttle"] == pytest.approx(0.4)
    assert now["boron_ppm"] == pytest.approx(900.0)
    assert now["scram"] is True
    assert "rod_command" not in now
    later = commands_due(timeline, 2.0)
    assert later["rod_command"] == pytest.approx(-1.0)
    assert later["scram"] is True

    plant = HTGRBrayton()
    start = plant.graphite_temperature_k
    core = plant.core_temperature_k
    rate = plant.ua_fuel_graphite / plant.graphite_capacity_j_k
    plant.advance_graphite(30.0)
    expected = core + (start - core) * math.exp(-rate * 30.0)
    assert plant.graphite_temperature_k == pytest.approx(expected)
    assert plant.passport.model_id == "htgr-helium-brayton"
