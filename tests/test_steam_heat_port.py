"""Analytic reciprocal thermal exchange, independent of a second steam solver."""

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from physics.power.steam_cycle import SteamPowerCycle


def model():
    raw = json.loads(Path("scenarios/ai05_dynamic_steam_cycle.json").read_text())
    cfg, c = deepcopy(raw["model"]), deepcopy(raw["controls"])
    cfg["hot_wall"].update(fluid_ua_w_k=0, initial_temperature_k=500)
    cfg["cold_wall"].update(fluid_ua_w_k=0, ambient_ua_w_k=0)
    cfg["turbine"]["bearing_drag_nm_s_rad"] = 0
    c.update(valve_opening=0, feed_mass_flow_kg_s=0, heater_power_w=0, generator_connected=False)
    return SteamPowerCycle(cfg, c)


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize("bath_initial", [400.0, 600.0])
def test_reciprocal_bath_exchange_matches_closed_form(method, bath_initial):
    cycle = model()
    ch = cycle.configuration["hot_wall"]["heat_capacity_j_k"]
    cb = 3e5
    ua = 1e3
    initial = np.zeros(15)

    def power(y):
        return ua * ((bath_initial - 500) + (y[13] - y[4]))

    def rates(t, y):
        q = power(y)
        return np.r_[cycle.derivatives(t, y[:13], hot_wall_port_w=q), -q / cb, q]

    solved = solve_ivp(
        rates, (0.0, 60.0), initial, method=method, rtol=1e-10, atol=1e-9, max_step=0.25
    )
    assert solved.success
    y = solved.y[:, -1]
    equilibrium = (cb * bath_initial + ch * 500) / (cb + ch)
    expected = equilibrium + (500 - equilibrium) * math.exp(-ua * (1 / ch + 1 / cb) * 60)
    assert 500 + y[4] == pytest.approx(expected, abs=1e-7)
    assert y[14] == pytest.approx(ch * (expected - 500), abs=0.02)
    assert abs(cb * y[13] + y[14]) < 1e-5
    row = cycle.observation(y[:13], 60, hot_wall_port_w=power(y), hot_wall_port_energy_j=y[14])
    assert abs(row["energy_residual_j"]) < 1e-5
    assert abs(row["stored_energy_change_j"] + cb * y[13]) < 1e-5
    assert abs(row["power_residual_w"]) < 1e-7
    assert row["energy_integrals_j"]["heater"] == 0
    assert math.copysign(1, row["hot_wall_port"]["energy_into_cycle_j"]) == math.copysign(
        1, bath_initial - 500
    )
    with pytest.raises(ValueError, match="ledger"):
        cycle.observation(y[:13], 60)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1000"])
@pytest.mark.parametrize("field", ["hot_wall_port_w", "hot_wall_port_energy_j"])
def test_port_arguments_are_finite_numbers_without_mutation(field, value):
    cycle = model()
    before = cycle.snapshot()
    with pytest.raises(ValueError):
        cycle.observation(**{field: value})
    assert cycle.snapshot() == before


def test_default_schema_and_snapshot_remain_standalone():
    cycle = model()
    assert "hot_wall_port" not in cycle.observation()
    assert SteamPowerCycle.from_snapshot(cycle.snapshot()).snapshot() == cycle.snapshot()
