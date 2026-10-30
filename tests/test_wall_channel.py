"""Стена и жидкость обмениваются энергией внутри одного консервативного шага."""

import json

import numpy as np
import pytest

from core.status import DomainError
from physics.hydraulics.isobaric import ChannelConfig, IsobaricChannel
from physics.hydraulics.wall_channel import WallChannel, WallConfig
from physics.materials.iapws import saturation_properties


def reservoir(off=0.7):
    liquid, vapor = saturation_properties(7e6)
    h0 = float(liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg))
    base = ChannelConfig(1, 0.01, 7e6, h0, 0.0, 5000.0)
    return WallConfig(base, 1000.0, 100.0, (float(liquid.T_k) + 50.0,), off)


@pytest.mark.parametrize("temperature_offset", [50.0, -10.0])
def test_interface_heat_cancels_in_total_energy_rate(temperature_offset):
    liquid, vapor = saturation_properties(7e6)
    h0 = float(liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg))
    c = WallChannel(
        WallConfig(
            ChannelConfig(1, 0.01, 7e6, h0, 1.0, 5000.0),
            1000.0,
            100.0,
            (float(liquid.T_k) + temperature_offset,),
        )
    )
    rates = c.rates(0.0, c._state)
    assert sum(rates["du"]) + rates["wall_energy_rate_w"] == pytest.approx(
        rates["energy_boundary_w"], abs=1e-8
    )
    assert sum(rates["dm"]) == pytest.approx(rates["mass_boundary_kg_s"], abs=1e-12)
    assert rates["wall_to_fluid_w"] == pytest.approx(100 * temperature_offset)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_coupled_wall_and_boiling_follow_piecewise_analytic_solution(method):
    config = reservoir()
    c = WallChannel(config)
    c.enable_continuation(2.0)
    row = c.advance(2.0, rtol=1e-9, method=method, max_step_s=0.1)
    liquid, vapor = saturation_properties(7e6)
    elapsed = 2.0 - config.power_off_time_s
    tau = config.total_heat_capacity_j_k / config.total_ua_w_k
    expected_tw = liquid.T_k + 50 * np.exp(-elapsed / tau)
    transferred = 5000 * config.power_off_time_s + 1000 * 50 * (1 - np.exp(-elapsed / tau))
    h0 = config.channel.initial_enthalpy_j_kg[0]
    v0 = liquid.v_m3_kg + 0.1 * (vapor.v_m3_kg - liquid.v_m3_kg)
    b = (vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg)
    v = v0 * np.exp(b * transferred / config.channel.total_volume_m3)
    assert row["max_wall_temperature_k"] == pytest.approx(expected_tw, abs=2e-6)
    assert row["outlet_enthalpy_j_kg"] == pytest.approx(h0 + (v - v0) / b, abs=0.02)
    assert row["stored_mass_kg"] == pytest.approx(config.channel.total_volume_m3 / v, abs=1e-7)
    assert abs(row["energy_residual_j"]) < 0.1
    assert row["heat_w"] == 0.0
    assert row["wall_to_fluid_w"] > 0


def test_wall_and_fluid_energy_are_not_double_counted():
    c = WallChannel(reservoir())
    row = c.observation()
    assert row["stored_energy_j"] == pytest.approx(row["fluid_energy_j"] + row["wall_energy_j"])
    assert c.initial_energy_j == row["stored_energy_j"]


def test_steady_wall_matches_bare_channel_energy_gain():
    liquid, _ = saturation_properties(7e6)
    inlet = float(liquid.h_j_kg - 20000)
    profile = tuple(inlet + (i + 1) * 75000 for i in range(4))
    cfg = ChannelConfig(4, 0.004, 7e6, inlet, 1.0, 3e5, initial_enthalpy_j_kg=profile)
    c = WallChannel(WallConfig(cfg, 8000.0, 10000.0))
    rates = c.rates(0.0, c._state)
    assert rates["dh"] == pytest.approx(np.zeros(4), abs=1e-7)
    assert rates["d_wall"] == pytest.approx(np.zeros(4), abs=1e-10)
    bare = IsobaricChannel(cfg)
    assert c.observation()["outlet_enthalpy_j_kg"] == bare.observation()["outlet_enthalpy_j_kg"]


def test_checkpoint_on_switch_resumes_with_right_hand_power():
    c = WallChannel(reservoir())
    c.advance(0.7, rtol=1e-9)
    payload = json.loads(json.dumps(c.snapshot()))
    restored = WallChannel.from_snapshot(payload)
    assert restored.snapshot() == payload
    assert restored.observation()["heat_w"] == 0.0
    a = c.advance(0.25, rtol=1e-9)
    b = restored.advance(0.25, rtol=1e-9)
    assert a == b


def test_failure_in_second_segment_rolls_back_complete_step(monkeypatch):
    c = WallChannel(reservoir())
    before = c.snapshot()
    original = c.derivatives

    def fail(t, y):
        if c._source_side is False:
            raise DomainError("after switch")
        return original(t, y)

    monkeypatch.setattr(c, "derivatives", fail)
    with pytest.raises(DomainError, match="after switch"):
        c.advance(2.0)
    assert c.snapshot() == before
    assert c._continuation is None
    assert c._source_side is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_ua_w_k", 0.0),
        ("total_heat_capacity_j_k", True),
        ("initial_temperature_k", (0.0,)),
        ("power_off_time_s", -1.0),
    ],
)
def test_invalid_wall_inputs_are_refused(field, value):
    data = dict(channel=reservoir().channel, total_heat_capacity_j_k=1000.0, total_ua_w_k=100.0)
    data[field] = value
    with pytest.raises(ValueError):
        WallConfig(**data)
