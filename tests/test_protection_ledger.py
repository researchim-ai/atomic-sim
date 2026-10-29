"""Защёлка внутри шага, снимок таймера и баланс запасов ВВЭР."""

import pytest

from plants.vver import VVERPlant


def _arm(plant: VVERPlant, delay_s: float) -> None:
    plant.power_protection.setpoint = 0.5 * plant.fission_power_nom
    plant.power_protection.delay_s = delay_s
    plant.power_protection.timer_s = 0.0
    plant.power_protection.latched = False


def _rod_after(steps, delay_s: float) -> VVERPlant:
    plant = VVERPlant()
    _arm(plant, delay_s)
    for step in steps:
        plant.advance(step, rtol=1e-5, atol=1e-5)
    return plant


def test_trip_time_does_not_follow_the_output_step():
    one = _rod_after([1.0], 0.4)
    two = _rod_after([0.5, 0.5], 0.4)
    four = _rod_after([0.25, 0.25, 0.25, 0.25], 0.4)
    expected = one.rod_ref + one.scram_speed * 0.6
    assert one.last_protection_event_s == pytest.approx(0.4, abs=1e-6)
    assert one.rod == pytest.approx(expected, abs=1e-3)
    assert two.rod == pytest.approx(one.rod, abs=1e-3)
    assert four.rod == pytest.approx(one.rod, abs=1e-3)
    assert one.power_protection.latched
    assert one.scram_latched


def test_protection_timer_survives_snapshot_and_finishes_on_the_same_rod():
    whole = _rod_after([1.0], 0.5)
    partial = VVERPlant()
    _arm(partial, 0.5)
    partial.advance(0.3, rtol=1e-5, atol=1e-5)
    assert partial.power_protection.latched is False
    assert partial.power_protection.timer_s == pytest.approx(0.3, abs=1e-4)
    restored = VVERPlant()
    restored.restore(partial.snapshot())
    restored.advance(0.7, rtol=1e-5, atol=1e-5)
    assert restored.power_protection.latched
    assert restored.rod == pytest.approx(whole.rod, abs=1e-3)
    assert restored.last_protection_event_s == pytest.approx(0.5, abs=1e-4)


def test_wall_wave_and_integral_ledger_close():
    plant = VVERPlant()
    equilibrium = plant.sg_wall_temperature
    plant.sg_wall_temperature = equilibrium + 30.0
    dot = plant.derivatives(0.0, plant.pack())
    primary = plant.network.volumes["sg_primary"].temperature_k
    secondary = plant.network.volumes["sg_secondary"].temperature_k
    to_wall, to_secondary = plant.sg_wall.heat_flows_w(primary, secondary)
    assert to_wall < to_secondary
    assert plant.account_rates_w(dot)["wall_w"] == pytest.approx(
        to_wall - to_secondary, rel=0.0, abs=1.0
    )
    assert dot[-1] < 0.0
    plant.advance(0.5, rtol=1e-5, atol=1e-4)
    assert plant.sg_wall_temperature < equilibrium + 29.0
    ledger = plant.last_energy_ledger
    scale = max(1.0, abs(ledger["supplied_j"]["nuclear_w"]))
    assert abs(ledger["residual_j"]) <= 1.0e-3 * scale
    assert ledger["delta_j"]["wall_j"] < 0.0
