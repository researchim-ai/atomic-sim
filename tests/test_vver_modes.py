"""Режимы петли ВВЭР: компенсатор, сток и остаточное тепло после SCRAM."""

import pytest

from core.status import DomainError
from plants.vver import VVERPlant
from sdk.api import apply_commands


def test_pressurizer_deadband_and_heater_enters_the_ledger():
    plant = VVERPlant()
    heater, spray = plant._pressurizer_controls(plant.p_pressurizer_ref_pa)
    assert heater == 0.0
    assert spray == 0.0
    heater, spray = plant._pressurizer_controls(plant.p_pressurizer_ref_pa - 2.0e6)
    assert heater == pytest.approx(plant.pressurizer_heater_max_w)
    assert spray == 0.0
    heater, spray = plant._pressurizer_controls(plant.p_pressurizer_ref_pa + 2.0e6)
    assert heater == 0.0
    assert spray == pytest.approx(plant.pressurizer_spray_max_kg_s)
    plant.p_pressurizer_ref_pa += 2.0e6
    dot = plant.derivatives(0.0, plant.pack())
    assert plant._pressurizer_heater_w > 0.0
    assert sum(plant.account_rates_w(dot).values()) == pytest.approx(
        sum(plant.boundary_terms_w().values()), abs=1.0
    )


def test_warmer_sink_removes_less_condenser_heat():
    plant = VVERPlant()
    before = plant.ua_condenser * (plant.hotwell_temperature_k - plant.t_sink)
    apply_commands(plant, {"sink_temperature_k": plant.t_sink + 8.0})
    after = plant.ua_condenser * (plant.hotwell_temperature_k - plant.t_sink)
    assert after < before
    with pytest.raises(DomainError):
        plant.set_sink_temperature_k(200.0)


def test_scram_drops_fission_and_keeps_decay_heat():
    plant = VVERPlant()
    fission_0 = plant.kinetics.power_w(plant.kinetics_state[0])
    plant.scram_latched = True
    for _ in range(6):
        plant.advance(0.25, rtol=1e-4, atol=1e-2)
    fission = plant.kinetics.power_w(plant.kinetics_state[0])
    decay = plant.decay.decay_power_w(plant.decay_state)
    assert plant.rod > plant.rod_ref
    assert fission < 0.7 * fission_0
    assert decay > 0.02 * fission_0
    scale = max(1.0, abs(plant.last_energy_ledger["supplied_j"]["nuclear_w"]))
    assert abs(plant.last_energy_ledger["residual_j"]) <= 0.02 * scale
