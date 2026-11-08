"""Short actual controlled-PWR transients; no long-horizon qualification claim."""

import pytest

from atomic_model_controlled_plant import Session
from atomic_model_controlled_plant.qualification import CRITERIA


def assert_balances(session):
    report = session.ledger.report()
    for key in ("energy_relative", "mass_relative", "boron_relative"):
        assert report[key] <= CRITERIA[key], (key, report)
    assert not report["full_horizon_accepted"]


def test_scram_residual_heat_coastdown_and_restart():
    session = Session()
    nominal = session.advance(0.1)["observation"]
    omega = session.plant.omega
    shutdown = session.advance(1.0, commands={"scram": True, "motor_off": True})["observation"]
    assert shutdown["scram_latched"]
    assert 0 < shutdown["fission_power_w"] < nominal["fission_power_w"]
    assert shutdown["thermal_power_w"] > shutdown["fission_power_w"]
    assert not session.plant.motor_on
    assert 0 < session.plant.omega < omega
    assert 0 < shutdown["mass_flow_kg_s"] < nominal["mass_flow_kg_s"]
    assert shutdown["applicability"] == "in_domain"
    assert_balances(session)
    restored = Session.restore(session.checkpoint())
    assert session.advance(0.1) == restored.advance(0.1)
    assert session.checkpoint() == restored.checkpoint()
    assert_balances(session)
    assert_balances(restored)


@pytest.mark.parametrize(
    "commands",
    [
        {"throttle": 0.0},
        {"feed_bias_kg_s": -10.0},
        {"sink_temperature_k": 305.0},
    ],
)
def test_short_boundary_disturbances_preserve_inventories(commands):
    session = Session()
    result = session.advance(0.2, commands=commands)["observation"]
    assert result["time_s"] == pytest.approx(0.2)
    assert result["applicability"] == "in_domain"
    assert result["feed_flow_kg_s"] >= 0
    assert_balances(session)
