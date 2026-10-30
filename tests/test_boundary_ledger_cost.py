"""Энергетический ledger использует те же граничные потоки без полного RHS."""

import numpy as np
import pytest

from plants.vver import VVERPlant


@pytest.mark.parametrize("mode", ["nominal", "hot_wall", "pump_off", "feed_bias", "trip"])
def test_boundary_refresh_matches_full_rhs(mode):
    plant = VVERPlant()
    if mode == "hot_wall":
        plant.sg_wall_temperature += 20.0
    if mode == "pump_off":
        plant.motor_on = False
    if mode == "feed_bias":
        plant.feed_bias_kg_s = -50.0
    if mode == "trip":
        plant.throttle = 0.0
    state = plant.pack()
    plant.derivatives(0.0, state)
    expected = plant.boundary_terms_w()
    plant.refresh_algebraic(state)
    assert plant.boundary_terms_w() == pytest.approx(expected, abs=1e-5)


def test_ledger_matches_full_rhs_quadrature_without_calling_rhs(monkeypatch):
    plant = VVERPlant()
    state0 = plant.pack()
    plant.advance(0.2, rtol=1e-5, atol=1e-4)
    state1 = plant.pack()
    plant.derivatives(0.0, state0)
    before = plant.boundary_terms_w()
    plant.derivatives(0.2, state1)
    after = plant.boundary_terms_w()
    expected = {key: 0.1 * (before[key] + after[key]) for key in before}

    def unnecessary(*args, **kwargs):
        raise AssertionError("Ledger must not compute neutron/phase/momentum RHS")

    monkeypatch.setattr(plant, "derivatives", unnecessary)
    report = plant.energy_ledger([(0.0, state0), (0.2, state1)])
    assert report["supplied_j"] == pytest.approx(expected, rel=1e-12, abs=1e-5)
    assert np.array_equal(plant.pack(), state1)
