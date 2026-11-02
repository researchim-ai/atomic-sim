"""Regression for sub-resolution equilibrium, restart and control changes."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from physics.power.primary_steam import PrimarySteamCycle
from scripts.design_primary_steam import design_steady_primary_steam
from scripts.primary_steam_references import energy_balances, power_balance, recorded_state, system
from sdk.primary_steam_experiment import prepare


def point(flow=100):
    raw = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    raw["timeline"] = []
    spec, _ = design_steady_primary_steam(raw, primary_mass_flow_kg_s=flow)
    return spec, prepare(spec)[1]


@pytest.mark.parametrize("flow", [100, -100])
def test_bdf_preserves_stationarity_for_120_seconds_without_step_collapse(flow):
    spec, model = point(flow)
    initial, _, _, _, data, offset = system(spec)
    result = model.advance_until_limit(
        120, method="BDF", rtol=1e-10, atol_scale=0.01, max_step_s=0.02
    )
    assert result["boundary"] is None
    assert len(result["solver"]["mesh_times_s"]) < 10000
    row = result["observation"]
    balances = energy_balances(recorded_state(row, data), initial, data, offset, spec)
    assert (
        max(
            abs(balances[k]) for k in ("primary_residual_j", "steam_residual_j", "total_residual_j")
        )
        < 0.01
    )
    assert abs(row["power_residual_w"]) < 1e-5
    assert abs(row["primary"]["loop"]["mass_flow_kg_s"] - flow) < 1e-6


def test_restart_reconstructs_rates_without_cache_or_trajectory_history():
    _, model = point()
    model.advance_until_limit(3, method="BDF", rtol=1e-10, atol_scale=0.01, max_step_s=0.02)
    payload = model.snapshot()
    restored = PrimarySteamCycle.from_snapshot(payload)
    np.testing.assert_array_equal(
        model.derivatives(model.time_s, model._state),
        restored.derivatives(restored.time_s, restored._state),
    )
    assert restored.snapshot() == payload
    for instance in (model, restored):
        instance.advance_until_limit(1, method="BDF", rtol=1e-10, atol_scale=0.01, max_step_s=0.02)
    np.testing.assert_array_equal(model._state, restored._state)


def test_control_changes_and_return_to_original_are_history_independent():
    _, model = point()
    state = model._state.copy()
    before = model.derivatives(0, state)
    original = model.boundary_controls
    changed = deepcopy(original)
    changed["primary_wall_heater_power_w"] *= 1.2
    model.set_controls(changed)
    other = model.derivatives(0, state)
    assert not np.array_equal(before, other)
    np.testing.assert_array_equal(other, model._exact_derivatives(0, state))
    model.set_controls(original)
    np.testing.assert_array_equal(before, model.derivatives(0, state))


@pytest.mark.parametrize("fraction", [0.25, 0.75, 1.1, 10])
def test_precision_window_preserves_nonlinear_rate_values_and_balances(fraction):
    spec, model = point()
    state = model._state.copy()
    indices, scales = model._physical_indices_scales()
    state[indices] += fraction * scales * 1e-12
    corrected = model.derivatives(0, state)
    exact = model._exact_derivatives(0, state)
    assert np.max(np.abs(corrected - exact)) < 1e-4
    if fraction >= 1.1:
        np.testing.assert_array_equal(corrected, exact)
    # Arbitrarily perturbed stocks have no matching integral history; test
    # the instantaneous conservation law rather than presenting a fake ledger.
    initial, _, _, _, data, offset = system(spec)
    n = data["n"]
    p = model._primary
    independent = initial.copy()
    independent[0] = state[0]
    independent[1] = state[p._size]
    independent[2 : 2 + 2 * n] += state[1 : 1 + 2 * n]
    independent[offset : offset + 13] += state[model._offset : model._offset + 13]
    rates = np.r_[
        corrected[0],
        corrected[p._size],
        corrected[1 : 1 + 2 * n],
        corrected[p._size + 1 : p._size + 4],
        corrected[1 + 2 * n],
        corrected[2 + 2 * n] + corrected[-1] - corrected[p._size + 3],
        corrected[model._offset :],
    ]
    assert abs(power_balance(independent, rates, data, offset, spec)) < 1e-5
