import json
from dataclasses import replace

import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.initialization import equilibrated_config
from scripts.verify_pipe_mesh import configuration
from sdk.dynamic_experiment import run_dynamic_experiment, validate_dynamic_experiment


@pytest.mark.parametrize("heat", [0.0, 3000.0])
def test_equilibrium_has_consistent_inventory_reference_and_small_rhs(heat):
    c = DynamicWallChannel(equilibrated_config(configuration(4, heat)))
    d = c.derivatives(0.0, c._state)
    assert max(abs(d[:8:2])) < 1e-12
    assert max(abs(d[1:8:2])) < 0.001
    assert max(abs(d[8:13])) < 1e-5
    assert max(abs(d[13:17])) < 1e-7
    assert abs(c.observation()["mass_residual_kg"]) < 1e-12
    assert abs(c.observation()["energy_residual_j"]) < 1e-8


@pytest.mark.parametrize("n", [4, 8, 16])
def test_uniform_wall_pulse_has_same_added_energy_on_every_grid(n):
    base = equilibrated_config(configuration(n, 0.0))
    pulse = equilibrated_config(configuration(n, 0.0), wall_offset_k=5.0)
    cold, warm = DynamicWallChannel(base), DynamicWallChannel(pulse)
    assert warm.initial_energy_j - cold.initial_energy_j == pytest.approx(40000.0, abs=1e-7)
    assert warm.initial_mass_kg == cold.initial_mass_kg
    assert warm.observation()["wall_to_fluid_w"] == pytest.approx(5000.0, abs=1e-6)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_pressure_profile_pa", [7e6]),
        ("initial_temperature_profile_k", [500.0, 500.0, True, 500.0]),
        ("initial_wall_temperature_profile_k", [500.0, 500.0, 0.0, 500.0]),
        ("initial_temperature_profile_k", [600.0] * 4),
    ],
)
def test_invalid_initial_profiles_are_refused(field, value):
    with pytest.raises((ValueError, DomainError)):
        replace(configuration(4, 0.0), **{field: value})


def test_profiled_snapshot_and_sdk_input_roundtrip(tmp_path):
    c = equilibrated_config(configuration(2, 0.0), wall_offset_k=5.0)
    s = dict(
        schema_version=1,
        name="profile",
        engine="pressure-driven-wall-channel",
        duration_s=0.01,
        step_s=0.005,
        hydraulics=c.as_dict(),
        solver=dict(jacobian_relative_step=1e-5),
    )
    out = tmp_path / "run"
    r = run_dynamic_experiment(s, out)
    assert r["numerical_acceptance"]
    p = json.loads((out / "checkpoint.json").read_text())
    restored = DynamicWallChannel.from_snapshot(p)
    assert restored.snapshot() == p
    normalized = json.loads((out / "input.json").read_text())
    assert validate_dynamic_experiment(normalized) == normalized


@pytest.mark.parametrize(
    "change", [dict(right_pressure_pa=7.03e6), dict(junction_loss_coefficients=(0.0,) * 5)]
)
def test_unsupported_equilibrium_is_explicitly_rejected(change):
    with pytest.raises(DomainError):
        equilibrated_config(replace(configuration(4, 0.0), **change))


def test_existing_profile_is_not_silently_replaced():
    c = equilibrated_config(configuration(4, 0.0))
    with pytest.raises(ValueError, match="unprofiled"):
        equilibrated_config(c)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_optional_central_jacobian_and_policy_switch_preserve_state(method):
    c = DynamicWallChannel(equilibrated_config(configuration(2, 0.0), wall_offset_k=1.0))
    c.enable_continuation(0.02)
    c.advance(0.01, method=method, jacobian_relative_step=1e-5)
    first = c._continuation
    c.advance(0.01, method=method, jacobian_relative_step=2e-5)
    assert c._continuation is not first
    assert c.time_s == 0.02 and abs(c.observation()["energy_residual_j"]) < 0.001


def test_invalid_central_step_does_not_mutate_state():
    c = DynamicWallChannel(configuration(2, 0.0))
    before = c.snapshot()
    with pytest.raises(ValueError):
        c.advance(0.01, jacobian_relative_step=True)
    assert c.snapshot() == before
