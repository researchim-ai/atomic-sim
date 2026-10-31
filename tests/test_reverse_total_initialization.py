"""Reflection and actual physical residual checks for reverse steady flow."""

from dataclasses import replace

import numpy as np
import pytest

from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
from scripts.verify_pipe_mesh import configuration


@pytest.mark.parametrize("cells", [1, 3, 8])
@pytest.mark.parametrize("heat", [0.0, 3000.0])
def test_reverse_equilibrium_satisfies_all_physical_equations(cells, heat):
    base = replace(
        configuration(cells, heat),
        energy_formulation="compatible_total",
        left_pressure_pa=7e6,
        right_pressure_pa=7.02e6,
        left_temperature_k=490.0,
        right_temperature_k=500.0,
        junction_loss_coefficients=tuple(np.linspace(1.0, 4.0, cells + 1)),
    )
    cfg = equilibrated_total_config(base)
    model = DynamicWallChannel(cfg)
    derivative = model.derivatives(0.0, model._state)
    assert cfg.initial_flow_kg_s < 0
    # Physical components only: work and boundary integrals need not be steady.
    assert max(abs(derivative[: 4 * cells + 1])) < 0.001
    assert model.observation()["energy_residual_j"] == 0.0
    assert model.observation()["mechanical_residual_j"] == 0.0
    assert cfg.junction_loss_coefficients == base.junction_loss_coefficients
    assert cfg.left_temperature_k == 490.0


@pytest.mark.parametrize("offset", [0.0, 2.0])
def test_reverse_profile_is_reflection_with_asymmetric_losses(offset):
    base = replace(
        configuration(3, 3000.0),
        energy_formulation="compatible_total",
        left_temperature_k=500.0,
        right_temperature_k=490.0,
        junction_loss_coefficients=(1.0, 2.0, 4.0, 3.0),
    )
    direct = equilibrated_total_config(base, wall_offset_k=offset)
    reverse = equilibrated_total_config(
        replace(
            base,
            left_pressure_pa=base.right_pressure_pa,
            right_pressure_pa=base.left_pressure_pa,
            left_temperature_k=base.right_temperature_k,
            right_temperature_k=base.left_temperature_k,
            junction_loss_coefficients=tuple(reversed(base.junction_loss_coefficients)),
        ),
        wall_offset_k=offset,
    )
    assert reverse.initial_flow_kg_s == pytest.approx(-direct.initial_flow_kg_s)
    for field in (
        "initial_pressure_profile_pa",
        "initial_temperature_profile_k",
        "initial_wall_temperature_profile_k",
    ):
        assert getattr(reverse, field) == pytest.approx(tuple(reversed(getattr(direct, field))))
    model = DynamicWallChannel(reverse)
    restored = DynamicWallChannel.from_snapshot(model.snapshot())
    assert restored.snapshot() == model.snapshot()


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_reverse_heated_equilibrium_remains_steady_after_advance(method):
    cfg = equilibrated_total_config(
        replace(
            configuration(3, 3000.0),
            energy_formulation="compatible_total",
            left_pressure_pa=7e6,
            right_pressure_pa=7.02e6,
        )
    )
    model = DynamicWallChannel(cfg)
    physical = model._state[:13].copy()
    model.advance(0.01, method=method, rtol=1e-9, jacobian_mode="analytic")
    assert model._state[:13] == pytest.approx(physical, rel=1e-10, abs=1e-5)
    obs = model.observation()
    assert abs(obs["energy_residual_j"]) < 1e-5
    assert abs(obs["mechanical_residual_j"]) < 1e-5
    restored = DynamicWallChannel.from_snapshot(model.snapshot())
    assert restored.snapshot() == model.snapshot()
