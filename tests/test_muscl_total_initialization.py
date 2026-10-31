"""Steady MUSCL solves checked against the production full-energy equations."""

from dataclasses import replace

import numpy as np
import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
from scripts.generate_total_energy_case import make_spec
from scripts.verify_pipe_mesh import configuration


def config(n=3, heat=3000.0, sign=1):
    return replace(
        configuration(n, heat),
        energy_formulation="compatible_total",
        enthalpy_transport="muscl_mc",
        left_pressure_pa=7.02e6 if sign > 0 else 7e6,
        right_pressure_pa=7e6 if sign > 0 else 7.02e6,
        left_temperature_k=500.0 if sign > 0 else 490.0,
        right_temperature_k=490.0 if sign > 0 else 500.0,
        junction_loss_coefficients=tuple(np.linspace(1, 4, n + 1)[::sign]),
    )


@pytest.mark.parametrize("n", [3, 8, 16])
@pytest.mark.parametrize("heat", [0.0, 3000.0])
@pytest.mark.parametrize("sign", [-1, 1])
def test_discrete_equilibrium_satisfies_full_rhs_and_exact_reflection(n, heat, sign):
    cfg = equilibrated_total_config(config(n, heat, sign))
    model = DynamicWallChannel(cfg)
    rhs = model.derivatives(0, model._state)
    assert np.max(abs(rhs[: 2 * n : 2])) == 0.0
    assert np.max(abs(rhs[1 : 2 * n : 2])) < 1e-3
    assert np.max(abs(rhs[2 * n : 3 * n + 1])) < 1e-6
    assert np.max(abs(rhs[3 * n + 1 : 4 * n + 1])) < 1e-9
    assert cfg.initial_flow_kg_s * sign > 0
    assert model.observation()["mechanical_residual_j"] == 0.0
    assert DynamicWallChannel.from_snapshot(model.snapshot()).snapshot() == model.snapshot()


def test_donor_seed_is_not_accepted_as_muscl_equilibrium():
    base = config()
    donor = equilibrated_total_config(replace(base, enthalpy_transport="donor"))
    unrefined = DynamicWallChannel(replace(donor, enthalpy_transport="muscl_mc"))
    before = max(abs(unrefined.derivatives(0, unrefined._state)[1:6:2]))
    refined = DynamicWallChannel(equilibrated_total_config(base))
    after = max(abs(refined.derivatives(0, refined._state)[1:6:2]))
    assert before > 100
    assert after < 1e-3


@pytest.mark.parametrize("sign", [-1, 1])
def test_wall_pulse_is_applied_once_to_actual_muscl_equilibrium(sign):
    base = config(sign=sign)
    steady = equilibrated_total_config(base)
    pulse = equilibrated_total_config(base, wall_offset_k=2.0)
    assert pulse.initial_pressure_profile_pa == steady.initial_pressure_profile_pa
    assert pulse.initial_temperature_profile_k == steady.initial_temperature_profile_k
    assert np.array(pulse.initial_wall_temperature_profile_k) == pytest.approx(
        np.array(steady.initial_wall_temperature_profile_k) + 2.0
    )
    model = DynamicWallChannel(pulse)
    assert model.observation()["wall_to_fluid_w"] == pytest.approx(5000.0, abs=1e-6)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_heated_reverse_muscl_equilibrium_stays_steady(method):
    model = DynamicWallChannel(equilibrated_total_config(config(sign=-1)))
    before = model._state[:13].copy()
    model.advance(0.02, method=method, rtol=1e-9, jacobian_mode="analytic")
    assert model._state[:13] == pytest.approx(before, rel=1e-10, abs=1e-5)
    assert abs(model.observation()["energy_residual_j"]) < 1e-5


def test_success_flag_does_not_replace_physical_residual_acceptance(monkeypatch):
    from types import SimpleNamespace

    import physics.hydraulics.muscl_initialization as module

    monkeypatch.setattr(
        module,
        "least_squares",
        lambda fun, x, **kw: SimpleNamespace(success=True, x=x, fun=np.zeros(7)),
    )
    with pytest.raises(DomainError, match="production physical RHS"):
        equilibrated_total_config(config())


def test_generator_selects_heated_reverse_muscl_and_rejects_invalid_inputs():
    spec = make_spec(sign=-1, transport="muscl_mc", heat_w=3000)
    assert spec["hydraulics"]["enthalpy_transport"] == "muscl_mc"
    assert spec["hydraulics"]["heat_w"] == 3000
    assert spec["hydraulics"]["initial_flow_kg_s"] < 0
    with pytest.raises(ValueError):
        make_spec(transport="muscl_mc", cells=2)
    with pytest.raises(ValueError):
        make_spec(heat_w=float("nan"))
