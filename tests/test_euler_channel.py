"""Conservative Euler balances, convection, Jacobian and actual integration."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from core.status import DomainError, SolverError
from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from sdk.euler_experiment import run_euler_experiment, validate_euler_experiment


@pytest.mark.parametrize("boundary", ["periodic", "reservoir"])
@pytest.mark.parametrize("velocity", [-2.0, 0.0, 2.0])
def test_uniform_flow_is_exact_stationary_without_sources(boundary, velocity):
    c = EulerConfig(
        boundary=boundary,
        initial_velocity_m_s=velocity,
        left_velocity_m_s=velocity,
        right_velocity_m_s=velocity,
        total_loss_coefficient=0,
        total_ua_w_k=0,
    )
    m = EulerLiquidChannel(c)
    assert max(abs(m.derivatives(0, m._state))) == 0.0
    assert m.snapshot() == EulerLiquidChannel.from_snapshot(m.snapshot()).snapshot()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_friction_is_converted_into_internal_heat_once(sign):
    m = EulerLiquidChannel(
        EulerConfig(boundary="periodic", initial_velocity_m_s=sign * 2, total_ua_w_k=0)
    )
    n = m.config.cells
    q = m._state[: 3 * n].reshape(n, 3)
    rhs = m.derivatives(0, m._state)[: 3 * n].reshape(n, 3)
    internal_rate = (
        rhs[:, 2] - q[:, 1] / q[:, 0] * rhs[:, 1] + 0.5 * (q[:, 1] / q[:, 0]) ** 2 * rhs[:, 0]
    )
    assert sum(rhs[:, 2]) == 0
    assert sum(internal_rate) == pytest.approx(m.observation()["dissipation_w"])
    assert sum(internal_rate) > 0


@pytest.mark.parametrize("boundary", ["periodic", "reservoir"])
def test_analytic_jacobian_full_directional_and_linear_inventory_identities(boundary):
    c = EulerConfig(
        boundary=boundary,
        cells=3,
        initial_pressure_profile_pa=(7.001e6, 7.003e6, 6.999e6),
        initial_temperature_profile_k=(499.0, 500.0, 501.0),
        initial_velocity_profile_m_s=(-0.5, 0.3, 1.2),
        initial_wall_temperature_k=503.0,
    )
    m = EulerLiquidChannel(c)
    y = m._state
    jac = m.analytic_jacobian(0, y)
    direction = np.random.default_rng(42).normal(size=len(y)) * np.maximum(abs(y), 1.0)
    eps = 1e-6
    fd = (m.derivatives(0, y + eps * direction) - m.derivatives(0, y - eps * direction)) / (2 * eps)
    assert jac @ direction == pytest.approx(fd, rel=1e-4, abs=0.1)
    for inventory, ledger in ((0, -3), (1, -2), (2, -1)):
        weights = np.zeros(len(y))
        weights[inventory:9:3] = 1
        if inventory == 2:
            weights[9:12] = c.total_heat_capacity_j_k / 3
        assert np.array(jac.T @ weights - jac[ledger, :].toarray().ravel()) == pytest.approx(
            np.zeros(len(y)), abs=1e-6
        )
    assert jac[:, -3:].nnz == 0


def test_convective_momentum_flux_and_characteristic_speeds_are_present():
    m = EulerLiquidChannel(EulerConfig(initial_velocity_m_s=2, total_loss_coefficient=0))
    q = m._state[:3]
    thermo, u, _, sound = m.cell(q)
    flux, jac, _ = m.flux(q, True)
    assert flux[1] - thermo.p_pa * m.config.area_m2 == pytest.approx(q[1] ** 2 / (q[0] * m.dx))
    eig = np.sort(np.linalg.eigvals(jac * m.dx).real)
    assert eig == pytest.approx([u - sound, u, u + sound], rel=1e-8, abs=1e-6)


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_uniform_periodic_friction_matches_exact_velocity_and_energy(method):
    m = EulerLiquidChannel(EulerConfig(boundary="periodic", initial_velocity_m_s=2, total_ua_w_k=0))
    initial = m._state.copy()
    m.advance(0.2, method=method, rtol=1e-9)
    assert m.observation()["max_velocity_m_s"] == pytest.approx(2 / (1 + 10 * 0.2), abs=1e-7)
    assert m._state[2:12:3] == pytest.approx(initial[2:12:3], abs=1e-8)
    assert abs(m.observation()["momentum_residual_kg_m_s"]) < 1e-10
    assert abs(m.observation()["energy_residual_j"]) < 1e-7


def test_failed_output_interval_rolls_back_entire_state(monkeypatch):
    m = EulerLiquidChannel(EulerConfig())
    before = m.snapshot()
    original = m.derivatives

    def failing(t, y):
        if t > 0.001:
            raise SolverError("injected")
        return original(t, y)

    monkeypatch.setattr(m, "derivatives", failing)
    with pytest.raises(SolverError):
        m.advance(0.01)
    assert m.snapshot() == before and m._continuation is None


@pytest.mark.parametrize(
    "change",
    [
        dict(wave_speed_bound_m_s=1),
        dict(initial_velocity_m_s=200),
        dict(initial_pressure_profile_pa=(1.0,) * 4),
    ],
)
def test_wave_mach_and_thermodynamic_limits_are_not_clipped(change):
    with pytest.raises((ValueError, DomainError)):
        EulerLiquidChannel(replace(EulerConfig(), **change))


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_id", "pressure-driven-wall-channel"),
        ("schema_version", 2),
        ("time_s", True),
        ("state", [True] * 19),
    ],
)
def test_euler_snapshot_cannot_be_relabelled_or_corrupted(key, value):
    payload = deepcopy(EulerLiquidChannel(EulerConfig()).snapshot())
    payload[key] = value
    with pytest.raises(DomainError):
        EulerLiquidChannel.from_snapshot(payload)
    with pytest.raises(DomainError):
        DynamicWallChannel.from_snapshot(EulerLiquidChannel(EulerConfig()).snapshot())


def spec():
    return dict(
        schema_version=1,
        engine="conservative-euler-liquid-channel",
        name="euler",
        duration_s=0.02,
        step_s=0.01,
        channel=EulerConfig(initial_wall_temperature_k=501).as_dict(),
    )


def test_sdk_actual_bundle_restart_and_independent_absolute_budgets(tmp_path):
    report = run_euler_experiment(spec(), tmp_path / "run")
    assert report["numerical_acceptance"]
    assert report["budget_checks"]["momentum_absolute_kg_m_s"]
    payload = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    m = EulerLiquidChannel.from_snapshot(payload)
    assert m.snapshot() == payload
    m.advance(0.001)
    assert m.time_s == pytest.approx(0.021)
    bad = spec()
    bad["engine"] = "compatible-total-energy-channel"
    with pytest.raises(ValueError):
        validate_euler_experiment(bad)


def test_cancelled_experiment_retains_exact_initial_checkpoint(tmp_path):
    flag = tmp_path / "cancel"
    flag.write_text("stop")
    report = run_euler_experiment(spec(), tmp_path / "cancelled", cancel_file=flag)
    assert report["status"] == "cancelled" and not report["numerical_acceptance"]
    payload = json.loads((tmp_path / "cancelled" / "checkpoint.json").read_text())
    assert payload["time_s"] == 0
    assert EulerLiquidChannel.from_snapshot(payload).snapshot() == payload


def test_reflection_preserves_whole_vector_with_correct_momentum_sign():
    cfg = EulerConfig(
        cells=3,
        initial_pressure_profile_pa=(7.001e6, 7.003e6, 6.999e6),
        initial_temperature_profile_k=(499.0, 500.0, 501.0),
        initial_velocity_profile_m_s=(-0.5, 0.3, 1.2),
    )
    original = EulerLiquidChannel(cfg)
    reflected = EulerLiquidChannel(
        replace(
            cfg,
            initial_pressure_profile_pa=tuple(reversed(cfg.initial_pressure_profile_pa)),
            initial_temperature_profile_k=tuple(reversed(cfg.initial_temperature_profile_k)),
            initial_velocity_profile_m_s=tuple(
                -v for v in reversed(cfg.initial_velocity_profile_m_s)
            ),
        )
    )

    def reflect(y):
        fluid = y[:9].reshape(3, 3)[::-1].copy()
        fluid[:, 1] *= -1
        tail = y[-3:].copy()
        tail[1] *= -1
        return np.r_[fluid.ravel(), y[9:12][::-1], tail]

    assert reflected.derivatives(0, reflected._state) == pytest.approx(
        reflect(original.derivatives(0, original._state)), rel=1e-10, abs=1e-6
    )
