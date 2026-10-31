"""Contact resolution, real-EOS flux chain rule and legacy compatibility."""

from copy import deepcopy

import numpy as np
import pytest

from core.status import DomainError
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from physics.hydraulics.hllc import interface
from sdk.euler_experiment import run_euler_experiment, validate_euler_experiment


@pytest.mark.parametrize("velocity", [-1.0, 0.0, 1.0])
def test_contact_flux_is_upwind_at_equal_pressure_and_velocity(velocity):
    m = EulerLiquidChannel(
        EulerConfig(numerical_flux="hllc", total_loss_coefficient=0, total_ua_w_k=0)
    )
    left = m.inventory(7e6, 499.99, velocity)
    right = m.inventory(7e6, 500.01, velocity)
    answer = interface(m, left, right, True)
    expected = m.flux(left if velocity >= 0 else right)
    assert answer["flux"] == pytest.approx(expected, rel=1e-9, abs=1e-5)
    assert answer["contact_velocity_m_s"] == pytest.approx(velocity, abs=1e-10)
    assert answer["pressure_star_pa"] == pytest.approx(7e6, abs=1e-5)


@pytest.mark.parametrize("boundary", ["periodic", "reservoir"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sparse_analytic_hllc_jacobian_and_conservation(boundary, sign):
    m = EulerLiquidChannel(
        EulerConfig(
            cells=3,
            numerical_flux="hllc",
            boundary=boundary,
            initial_pressure_profile_pa=(7.001e6, 7.003e6, 6.999e6),
            initial_temperature_profile_k=(499.0, 500.0, 501.0),
            initial_velocity_profile_m_s=tuple(sign * np.array([0.5, 0.7, 1.2])),
            initial_wall_temperature_k=503,
        )
    )
    y = m._state
    direction = np.random.default_rng(44).normal(size=len(y)) * np.maximum(abs(y), 1)
    eps = 1e-6
    jac = m.analytic_jacobian(0, y)
    fd = (m.derivatives(0, y + eps * direction) - m.derivatives(0, y - eps * direction)) / (2 * eps)
    assert jac @ direction == pytest.approx(fd, rel=2e-4, abs=0.2)
    for item, ledger in ((0, -3), (1, -2), (2, -1)):
        weights = np.zeros(len(y))
        weights[item:9:3] = 1
        if item == 2:
            weights[9:12] = m.config.total_heat_capacity_j_k / 3
        assert jac.T @ weights - jac[ledger, :].toarray().ravel() == pytest.approx(
            np.zeros(len(y)), abs=1e-6
        )
    assert jac[:, -3:].nnz == 0


def test_stationary_entropy_jump_and_nondifferentiable_branch_are_finite():
    m = EulerLiquidChannel(
        EulerConfig(
            cells=4,
            numerical_flux="hllc",
            boundary="periodic",
            total_loss_coefficient=0,
            total_ua_w_k=0,
            initial_temperature_profile_k=(499.99, 499.99, 500.01, 500.01),
        )
    )
    assert max(abs(m.derivatives(0, m._state))) < 1e-5
    assert np.isfinite(m.analytic_jacobian(0, m._state).data).all()
    assert m.snapshot() == EulerLiquidChannel.from_snapshot(m.snapshot()).snapshot()


def test_model_schema_and_explicit_engine_cannot_be_relabelled():
    m = EulerLiquidChannel(EulerConfig(numerical_flux="hllc"))
    payload = m.snapshot()
    assert payload["model_id"] == "contact-euler-liquid-channel" and payload["schema_version"] == 4
    for field, value in (("schema_version", 3), ("model_id", "conservative-euler-liquid-channel")):
        bad = deepcopy(payload)
        bad[field] = value
        with pytest.raises(DomainError):
            EulerLiquidChannel.from_snapshot(bad)
    spec = dict(
        schema_version=1,
        engine="conservative-euler-liquid-channel",
        name="wrong",
        duration_s=0.01,
        channel=m.config.as_dict(),
    )
    with pytest.raises(ValueError):
        validate_euler_experiment(spec)
    old = EulerLiquidChannel(EulerConfig()).snapshot()
    assert "numerical_flux" not in old["config"]
    assert EulerLiquidChannel.from_snapshot(old).snapshot() == old


@pytest.mark.parametrize("cells", [3, 4, 8])
def test_ports_and_interval_integrals_use_fixed_physical_locations(cells):
    m = EulerLiquidChannel(EulerConfig(cells=cells, numerical_flux="hllc"))
    left = m.interval_inventory(0, 1.5)
    right = m.interval_inventory(1.5, 3.0)
    total = m.inventories(m._state)
    assert left["mass_kg"] + right["mass_kg"] == pytest.approx(total[0])
    assert left["total_energy_j"] + right["total_energy_j"] == pytest.approx(total[2])
    ports = m.port_observation()
    assert ports["left"]["x_m"] == 0 and ports["right"]["x_m"] == 3
    with pytest.raises(ValueError):
        m.interval_inventory(-1, 2)
    with pytest.raises(ValueError):
        m.interval_inventory(0, 4)


def test_hllc_sdk_and_restart(tmp_path):
    m = EulerLiquidChannel(
        EulerConfig(cells=3, numerical_flux="hllc", initial_wall_temperature_k=501)
    )
    spec = dict(
        schema_version=1,
        engine="contact-euler-liquid-channel",
        name="contact",
        duration_s=0.02,
        step_s=0.01,
        channel=m.config.as_dict(),
    )
    result = run_euler_experiment(spec, tmp_path / "run")
    assert result["numerical_acceptance"]
    assert "left_port_hllc_pressure_pa" in result["extrema"]


@pytest.mark.parametrize("sign", [-1, 1])
def test_heated_steady_hllc_initializer_uses_full_production_equations(sign):
    from physics.hydraulics.euler_initialization import steady_contact_config

    c = steady_contact_config(
        EulerConfig(
            cells=4,
            numerical_flux="hllc",
            heat_w=10,
            left_pressure_pa=7.02e6 if sign > 0 else 7e6,
            right_pressure_pa=7e6 if sign > 0 else 7.02e6,
            initial_pressure_pa=7.01e6,
        )
    )
    m = EulerLiquidChannel(c)
    assert max(abs(m.derivatives(0, m._state)[:16])) < 1e-3
    assert c.initial_velocity_profile_m_s[0] * sign > 0
    before = m._state[:16].copy()
    m.advance(0.01, rtol=1e-9)
    assert m._state[:16] == pytest.approx(before, rel=1e-10, abs=1e-4)


def test_net_momentum_event_is_localized_inside_output_interval():
    m = EulerLiquidChannel(
        EulerConfig(
            cells=4,
            numerical_flux="hllc",
            initial_velocity_m_s=-0.005,
            left_pressure_pa=7.02e6,
            initial_pressure_pa=7.01e6,
            total_ua_w_k=0,
        )
    )
    m.advance(0.005, rtol=1e-9, max_step_s=5e-5)
    events = [e for e in m.last_events if e["direction"] == 1]
    assert events and 0 < events[0]["t"] < 0.005
    assert events[0]["kind"] == "net_momentum_zero"
    assert abs(m.observation()["momentum_residual_kg_m_s"]) < 1e-7
