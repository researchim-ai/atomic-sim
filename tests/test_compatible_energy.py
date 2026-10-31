import json
from dataclasses import replace

import numpy as np
import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicWallChannel
from physics.hydraulics.initialization import equilibrated_config
from physics.hydraulics.mechanical_energy import (
    allocation,
    kinetic_gradient_hessian,
)
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
from scripts.verify_pipe_mesh import configuration
from sdk.dynamic_experiment import run_dynamic_experiment, validate_dynamic_experiment


def config(n=3, **kw):
    return replace(configuration(n, 0.0), energy_formulation="compatible_total", **kw)


@pytest.mark.parametrize("n", [1, 3, 8])
def test_dual_volume_allocation_is_positive_and_does_not_duplicate_kinetic_energy(n):
    w = allocation(n)
    assert w.sum(axis=0) == pytest.approx(np.ones(n + 1))
    c = DynamicWallChannel(config(n, initial_flow_kg_s=2.0))
    lengths = np.array(c.config.junction_lengths_m)
    assert w @ lengths == pytest.approx(np.full(n, 3.0 / n))
    assert np.all(w >= 0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_friction_converts_kinetic_loss_to_heat_once(sign):
    c = DynamicWallChannel(
        config(
            left_pressure_pa=7e6,
            right_pressure_pa=7e6,
            initial_pressure_pa=7e6,
            initial_flow_kg_s=sign * 2.0,
        )
    )
    r = c.rates(0.0, c._state)
    d = r["derivative"]
    mech = r["mechanical"]
    heat = sum(d[1:6:2]) + r["wall_energy_rate_w"]
    assert sum(mech["dissipation"]) > 0
    assert heat == pytest.approx(sum(mech["dissipation"]), abs=1e-7)
    assert sum(mech["kinetic_rate"]) == pytest.approx(-sum(mech["dissipation"]))
    assert sum(mech["pressure_work"]) == 0.0
    assert abs(heat + sum(mech["kinetic_rate"]) - r["boundary_energy_w"]) < 1e-7


@pytest.mark.parametrize("scheme", ["donor", "muscl_mc"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_mass_density_work_and_full_energy_chain_rule(scheme, sign):
    c = DynamicWallChannel(
        config(
            enthalpy_transport=scheme,
            initial_flow_kg_s=sign,
            initial_temperature_profile_k=(497.0, 500.0, 504.0),
            initial_wall_temperature_profile_k=(510.0,) * 3,
        )
    )
    y = c._state.copy()
    y[6:10] = sign * np.array([0.8, 1.1, 0.9, 1.2])
    r = c.rates(0.0, y)
    d = r["derivative"]
    g, _ = kinetic_gradient_hessian(c, y)
    thermal = sum(d[1:6:2]) + r["wall_energy_rate_w"]
    assert abs(sum(r["mechanical"]["density_work"])) > 1e-6
    assert g @ d == pytest.approx(sum(r["mechanical"]["kinetic_rate"]), abs=1e-10)
    assert thermal + g @ d == pytest.approx(r["boundary_energy_w"], abs=1e-7)
    assert abs(r["mechanical"]["identity_residual_w"]) < 1e-10
    assert np.all(r["mechanical"]["dissipation"] >= 0)
    assert sum(d[:6:2]) == pytest.approx(r["boundary_mass_kg_s"], abs=1e-12)


@pytest.mark.parametrize("scheme", ["donor", "muscl_mc"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_compatible_sparse_jacobian_and_nonlinear_conservation_identity(scheme, sign):
    c = DynamicWallChannel(
        config(
            enthalpy_transport=scheme,
            initial_flow_kg_s=sign,
            initial_temperature_profile_k=(496.0, 500.0, 505.0),
            initial_wall_temperature_profile_k=(510.0,) * 3,
        )
    )
    y = c._state.copy()
    y[6:10] = sign * np.array([0.8, 1.1, 0.9, 1.2])
    jac = c.analytic_jacobian(0.0, y)
    direction = np.random.default_rng(44).normal(size=len(y)) * np.maximum(abs(y), 1.0)
    eps = 1e-6
    fd = (c.derivatives(0.0, y + eps * direction) - c.derivatives(0.0, y - eps * direction)) / (
        2 * eps
    )
    assert jac @ direction == pytest.approx(fd, rel=1e-4, abs=0.02)
    g, hess = kinetic_gradient_hessian(c, y)
    d = c.derivatives(0.0, y)
    for i in range(3):
        g[2 * i + 1] += 1.0
        g[10 + i] += c.config.total_heat_capacity_j_k / 3
    identity = jac.T @ g + hess @ d - jac[-1, :].toarray().ravel()
    assert max(abs(identity)) < 1e-6
    assert jac[:, -5:].nnz == 0


@pytest.mark.parametrize("heat", [0.0, 3000.0])
def test_compatible_equilibrium_has_small_full_physical_rhs(heat):
    cfg = equilibrated_total_config(replace(config(), heat_w=heat))
    c = DynamicWallChannel(cfg)
    d = c.derivatives(0.0, c._state)
    assert max(abs(d[:13])) < 0.001
    assert c.observation()["energy_residual_j"] == 0.0
    assert c.observation()["mechanical_residual_j"] == 0.0
    with pytest.raises(DomainError):
        equilibrated_config(config())


def spec():
    cfg = equilibrated_total_config(config(), wall_offset_k=2.0)
    return dict(
        schema_version=1,
        name="compatible",
        engine="compatible-total-energy-channel",
        duration_s=0.02,
        step_s=0.01,
        hydraulics=cfg.as_dict(),
        solver=dict(jacobian_mode="analytic", rtol=1e-8),
    )


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_schema2_bundle_ledger_and_restart(method, tmp_path):
    s = spec()
    s["solver"]["method"] = method
    folder = tmp_path / "run"
    r = run_dynamic_experiment(s, folder)
    assert r["numerical_acceptance"] and r["balance"]["mechanical_absolute_j"] < 1e-4
    p = json.loads((folder / "checkpoint.json").read_text())
    assert p["schema_version"] == 2 and p["model_id"] == "compatible-total-energy-liquid-channel"
    c = DynamicWallChannel.from_snapshot(p)
    assert c.snapshot() == p
    c.advance(0.001, jacobian_mode="analytic")
    assert abs(c.observation()["mechanical_residual_j"]) < 1e-4
    normalized = json.loads((folder / "input.json").read_text())
    assert validate_dynamic_experiment(normalized) == normalized


@pytest.mark.parametrize(
    "key,value",
    [("schema_version", 1), ("model_id", "pressure-driven-wall-channel"), ("schema_version", True)],
)
def test_energy_formulation_cannot_be_relabelled_in_checkpoint(key, value):
    p = DynamicWallChannel(config()).snapshot()
    p[key] = value
    with pytest.raises(DomainError):
        DynamicWallChannel.from_snapshot(p)


def test_engine_and_ledger_budget_are_explicit(tmp_path):
    s = spec()
    s["engine"] = "pressure-driven-wall-channel"
    with pytest.raises(ValueError):
        run_dynamic_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()
    s = spec()
    s["solver"]["absolute_tolerances"] = {"mechanical_work_j": 1e-8}
    assert (
        validate_dynamic_experiment(s)["solver"]["absolute_tolerances"]["mechanical_work_j"] == 1e-8
    )


@pytest.mark.parametrize(
    "field,budget",
    [
        ("energy_residual_j", "energy_absolute_j"),
        ("mechanical_residual_j", "mechanical_absolute_j"),
    ],
)
def test_absolute_ledgers_gate_acceptance_independently_of_relative_thermal_energy(
    monkeypatch, tmp_path, field, budget
):
    original = DynamicWallChannel.observation

    def faulty_observation(self):
        result = original(self)
        result[field] += 1.0
        return result

    monkeypatch.setattr(DynamicWallChannel, "observation", faulty_observation)
    r = run_dynamic_experiment(spec(), tmp_path / "run")
    assert r["status"] == "completed" and not r["numerical_acceptance"]
    assert not r["budget_checks"][budget]
    assert r["budget_checks"]["energy_relative"]


def test_failed_total_energy_step_restores_all_work_integrals(monkeypatch):
    cfg = equilibrated_total_config(config(), wall_offset_k=2.0)
    m = DynamicWallChannel(cfg)
    before = m.snapshot()
    original = m.derivatives

    def fail(t, y):
        result = original(t, y)
        if t > 0.001:
            raise DomainError("injected")
        return result

    monkeypatch.setattr(m, "derivatives", fail)
    with pytest.raises(DomainError):
        m.advance(0.02)
    assert m.snapshot() == before
    assert m.observation()["mechanical_residual_j"] == 0.0


def test_zero_flow_generalized_jacobian_obeys_total_energy_identity():
    c = DynamicWallChannel(config())
    y = c._state
    jac = c.analytic_jacobian(0.0, y)
    d = c.derivatives(0.0, y)
    g, hess = kinetic_gradient_hessian(c, y)
    for i in range(3):
        g[2 * i + 1] += 1.0
        g[10 + i] += c.config.total_heat_capacity_j_k / 3
    assert np.isfinite(jac.data).all()
    assert max(abs(jac.T @ g + hess @ d - jac[-1, :].toarray().ravel())) < 1e-6


@pytest.mark.parametrize("key", ["mechanical_absolute_j", "energy_absolute_j"])
def test_bundle_loader_recomputes_absolute_budgets(tmp_path, key):
    import hashlib

    from scripts.compare_isobaric_runs import load

    folder = tmp_path / "run"
    run_dynamic_experiment(spec(), folder)
    assert load(folder, 0.02)[1]["numerical_acceptance"]
    report = folder / "report.json"
    payload = json.loads(report.read_text())
    payload["balance"][key] += 1e-3
    report.write_text(json.dumps(payload))
    manifest = folder / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["artifacts"]["report.json"] = hashlib.sha256(report.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="balance"):
        load(folder, 0.02)


def test_total_energy_rejects_implicit_or_nonuniform_geometry():
    with pytest.raises(ValueError):
        replace(config(), junction_lengths_m=None)
    with pytest.raises(ValueError):
        replace(config(), junction_lengths_m=(1.0, 1.0, 1.0, 1.0))
    with pytest.raises(DomainError):
        equilibrated_total_config(config(left_pressure_pa=7e6, right_pressure_pa=7e6))
