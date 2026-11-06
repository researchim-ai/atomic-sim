"""Independent analytic/dense checks and hostile input/restart cases."""

import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from atomic_model_spatial import Model, checkpoint, eigenvalue, fixed_source, restore, start, step
from atomic_model_spatial.__main__ import main
from atomic_model_spatial.cases import slab
from atomic_model_spatial.experiment import execute
from atomic_model_spatial.model import array
from atomic_model_spatial.reference import continuum_k, dense_k, dense_operators, exact_evolution
from atomic_model_spatial.transient import equilibrium_precursors
from atomic_model_spatial.workflow import audit, qualify
from atomic_model_workspace.common import digest


@pytest.mark.parametrize(
    "field",
    [
        "diffusion_m",
        "absorption_per_m",
        "nu_fission_per_m",
        "speed_m_s",
        "chi_prompt",
        "chi_delayed",
        "scatter_per_m",
        "beta",
        "decay_per_s",
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_reject_invalid_coefficients(field, bad):
    config = slab(4)
    value = np.array(config[field])
    value.flat[0] = bad
    config[field] = value.tolist()
    with pytest.raises(ValueError):
        Model(config)


@pytest.mark.parametrize(
    "change",
    [
        {"surprise": 1},
        {"faces_m": [0, 0, 1]},
        {"faces_m": [0, float("inf")]},
        {"boundaries": ["vacuum", "reflecting"]},
        {"boundaries": "reflecting"},
        {"provenance": ""},
        {"beta": [[0.8] * 4, [0.3] * 4]},
        {"chi_prompt": [[0.8] * 4, [0.1] * 4]},
        {"diffusion_m": [[0.0] * 4] * 2},
        {"speed_m_s": [[0.0] * 4] * 2},
        {"decay_per_s": [0, 0.3]},
        {"diffusion_m": [[0.1] * 4] * 9},
        {"beta": [[0.01] * 4] * 9, "decay_per_s": [0.1] * 9},
    ],
)
def test_strict_model_contract(change):
    config = slab(4)
    config.update(change)
    with pytest.raises(ValueError):
        Model(config)


def test_owned_immutable_configuration_and_diagonal_scatter_guard():
    config = slab(4)
    model = Model(config)
    config["diffusion_m"][0][0] = 100
    assert model.diffusion_m[0, 0] == 0.014
    with pytest.raises(ValueError):
        model.diffusion_m[0, 0] = 1
    with pytest.raises(AttributeError):
        model.width = np.ones(4)
    config = slab(4)
    config["scatter_per_m"][0][0][0] = 0.1
    with pytest.raises(ValueError):
        Model(config)
    with pytest.raises(ValueError):
        array(np.array([1 + 2j]), (1,), "complex")


@pytest.mark.parametrize(
    "boundaries",
    [("reflecting", "reflecting"), ("zero_flux", "reflecting"), ("zero_flux", "zero_flux")],
)
@pytest.mark.parametrize("cells", [1, 7, 16])
def test_independent_operators_and_nonuniform_mesh(boundaries, cells):
    config = slab(cells, True, boundaries)
    config["faces_m"] = (np.linspace(0, 1, cells + 1) ** 1.4).tolist()
    model = Model(config)
    loss, fiss, kinetics = dense_operators(model)
    np.testing.assert_allclose(model.loss.toarray(), loss, atol=1e-13)
    np.testing.assert_allclose(model.fission.toarray(), fiss, atol=1e-13)
    np.testing.assert_allclose(model.kinetics.toarray(), kinetics, atol=1e-10)
    result = eigenvalue(model)
    assert result["k"] == pytest.approx(dense_k(model), rel=1e-8)
    assert np.sum(model.fission @ np.ravel(result["flux"])) == pytest.approx(1, abs=1e-12)


def test_one_group_reflecting_analytic_source_and_k():
    config = slab(5, boundaries=("reflecting", "reflecting"))
    for key in ["diffusion_m", "absorption_per_m", "nu_fission_per_m", "speed_m_s"]:
        config[key] = [config[key][0]]
    config.update(chi_prompt=[[1] * 5], chi_delayed=[[1] * 5], scatter_per_m=[[[0] * 5]])
    model = Model(config)
    assert eigenvalue(model)["k"] == pytest.approx(0.035 / 0.09, rel=1e-12)
    result = fixed_source(model, [[0.11] * 5])
    np.testing.assert_allclose(result["flux"], [[2] * 5], rtol=1e-12)
    assert result["relative_balance_defect"] < 1e-12


def test_scatter_cancels_and_boundary_leakage_is_counted_once():
    model = Model(slab(9, True))
    phi = np.random.default_rng(33).uniform(0.1, 1, (2, 9))
    integrated_loss = np.sum(model.loss @ phi.ravel())
    absorption = np.sum(model.absorption_per_m * phi * model.width)
    leakage = np.sum(model.leakage * phi)
    assert integrated_loss == pytest.approx(absorption + leakage, rel=1e-12)
    assert leakage > 0


def test_supercritical_fixed_source_singular_and_iteration_failure():
    config = slab(4, boundaries=("reflecting", "reflecting"))
    config["nu_fission_per_m"] = (np.array(config["nu_fission_per_m"]) * 10).tolist()
    model = Model(config)
    assert eigenvalue(model)["k"] > 1
    with pytest.raises(ValueError):
        fixed_source(model, np.ones((2, 4)))
    with pytest.raises(ValueError):
        eigenvalue(Model(slab(12, True)), max_iterations=1)
    with pytest.raises(ValueError):
        eigenvalue(model, tolerance=float("nan"))
    config["nu_fission_per_m"] = np.zeros((2, 4)).tolist()
    config["absorption_per_m"] = np.zeros((2, 4)).tolist()
    config["scatter_per_m"] = np.zeros((2, 2, 4)).tolist()
    model = Model(config)
    with pytest.raises(ValueError):
        eigenvalue(model)
    with pytest.raises(ValueError):
        fixed_source(model, np.ones((2, 4)))


def test_k_continuum_and_absorber_monotonicity():
    config = slab(32)
    base = eigenvalue(Model(config))["k"]
    assert abs(base - continuum_k(Model(config))) < 0.001
    config["absorption_per_m"][1][10] *= 3
    assert eigenvalue(Model(config))["k"] < base


def test_critical_state_local_precursors_and_balance():
    config = slab(8, True, ("reflecting", "reflecting"))
    k = eigenvalue(Model(config))["k"]
    config["nu_fission_per_m"] = (np.array(config["nu_fission_per_m"]) / k).tolist()
    model = Model(config)
    phi = np.array(eigenvalue(model)["flux"])
    c = equilibrium_precursors(model, phi)
    assert np.ptp(c[0]) > 1e-6
    state = start(model, phi, c)
    for _ in range(20):
        state = step(model, state, 0.001, np.zeros_like(phi))
    np.testing.assert_allclose(state.flux, phi, rtol=2e-9)
    np.testing.assert_allclose(state.precursors, c, rtol=2e-9)
    assert max(r["relative_inventory_defect"] for r in state.history) < 1e-12


def test_independent_homogeneous_reflecting_local_kinetics_exponential():
    model = Model(slab(3, boundaries=("reflecting", "reflecting")))
    # Explicit 2-group/2-family ODE, independent of both spatial assemblers.
    g = np.array(
        [
            [-13 + 3.479, 0.8 + 9.94, 6.4, 24],
            [0.8, -2.56, 0.32, 1.2],
            [0.00014, 0.0004, -0.08, 0],
            [0.00007, 0.0002, 0, -0.3],
        ]
    )
    initial = np.array([1.0, 0.4, 0.01, 0.02])
    phi = np.tile(initial[:2, None], (1, 3))
    c = np.tile(initial[2:, None], (1, 3))
    state = start(model, phi, c)
    for _ in range(100):
        state = step(model, state, 0.0001, np.zeros_like(phi))
    expected = expm(g * 0.01) @ initial
    actual = np.concatenate((np.array(state.flux)[:, 0], np.array(state.precursors)[:, 0]))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf"), 100])
def test_rejected_step_keeps_original_state(dt):
    model = Model(slab(4))
    state = start(model, np.ones((2, 4)), np.ones((2, 4)))
    before = checkpoint(model, state)
    with pytest.raises(ValueError):
        step(model, state, dt, np.zeros((2, 4)))
    assert checkpoint(model, state) == before


def test_time_dependent_source_and_restart_full_history():
    model = Model(slab(6, True))
    state = start(model, np.ones((2, 6)), np.zeros((2, 6)))
    for i in range(10):
        state = step(model, state, 0.001, np.full((2, 6), i / 100))
    saved = json.loads(json.dumps(checkpoint(model, state)))
    resumed = restore(model, saved)
    a = step(model, state, 0.0005, np.ones((2, 6)))
    b = step(model, resumed, 0.0005, np.ones((2, 6)))
    assert checkpoint(model, a) == checkpoint(model, b)
    other = Model(slab(6))
    with pytest.raises(ValueError):
        restore(other, saved)
    with pytest.raises(ValueError):
        step(other, state, 0.001, np.ones((2, 6)))
    for key in ["flux", "history", "time_s", "source"]:
        corrupt = copy.deepcopy(saved)
        if key == "flux":
            corrupt["payload"]["flux"][0][0] *= 1.01
        elif key == "history":
            corrupt["payload"]["history"][0]["source"][0][0] = 0.4
        elif key == "time_s":
            corrupt["payload"]["time_s"] += 0.1
        else:
            corrupt["payload"]["source"]["runtime"]["numpy"] = "wrong"
        corrupt["sha256"] = digest(corrupt["payload"])
        with pytest.raises(ValueError):
            restore(model, corrupt)


def test_step_source_shape_negative_and_no_projection():
    model = Model(slab(4))
    state = start(model, np.ones((2, 4)), np.zeros((2, 4)))
    for source in [np.ones(4), -np.ones((2, 4)), np.full((2, 4), np.nan)]:
        with pytest.raises(ValueError):
            step(model, state, 0.001, source)
    with pytest.raises(ValueError):
        start(model, -np.ones((2, 4)), np.zeros((2, 4)))


def test_dense_transient_reference_and_precursor_decay_without_fission():
    config = slab(5)
    config["nu_fission_per_m"] = np.zeros((2, 5)).tolist()
    model = Model(config)
    phi = np.zeros((2, 5))
    c = np.ones((2, 5))
    state = start(model, phi, c)
    for _ in range(500):
        state = step(model, state, 0.00002, phi)
    expected = exact_evolution(model, phi, c, phi, 0.01)
    actual = np.concatenate((np.ravel(state.flux), np.ravel(state.precursors)))
    np.testing.assert_allclose(actual, expected, rtol=0.001)
    for j, decay in enumerate(model.decay):
        np.testing.assert_allclose(
            np.array(state.precursors)[j], (1 + 0.00002 * decay) ** -500, rtol=1e-12
        )


def test_requests_restart_cli_and_no_overwrite(tmp_path):
    config = slab(4)
    request = dict(schema_version=1, mode="eigen", model=config)
    result = execute(request)
    assert result["validated"] is False
    assert result["result"]["k"] > 0
    transient = dict(
        schema_version=1,
        mode="transient",
        model=config,
        initial_flux=np.ones((2, 4)).tolist(),
        initial_precursors=np.zeros((2, 4)).tolist(),
        steps=[dict(dt_s=0.001, source=np.ones((2, 4)).tolist())],
    )
    first = execute(transient)
    transient.update(initial_flux=None, initial_precursors=None)
    resumed = execute(transient, first["result"]["checkpoint"])
    assert resumed["result"]["time_s"] == 0.002
    bad = dict(request, unexpected=True)
    with pytest.raises(ValueError):
        execute(bad)
    with pytest.raises(ValueError):
        execute(request, first["result"]["checkpoint"])
    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    inp.write_text(json.dumps(request))
    assert main(["run", str(inp), "--output", str(out)]) == 0
    envelope = json.loads(out.read_text())
    assert digest(envelope["payload"]) == envelope["sha256"]
    with pytest.raises(SystemExit):
        main(["run", str(inp), "--output", str(out)])


def test_qualification_and_tamper_resistant_audit(tmp_path):
    result = qualify()
    assert all(min(case["orders"]) > 1.8 for case in result["spatial"])
    assert all(case["complete_history_restart_equal"] for case in result["transients"])
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(dict(payload=result, sha256=digest(result))))
    assert audit(path) == result
    result["validated"] = True
    path.write_text(json.dumps(dict(payload=result, sha256=digest(result))))
    with pytest.raises(ValueError):
        audit(path)


@pytest.mark.parametrize("groups,families", [(3, 6), (8, 2)])
def test_general_multigroup_and_local_family_dimensions(groups, families):
    n = 4
    config = slab(n, boundaries=("reflecting", "reflecting"))
    config.update(
        diffusion_m=np.full((groups, n), 0.01).tolist(),
        absorption_per_m=np.full((groups, n), 0.3).tolist(),
        nu_fission_per_m=np.full((groups, n), 0.005).tolist(),
        speed_m_s=np.full((groups, n), 20).tolist(),
        chi_prompt=np.full((groups, n), 1 / groups).tolist(),
        chi_delayed=np.full((groups, n), 1 / groups).tolist(),
        beta=np.full((families, n), 0.006 / families).tolist(),
        decay_per_s=np.linspace(0.08, 1, families).tolist(),
    )
    scatter = np.full((groups, groups, n), 0.01)
    scatter[np.arange(groups), np.arange(groups)] = 0
    config["scatter_per_m"] = scatter.tolist()
    model = Model(config)
    assert eigenvalue(model)["k"] == pytest.approx(0.005 / 0.3, rel=1e-12)
    phi = np.ones((groups, n))
    q = np.full_like(phi, 0.3 - 0.005)
    actual = fixed_source(model, q)
    np.testing.assert_allclose(actual["flux"], phi, rtol=1e-12)
    c = equilibrium_precursors(model, phi)
    state = step(model, start(model, phi, c), 0.001, q)
    np.testing.assert_allclose(state.flux, phi, rtol=1e-12)
    np.testing.assert_allclose(state.precursors, c, rtol=1e-12)


def test_current_divergence_boundary_signs_and_local_balance():
    from atomic_model_spatial.diagnostics import face_currents

    model = Model(slab(11, True))
    phi = np.random.default_rng(19).uniform(0.3, 1, (2, 11))
    currents = face_currents(model, phi)
    assert np.all(currents[:, 0] < 0) and np.all(currents[:, -1] > 0)
    removal = (model.absorption_per_m + model.scatter.sum(axis=0)) * phi * model.width
    incoming = np.einsum("ghn,hn->gn", model.scatter, phi) * model.width
    np.testing.assert_allclose(
        (model.loss @ phi.ravel()).reshape(2, 11),
        currents[:, 1:] - currents[:, :-1] + removal - incoming,
        atol=1e-13,
    )
    state = step(model, start(model, phi, np.ones_like(phi)), 0.001, np.ones_like(phi))
    assert state.history[-1]["max_local_inventory_relative_defect"] < 1e-12


@pytest.mark.parametrize("name", ["slab_eigen_v1", "slab_transient_v1"])
def test_packaged_examples_and_cli_resume(name, tmp_path):
    inp = tmp_path / "example.json"
    out = tmp_path / "run.json"
    assert main(["example", name, "--output", str(inp)]) == 0
    assert main(["run", str(inp), "--output", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["payload"]["result"]["face_currents"]
    if name == "slab_transient_v1":
        request = json.loads(inp.read_text())
        request.update(initial_flux=None, initial_precursors=None)
        request["steps"] = request["steps"][:1]
        resume = tmp_path / "resume.json"
        resume.write_text(json.dumps(request))
        destination = tmp_path / "resumed.json"
        assert (
            main(["run", str(resume), "--checkpoint", str(out), "--output", str(destination)]) == 0
        )
        assert json.loads(destination.read_text())["payload"]["result"]["time_s"] == pytest.approx(
            0.021
        )


def test_physical_neutron_speeds_do_not_change_critical_equilibrium():
    config = slab(6, True, ("reflecting", "reflecting"))
    config["speed_m_s"] = [[1e7] * 6, [2200.0] * 6]
    k = eigenvalue(Model(config))["k"]
    config["nu_fission_per_m"] = (np.array(config["nu_fission_per_m"]) / k).tolist()
    model = Model(config)
    phi = np.array(eigenvalue(model)["flux"])
    c = equilibrium_precursors(model, phi)
    state = start(model, phi, c)
    # The similarity certificate uses positive populations, without slowing neutrons.
    for _ in range(20):
        state = step(model, state, 0.001, np.zeros_like(phi))
    np.testing.assert_allclose(state.flux, phi, rtol=2e-8)
    np.testing.assert_allclose(state.precursors, c, rtol=2e-8)
