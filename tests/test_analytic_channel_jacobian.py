from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from core.continuation import ImplicitContinuation
from core.solvers import integrate
from core.status import SolverError
from physics.hydraulics.dynamic_channel import DynamicWallChannel, _liquid_state
from physics.hydraulics.liquid_derivatives import conserved_derivatives
from physics.hydraulics.reconstruction import face_value_jacobian, face_values
from physics.materials.iapws import properties_pt
from scripts.verify_pipe_mesh import configuration


@pytest.mark.parametrize("p,t", [(1e6, 350.0), (7e6, 500.0), (15.5e6, 580.0), (30e6, 620.0)])
def test_conserved_thermo_derivatives_match_independent_inverse_perturbations(p, t):
    state = properties_pt(p, t)
    v = 0.002
    m = state.rho_kg_m3 * v
    u = m * state.u_j_kg
    derivative = conserved_derivatives(state, m, u, v)
    for col, step in ((0, m * 1e-6), (1, u * 1e-6)):
        pert = []
        for sign in (-1, 1):
            mm = m + sign * step if col == 0 else m
            uu = u + sign * step if col == 1 else u
            s = _liquid_state(mm / v, uu / mm, p, t)
            pert.append(np.array([s.p_pa, s.T_k, s.h_j_kg, s.rho_kg_m3]))
        numeric = (pert[1] - pert[0]) / (2 * step)
        exact = np.array(
            [
                derivative.pressure[col],
                derivative.temperature[col],
                derivative.enthalpy[col],
                derivative.density[col],
            ]
        )
        assert exact == pytest.approx(numeric, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("scheme", ["donor", "muscl_mc"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_face_branch_jacobian_matches_finite_differences(scheme, sign):
    q = np.array([1.0, 2.5, 5.5, 9.5])
    flows = np.full(5, sign)
    _, jac = face_value_jacobian(q, flows, 0.0, 11.0, scheme=scheme)
    for i in range(4):
        perturb = np.zeros(4)
        perturb[i] = 1e-6
        numeric = (
            face_values(q + perturb, flows, 0.0, 11.0, scheme=scheme)
            - face_values(q - perturb, flows, 0.0, 11.0, scheme=scheme)
        ) / (2e-6)
        assert jac[:, i] == pytest.approx(numeric, abs=1e-8)


@pytest.mark.parametrize("scheme", ["donor", "muscl_mc"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sparse_coupled_jacobian_directional_derivatives_and_energy_identity(scheme, sign):
    cfg = replace(
        configuration(4, 0.0),
        enthalpy_transport=scheme,
        initial_flow_kg_s=sign,
        initial_temperature_profile_k=(496.0, 498.0, 501.0, 505.0),
        initial_wall_temperature_profile_k=(510.0,) * 4,
    )
    c = DynamicWallChannel(cfg)
    y = c._state.copy()
    jac = c.analytic_jacobian(0.0, y)
    assert isinstance(jac, csc_matrix) and jac.nnz < 0.5 * y.size**2
    assert np.count_nonzero(jac[:, -2:].toarray()) == 0
    direction = np.random.default_rng(71).normal(size=y.size) * np.maximum(abs(y), 1.0)
    eps = 1e-6
    numeric = (
        c.derivatives(0.0, y + eps * direction) - c.derivatives(0.0, y - eps * direction)
    ) / (2 * eps)
    analytic = jac @ direction
    assert analytic == pytest.approx(numeric, rel=5e-5, abs=0.01)
    # Global fluid+wall energy derivative equals boundary integral derivative.
    row = sum(jac[2 * i + 1, :] for i in range(4))
    row += sum(cfg.total_heat_capacity_j_k / 4 * jac[13 + i, :] for i in range(4))
    row -= jac[-1, :]
    assert max(abs(row.toarray().ravel())) < 1e-6
    mass = sum(jac[2 * i, :] for i in range(4)) - jac[-2, :]
    assert max(abs(mass.toarray().ravel())) < 1e-12


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_core_callable_sparse_jacobian_and_continuation(method):
    matrix = csc_matrix([[-3.0, 0.0], [2.0, -1.0]])

    def rhs(t, y):
        return matrix @ y

    def callback(t, y):
        return matrix

    solution = integrate(
        rhs,
        np.array([1.0, 0.0]),
        (0.0, 0.1),
        method=method,
        jacobian=callback,
        rtol=1e-10,
        atol=1e-12,
    )
    expected = np.array([np.exp(-0.3), np.exp(-0.1) - np.exp(-0.3)])
    assert solution.final == pytest.approx(expected, abs=1e-9)
    continuation = ImplicitContinuation(
        rhs,
        [1.0, 0.0],
        0.0,
        0.1,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01,
        jacobian=callback,
        method=method,
        initial_target=0.05,
    )
    continuation.advance_to(0.05)
    assert continuation.advance_to(0.1).final == pytest.approx(expected, abs=1e-9)
    with pytest.raises(SolverError):
        integrate(
            rhs, np.array([1.0, 0.0]), (0.0, 0.1), jacobian=callback, jacobian_relative_step=1e-5
        )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_periodic_face_jacobian_shares_endpoint_derivative(sign):
    q = np.array([1.0, 3.0, 7.0, 8.0])
    flows = np.full(5, sign)
    _, jac = face_value_jacobian(q, flows, scheme="muscl_mc", periodic=True)
    assert np.array_equal(jac[0], jac[-1])
    for i in range(4):
        perturb = np.zeros(4)
        perturb[i] = 1e-6
        fd = (
            face_values(q + perturb, flows, scheme="muscl_mc", periodic=True)
            - face_values(q - perturb, flows, scheme="muscl_mc", periodic=True)
        ) / (2e-6)
        assert jac[:, i] == pytest.approx(fd, abs=1e-8)


def test_flat_state_generalized_jacobian_is_finite_without_mutating_snapshot():
    m = DynamicWallChannel(replace(configuration(4, 0.0), enthalpy_transport="muscl_mc"))
    before = m.snapshot()
    jac = m.analytic_jacobian(0.0, m._state)
    assert np.isfinite(jac.data).all() and m.snapshot() == before
    assert jac[:, -2:].nnz == 0
