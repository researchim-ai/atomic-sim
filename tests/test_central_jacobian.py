"""Производные, границы области, отмена и штатные интеграторы."""

import numpy as np
import pytest

from core.continuation import BDFContinuation
from core.jacobian import CentralJacobian
from core.solvers import integrate
from core.status import DomainError, SolverError
from sdk.api import create_session, export_snapshot
from sdk.experiment import validate_experiment


def test_scaled_nonlinear_jacobian_converges_and_restores_rhs_state():
    seen = []

    def rhs(t, y):
        seen.append(y.copy())
        return np.array([y[0] ** 3 + y[1] / 1e12, y[0] * y[1]])

    y = np.array([2.0, 3e12])
    exact = np.array([[12.0, 1e-12], [3e12, 2.0]])
    errors = []
    for step in (1e-2, 1e-3, 1e-4):
        jac = CentralJacobian(rhs, step)
        errors.append(np.max(abs((jac(0.0, y) - exact) / exact)))
        assert np.array_equal(seen[-1], y)
        assert jac.rhs_evaluations == 6
    assert errors[1] < errors[0] / 50
    assert errors[2] < errors[1] / 50


def test_domain_boundary_uses_only_valid_one_sided_derivative():
    def rhs(t, y):
        if y[0] < 0:
            raise DomainError("negative")
        return 3 * y

    assert CentralJacobian(rhs, 1e-5)(0, np.array([0.0])) == pytest.approx(np.array([[3.0]]))


def test_two_invalid_probes_fail_and_original_state_is_restored():
    seen = []

    def rhs(t, y):
        seen.append(y.copy())
        if y[0] != 0:
            raise DomainError("isolated point")
        return y

    with pytest.raises(DomainError, match="Обе пробы"):
        CentralJacobian(rhs, 1e-5)(0, np.array([0.0]))
    assert seen[-1][0] == 0.0


@pytest.mark.parametrize("error", [SolverError, RuntimeError])
def test_solver_and_program_errors_are_not_treated_as_domain_boundaries(error):
    def rhs(t, y):
        if y[0] != 1.0:
            raise error("injected")
        return -y

    with pytest.raises(error, match="injected"):
        CentralJacobian(rhs, 1e-5)(0, np.array([1.0]))


def test_both_integrators_match_analytic_solution_and_count_jacobian_rhs():
    for continuous in (False, True):
        calls = []

        def rhs(t, y):
            calls.append(t)
            return -np.array([1.0, 100.0]) * y

        options = dict(rtol=1e-8, atol=1e-10, max_step=0.1, jacobian_relative_step=1e-5)
        if continuous:
            solver = BDFContinuation(rhs, [1.0, 1.0], 0.0, 1.0, **options)
            a, b = solver.advance_to(0.3), solver.advance_to(1.0)
            assert a.n_rhs_evaluations + b.n_rhs_evaluations == len(calls)
        else:
            b = integrate(rhs, [1.0, 1.0], (0.0, 1.0), **options)
            assert b.n_rhs_evaluations == len(calls)
        assert b.final == pytest.approx([np.exp(-1), np.exp(-100)], abs=2e-8)


def test_plant_jacobian_policy_changes_restart_continuation_and_bad_policy_rolls_back():
    plant = create_session()
    plant.enable_continuation(1.0)
    plant.advance(0.1, jacobian_relative_step=1e-5)
    old = plant._continuation
    plant.advance(0.1, jacobian_relative_step=5e-6)
    assert plant._continuation is not old
    saved = export_snapshot(plant)
    with pytest.raises(SolverError):
        plant.advance(0.1, jacobian_relative_step=-1)
    assert export_snapshot(plant) == saved


@pytest.mark.parametrize("value", [True, 0.0, float("nan"), 0.1])
def test_experiment_rejects_invalid_jacobian_step(value):
    with pytest.raises(ValueError):
        validate_experiment(
            dict(
                name="bad",
                schema_version=1,
                engine="vver",
                duration_s=1,
                solver={"jacobian_relative_step": value},
            )
        )
