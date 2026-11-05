"""Nonnegative two-gain feasible set with primal/dual/ray certificates, no fit."""

import numpy as np
from scipy.optimize import linprog

from atomic_model_workspace.common import keys, number

TOL = 2e-8
OPTIONS = dict(
    maxiter=10000,
    time_limit=5.0,
    primal_feasibility_tolerance=1e-9,
    dual_feasibility_tolerance=1e-9,
)


def constraints(observations):
    if not isinstance(observations, list) or not 1 <= len(observations) <= 512:
        raise ValueError("Observation budget1..512")
    A = []
    b = []
    seen = set()
    for row in observations:
        keys(row, ("id", "lower_basis", "upper_basis", "observed_q_bounds"))
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Unique observation identifier")
        seen.add(row["id"])
        parsed = []
        for field in ("lower_basis", "upper_basis", "observed_q_bounds"):
            pair = row[field]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Expected pair")
            parsed.append([number(v, 0, 1e12, field) for v in pair])
        low, high, (qlo, qhi) = parsed
        if qlo > qhi or np.any(np.array(low) > high):
            raise ValueError("Reversed measurement/response interval")
        for normal, right in [(low, qhi), ([-v for v in high], -qlo)]:
            scale = max(1.0, abs(right), max(map(abs, normal)))
            A.append([v / scale for v in normal])
            b.append(right / scale)
    return np.array(A), np.array(b)


def check_primal(A, b, x):
    return bool(np.isfinite(x).all() and np.min(x) >= -TOL and np.max(A @ x - b) <= TOL)


def certificate(result, c, A, b):
    x = result.x
    y = result.ineqlin.marginals
    z = result.lower.marginals
    residual = float(np.max(abs(c - A.T @ y - z)))
    gap = float(c @ x - b @ y)
    if (
        not check_primal(A, b, x)
        or np.max(y) > TOL
        or np.min(z) < -TOL
        or residual > TOL
        or abs(gap) > TOL * (1 + abs(c @ x))
    ):
        raise ValueError("LP primal/dual certificate failed")
    return dict(
        point=x.tolist(),
        dual_inequalities=y.tolist(),
        dual_nonnegative_bounds=z.tolist(),
        objective=float(c @ x),
        stationarity_residual=residual,
        duality_gap=gap,
        primal_residual=float(max(0.0, np.max(A @ x - b))),
    )


def solve(observations):
    A, b = constraints(observations)
    feasible = linprog(
        [0.0, 0.0],
        A_ub=A,
        b_ub=b,
        bounds=[(0, None), (0, None)],
        method="highs-ds",
        options=OPTIONS,
    )
    if feasible.status == 2:
        return dict(status="infeasible", bounds=None, certificates=[], physical_acceptance=False)
    if feasible.status != 0 or not check_primal(A, b, feasible.x):
        raise ValueError("Feasibility solve failed or exhausted budget")
    certs = []
    bounds = []
    for j in (0, 1):
        objective = np.eye(2)[j]
        lower = linprog(
            objective,
            A_ub=A,
            b_ub=b,
            bounds=[(0, None), (0, None)],
            method="highs-ds",
            options=OPTIONS,
        )
        if lower.status != 0:
            raise ValueError("Lower parameter bound unresolved")
        lowcert = certificate(lower, objective, A, b)
        certs.append(lowcert)
        upper = linprog(
            -objective,
            A_ub=A,
            b_ub=b,
            bounds=[(0, None), (0, None)],
            method="highs-ds",
            options=OPTIONS,
        )
        if upper.status == 3:
            # For our nonnegative monotone basis, this exact coordinate ray
            # proves unboundedness; a solver status alone is never sufficient.
            ray = objective
            if np.any(A @ ray > 0):
                raise ValueError("Unbounded result has no monotone recession certificate")
            high = None
            certs.append(
                dict(
                    recession_ray=ray.tolist(),
                    maximum_constraint_direction=float(np.max(A @ ray)),
                    objective_direction=-1.0,
                )
            )
        elif upper.status == 0:
            highcert = certificate(upper, -objective, A, b)
            certs.append(highcert)
            high = float(upper.x[j])
        else:
            raise ValueError("Upper parameter bound unresolved")
        bounds.append(dict(lower=float(lower.x[j]), upper=high))
    return dict(
        status="unbounded" if any(r["upper"] is None for r in bounds) else "bounded",
        bounds=bounds,
        certificates=certs,
        feasibility_witness=feasible.x.tolist(),
        baseline_feasible=check_primal(A, b, np.ones(2)),
        physical_acceptance=False,
        numerical_certificate_tolerance=TOL,
        interpretation="Conditional diagnostic gains; witnesses are feasibility certificates, not calibrated parameters",
    )
