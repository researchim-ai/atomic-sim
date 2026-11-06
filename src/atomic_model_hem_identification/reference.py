"""Independent 2D active-line intersection oracle; no LP solver invocation."""

import numpy as np


def enumerate_region(A, b):
    A = np.vstack([A, -np.eye(2)])
    b = np.r_[b, 0.0, 0.0]
    if len(b) > 1026:
        raise ValueError("Reference constraint budget")
    vertices = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            a, c = A[i]
            d, e = A[j]
            det = a * e - c * d
            if abs(det) < 1e-14:
                continue
            point = np.array([(b[i] * e - c * b[j]) / det, (a * b[j] - b[i] * d) / det])
            if np.isfinite(point).all() and np.max(A @ point - b) <= 2e-8:
                vertices.append(point)
    if not vertices:
        return dict(status="infeasible", bounds=None, vertices=0)
    values = np.array(vertices)
    bounds = [
        dict(
            lower=float(np.min(values[:, j])),
            upper=None if np.all(A[:, j] <= 0) else float(np.max(values[:, j])),
        )
        for j in (0, 1)
    ]
    return dict(
        status="unbounded" if any(x["upper"] is None for x in bounds) else "bounded",
        bounds=bounds,
        vertices=len(vertices),
    )


def monotone_cone_certificate(observations):
    """Construct a feasible point/rays directly from RAW response boxes, no LP.

    When all lower-temperature bases are zero, increasing either nonnegative
    gain preserves every upper observation constraint. A sufficiently large
    common gain satisfies all lower heat constraints. This is a data-conditional
    unboundedness certificate, not a proposed parameter fit.
    """
    if any(any(x != 0 for x in r["lower_basis"]) for r in observations):
        return None
    if any(sum(r["upper_basis"]) <= 0 for r in observations):
        return None
    gain = 1 + max(r["observed_q_bounds"][0] / sum(r["upper_basis"]) for r in observations)
    margins = [gain * sum(r["upper_basis"]) - r["observed_q_bounds"][0] for r in observations]
    if min(margins) <= 0:
        raise ValueError("Constructive cone witness failed")
    return dict(
        feasible_witness=[gain, gain],
        nonnegative_recession_rays=[[1.0, 0.0], [0.0, 1.0]],
        minimum_lower_constraint_margin_w_m2=min(margins),
        upper_response_at_lower_temperature_w_m2=0.0,
        meaning="Feasibility witness and coordinate rays, not calibrated gains",
    )
