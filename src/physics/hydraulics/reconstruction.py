"""Conservative donor/MUSCL-MC face states on uniform finite-volume grids.

The operator reconstructs a specific transported quantity, not an EOS state.
Thermodynamic applicability must be checked by the caller. MC is a spatial
limiter; implicit time integration is not thereby certified TVD.
"""

from __future__ import annotations

import numpy as np


def mc_slopes(values, left_ghost, right_ghost):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("A finite nonempty cell vector is required")
    extended = np.r_[left_ghost, values, right_ghost]
    if not np.all(np.isfinite(extended)):
        raise ValueError("Ghost values must be finite")
    dl = extended[1:-1] - extended[:-2]
    dr = extended[2:] - extended[1:-1]
    if not np.isfinite(dl).all() or not np.isfinite(dr).all():
        raise ValueError("Transport differences exceed numeric range")
    central = 0.5 * dl + 0.5 * dr
    magnitude = 2 * np.minimum(np.minimum(abs(dl), abs(dr)), 0.5 * abs(central))
    same_sign = (np.signbit(dl) == np.signbit(dr)) & (dl != 0) & (dr != 0)
    return np.where(same_sign, np.sign(dl) * magnitude, 0.0)


def face_values(
    values, flows, left_value=None, right_value=None, *, scheme="donor", periodic=False
):
    """Return one shared donor/reconstructed value per oriented face.

    Reservoir values are imposed only on inflow; outflow uses one-sided
    linear extrapolation for MUSCL. Internal face states remain bounded by
    their adjacent cell values. Periodic endpoint fluxes must be identical.
    """
    values = np.asarray(values, dtype=float)
    flows = np.asarray(flows, dtype=float)
    if values.ndim != 1 or not values.size or flows.shape != (values.size + 1,):
        raise ValueError("Expected n cell values and n+1 face flows")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(flows)):
        raise ValueError("Nonfinite transport input")
    if scheme not in ("donor", "muscl_mc"):
        raise ValueError("Unknown transport reconstruction")
    n = values.size
    if periodic:
        if flows[0] != flows[-1]:
            raise ValueError("Periodic boundary flows must agree")
        lg, rg = values[-1], values[0]
    else:
        if (
            left_value is None
            or right_value is None
            or not np.isfinite([left_value, right_value]).all()
        ):
            raise ValueError("Finite reservoir values are required")
        lg = (
            2 * left_value - values[0]
            if flows[0] >= 0
            else (2 * values[0] - values[1] if n > 1 else values[0])
        )
        rg = (
            2 * right_value - values[-1]
            if flows[-1] < 0
            else (2 * values[-1] - values[-2] if n > 1 else values[-1])
        )
    slopes = np.zeros(n) if scheme == "donor" else mc_slopes(values, lg, rg)
    low, high = values - 0.5 * slopes, values + 0.5 * slopes
    result = np.empty(n + 1)
    result[1:n] = np.where(flows[1:n] >= 0, high[:-1], low[1:])
    if periodic:
        result[0] = high[-1] if flows[0] >= 0 else low[0]
        result[-1] = result[0]
    else:
        result[0] = left_value if flows[0] >= 0 else low[0]
        result[-1] = high[-1] if flows[-1] >= 0 else right_value
    return result


def conservative_rate(
    values, flows, *, scheme="donor", periodic=False, left_value=None, right_value=None
):
    """Net extensive influx; adjacent cells use the exact same face flux."""
    face = face_values(values, flows, left_value, right_value, scheme=scheme, periodic=periodic)
    flux = np.asarray(flows) * face
    return flux[:-1] - flux[1:]


def face_value_jacobian(
    values, flows, left_value=None, right_value=None, *, scheme="donor", periodic=False
):
    """Face values and piecewise analytic derivative w.r.t. cell values.

    Flow orientation and the active MC branch are fixed locally. At a tie or
    zero slope one deterministic generalized derivative is selected; a unique
    classical derivative is not claimed at those nonsmooth points.
    """
    face = face_values(values, flows, left_value, right_value, scheme=scheme, periodic=periodic)
    q = np.asarray(values, dtype=float)
    f = np.asarray(flows, dtype=float)
    n = len(q)
    eye = np.eye(n)
    if periodic:
        lg, rg = q[-1], q[0]
        gl, gr = eye[-1], eye[0]
    else:
        if f[0] >= 0:
            lg = 2 * left_value - q[0]
            gl = -eye[0]
        elif n > 1:
            lg = 2 * q[0] - q[1]
            gl = 2 * eye[0] - eye[1]
        else:
            lg = q[0]
            gl = eye[0]
        if f[-1] < 0:
            rg = 2 * right_value - q[-1]
            gr = -eye[-1]
        elif n > 1:
            rg = 2 * q[-1] - q[-2]
            gr = 2 * eye[-1] - eye[-2]
        else:
            rg = q[-1]
            gr = eye[-1]
    ext = np.r_[lg, q, rg]
    grad = np.vstack([gl, eye, gr])
    sgrad = np.zeros((n, n))
    if scheme == "muscl_mc":
        for i in range(n):
            dl = ext[i + 1] - ext[i]
            dr = ext[i + 2] - ext[i + 1]
            if dl == 0 or dr == 0 or np.signbit(dl) != np.signbit(dr):
                continue
            gdl = grad[i + 1] - grad[i]
            gdr = grad[i + 2] - grad[i + 1]
            branch = int(np.argmin([abs(2 * dl), abs(0.5 * dl + 0.5 * dr), abs(2 * dr)]))
            sgrad[i] = (2 * gdl, 0.5 * gdl + 0.5 * gdr, 2 * gdr)[branch]
    low, high = eye - 0.5 * sgrad, eye + 0.5 * sgrad
    jac = np.zeros((n + 1, n))
    for j in range(1, n):
        jac[j] = high[j - 1] if f[j] >= 0 else low[j]
    if periodic:
        jac[0] = high[-1] if f[0] >= 0 else low[0]
        jac[-1] = jac[0]
    else:
        if f[0] < 0:
            jac[0] = low[0]
        if f[-1] >= 0:
            jac[-1] = high[-1]
    return face, jac
