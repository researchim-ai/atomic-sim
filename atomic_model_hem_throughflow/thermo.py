"""Pressure/enthalpy chart, sharing the frozen IF97 material coefficients."""

import numpy as np
from scipy.optimize import brentq

from atomic_model_hem_qualified import eos


def ph(pressure_pa, enthalpy_j_kg):
    p = eos.number(pressure_pa, "pressure")
    h = eos.number(enthalpy_j_kg, "enthalpy")
    f, g = eos.px(p, 0.0), eos.px(p, 1.0)
    if f.enthalpy_j_kg <= h <= g.enthalpy_j_kg:
        state = eos.px(p, (h - f.enthalpy_j_kg) / (g.enthalpy_j_kg - f.enthalpy_j_kg))
    else:
        region = 1 if h < f.enthalpy_j_kg else 2
        lo, hi = (eos.T_MIN, f.temperature_k) if region == 1 else (g.temperature_k, eos.T_MAX)

        def residual(t):
            return eos._raw(p, t, region)["h"] * 1000 - h

        t = brentq(residual, lo, hi, xtol=1e-10)
        state = eos._state(p, t, region)
    if abs(state.enthalpy_j_kg - h) > 2e-6:
        raise ValueError("p/h forward residual; no projection")
    return state


def checked_side(state, side):
    if state.region not in (1, 2, 4):
        raise ValueError("Unsupported derivative region")
    if state.region != 4:
        if side not in (None, "single"):
            raise ValueError("Wrong single-phase derivative side")
    elif side in ("liquid", "vapor"):
        if state.quality != (0.0 if side == "liquid" else 1.0):
            raise ValueError("Wrong endpoint derivative side")
    elif side not in (None, "mixture"):
        raise ValueError("Unknown derivative side")


def volume_derivatives(state, side=None):
    """v_p|h,v_h|p from the unchanged Gibbs/mixture Jacobian, SI."""
    checked_side(state, side)
    p = state.pressure_pa
    if state.region == 4 and side not in ("liquid", "vapor"):
        j, _ = eos.mixture_jacobian(p, state.quality)
    else:
        region = state.region if state.region != 4 else 1 if side == "liquid" else 2
        j = eos.single_jacobian(p, state.temperature_k, region)
    v = 1 / state.density_kg_m3
    hp = j[1, 0] + v + p * j[0, 0]
    hs = j[1, 1] + p * j[0, 1]
    if not np.isfinite(j).all() or hs <= 0:
        raise ValueError("p/h derivative domain")
    return float(j[0, 0] - j[0, 1] * hp / hs), float(j[0, 1] / hs)


def saturated_enthalpy_tangent(p, quality):
    s = eos.px(p, quality)
    j, _ = eos.mixture_jacobian(p, quality)
    return float(j[1, 0] + 1 / s.density_kg_m3 + p * j[0, 0])


def continued_ph(p, h, region):
    """Numerical branch continuation ONLY for locating an endpoint, never admission.

    Like Newton trial charts, a tiny metastable algebraic continuation is not an
    accepted physical state. The caller must finally admit it through strict ph.
    """
    f, g = eos.saturation(p)
    if region == 4:
        x = (h - f.enthalpy_j_kg) / (g.enthalpy_j_kg - f.enthalpy_j_kg)
        if not -0.2 <= x <= 1.2:
            raise ValueError("Phase event continuation bracket")
        v = (1 - x) / f.density_kg_m3 + x / g.density_kg_m3
        if v <= 0:
            raise ValueError("Phase event nonpositive volume")
        u = (1 - x) * f.internal_energy_j_kg + x * g.internal_energy_j_kg
        return eos.State(
            p,
            f.temperature_k,
            1 / v,
            u,
            h,
            (1 - x) * f.entropy_j_kg_k + x * g.entropy_j_kg_k,
            x,
            "numerical_trial",
            4,
        )
    lo, hi = (eos.T_MIN, 623.15) if region == 1 else (eos.T_MIN, eos.T_MAX)
    t = brentq(lambda tt: eos._raw(p, tt, region)["h"] * 1000 - h, lo, hi, xtol=1e-10)
    return eos._state(p, t, region)


def entropy_derivatives(state, side=None):
    """Differentiate actual IF97 entropy, exposing mixture Gibbs-fit inconsistency."""
    checked_side(state, side)
    p, t = state.pressure_pa, state.temperature_k
    v = 1 / state.density_kg_m3
    if state.region != 4 or side in ("liquid", "vapor"):
        return -v / t, 1 / t
    x = state.quality
    f, g = eos.saturation(p)
    j, dt = eos.mixture_jacobian(p, x)
    rawf = eos._raw(p, t, 1)
    rawg = eos._raw(p, t, 2)
    fp = -rawf["dv_dt"] + rawf["cp"] * 1000 / t * dt
    gp = -rawg["dv_dt"] + rawg["cp"] * 1000 / t * dt
    sp = (1 - x) * fp + x * gp
    sx = g.entropy_j_kg_k - f.entropy_j_kg_k
    hp = j[1, 0] + v + p * j[0, 0]
    hx = j[1, 1] + p * j[0, 1]
    return float(sp - sx * hp / hx), float(sx / hx)
