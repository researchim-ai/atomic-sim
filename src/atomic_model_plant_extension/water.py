# Adapted from the frozen atomic_model_hem_qualified/eos.py, Apache-2.0.
# Separate pressure admission matched to the lumped-plant16.5MPa chart ceiling.
"""Consistent forward charts and damped analytic Newton in physical SI coordinates.

IF97 coefficients remain in the unchanged material kernel. Numerical trial
charts may extend across saturation; accepted states never do. No stock clipping.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from atomic_model_hem.eos import State, number
from physics.materials import iapws

P_MIN, P_MAX, T_MIN, T_MAX = 1e5, 16.5e6, 300.0, 900.0
RHO_RTOL, U_ATOL = 2e-12, 2e-6
SOURCE = "https://iapws.org/relguide/IF97-Rev.html"


def _raw(p, t, region):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return (iapws._region1_raw if region == 1 else iapws._region2_raw)(t, p * 1e-6)


def _state(p, t, region):
    raw = _raw(p, t, region)
    return State(
        float(p),
        float(t),
        1 / float(raw["v"]),
        float(raw["u"]) * 1000,
        float(raw["h"]) * 1000,
        float(raw["s"]) * 1000,
        None,
        "liquid" if region == 1 else "vapor",
        region,
    )


def _domain(p, t):
    if not P_MIN <= p <= P_MAX or not T_MIN <= t <= T_MAX:
        raise ValueError("Plant-extension IF97 domain0.1..16.5MPa,300..900K")


def pt(pressure_pa, temperature_k):
    p = number(pressure_pa, "pressure")
    t = number(temperature_k, "temperature")
    _domain(p, t)
    ts = iapws.saturation_temperature_k(p * 1e-6)
    if abs(t - ts) < 1e-9:
        raise ValueError("Saturation p/T requires explicit quality")
    region = iapws.classify_region(t, p * 1e-6)
    if region not in (1, 2):
        raise ValueError("Only IF97 regions1/2/4 admitted")
    return _state(p, t, region)


@lru_cache(maxsize=4096)
def saturation(p):
    p = number(p, "saturation pressure")
    if not P_MIN <= p <= P_MAX:
        raise ValueError("Saturation pressure domain")
    t = float(iapws.saturation_temperature_k(p * 1e-6))
    return _state(p, t, 1), _state(p, t, 2)


def px(pressure_pa, quality):
    p = number(pressure_pa, "pressure")
    x = number(quality, "quality")
    if not 0 <= x <= 1:
        raise ValueError("Quality outside[0,1]")
    f, g = saturation(p)
    v = (1 - x) / f.density_kg_m3 + x / g.density_kg_m3
    u = (1 - x) * f.internal_energy_j_kg + x * g.internal_energy_j_kg
    return State(
        p,
        f.temperature_k,
        1 / v,
        u,
        u + p * v,
        (1 - x) * f.entropy_j_kg_k + x * g.entropy_j_kg_k,
        x,
        "two_phase" if 0 < x < 1 else "saturated_liquid" if x == 0 else "saturated_vapor",
        4,
    )


def single_jacobian(p, t, region):
    """Rows(v,u),columns(p,T); SI derivatives of Gibbs thermodynamics."""
    raw = _raw(p, t, region)
    v = raw["v"]
    vt = raw["dv_dt"]
    vp = raw["dv_dp_mpa"] / 1e6 if region == 1 else -raw["cp"] / (raw["cv"] * raw["w"] ** 2) * v * v
    return np.array([[vp, vt], [-t * vt - p * vp, raw["cp"] * 1000 - p * vt]])


def mixture_jacobian(p, x):
    f, g = saturation(p)
    vf, vg = 1 / f.density_kg_m3, 1 / g.density_kg_m3
    # Differentiate the actual IF97 region-4 curve, rather than assuming an
    # exactly Maxwell-consistent Clapeyron tangent between approximate charts.
    n = iapws._SAT_N
    tt = complex(f.temperature_k, 1e-20)
    theta = tt + n[8] / (tt - n[9])
    aa = theta**2 + n[0] * theta + n[1]
    bb = n[2] * theta**2 + n[3] * theta + n[4]
    cc = n[5] * theta**2 + n[6] * theta + n[7]
    ps = (2 * cc / (-bb + np.sqrt(bb**2 - 4 * aa * cc))) ** 4
    dt_dp = 1e-6 / (ps.imag / 1e-20)
    jf = single_jacobian(p, f.temperature_k, 1)
    jg = single_jacobian(p, g.temperature_k, 2)
    fp = jf[:, 0] + dt_dp * jf[:, 1]
    gp = jg[:, 0] + dt_dp * jg[:, 1]
    return np.column_stack(
        ((1 - x) * fp + x * gp, [vg - vf, g.internal_energy_j_kg - f.internal_energy_j_kg])
    ), dt_dp


def derivatives(state, side=None):
    p, t = state.pressure_pa, state.temperature_k
    if state.region == 4:
        if state.quality in (0.0, 1.0) and side is None:
            raise ValueError("Endpoint derivative requires explicit side")
        if side in ("liquid", "vapor"):
            if state.quality != (0.0 if side == "liquid" else 1.0):
                raise ValueError("Wrong endpoint side")
            j = single_jacobian(p, t, 1 if side == "liquid" else 2)
            transform = np.eye(2)
        else:
            if side not in (None, "mixture"):
                raise ValueError("Unknown derivative side")
            j, dt = mixture_jacobian(p, state.quality)
            transform = np.array([[1.0, 0.0], [dt, 0.0]])
    else:
        if side not in (None, "single"):
            raise ValueError("Single-phase derivative side")
        j = single_jacobian(p, t, state.region)
        transform = np.eye(2)
    inv = transform @ np.linalg.solve(j, np.array([[-1 / state.density_kg_m3**2, 0.0], [0.0, 1.0]]))
    c2 = float(inv[0, 0] + inv[0, 1] * p / state.density_kg_m3**2)
    if not np.isfinite(inv).all() or c2 <= 0:
        raise ValueError("Invalid thermodynamic Jacobian")
    return dict(
        pressure_wrt_density=float(inv[0, 0]),
        pressure_wrt_energy=float(inv[0, 1]),
        temperature_wrt_density=float(inv[1, 0]),
        temperature_wrt_energy=float(inv[1, 1]),
        equilibrium_sound_speed_squared_m2_s2=c2,
        side=side or "interior",
        method="Gibbs derivatives and exact IF97 region4 complex-step tangent; inverse-function theorem",
    )


def characteristic_speeds(state):
    if not P_MIN + 100 < state.pressure_pa < P_MAX - 100:
        raise ValueError("Pressure derivative margin")
    choices = [
        derivatives(state, "mixture" if state.region == 4 else None)[
            "equilibrium_sound_speed_squared_m2_s2"
        ]
        ** 0.5
    ]
    if state.region == 4 and state.quality < 1e-5:
        choices.append(
            derivatives(px(state.pressure_pa, 0.0), "liquid")[
                "equilibrium_sound_speed_squared_m2_s2"
            ]
            ** 0.5
        )
    if state.region == 4 and state.quality > 1 - 1e-5:
        choices.append(
            derivatives(px(state.pressure_pa, 1.0), "vapor")[
                "equilibrium_sound_speed_squared_m2_s2"
            ]
            ** 0.5
        )
    return min(choices), max(choices)


def capacity(state):
    values = [
        derivatives(state, "mixture" if state.region == 4 else None)["temperature_wrt_energy"]
    ]
    if state.region == 4 and state.quality in (0.0, 1.0):
        values.append(
            derivatives(state, "liquid" if state.quality == 0 else "vapor")[
                "temperature_wrt_energy"
            ]
        )
    if min(values) <= 0:
        raise ValueError("Nonpositive isochoric capacity")
    return 1 / max(values)


def _matches(state, rho, u, strict=False):
    return abs(state.density_kg_m3 / rho - 1) <= (5e-14 if strict else RHO_RTOL) and abs(
        state.internal_energy_j_kg - u
    ) <= (1e-7 if strict else U_ATOL)


def rho_u(density, internal_energy, pressure_seed_pa=1e6, temperature_seed_k=450.0):
    rho = number(density, "density")
    u = number(internal_energy, "energy")
    p = number(pressure_seed_pa, "pressure seed")
    t = number(temperature_seed_k, "temperature seed")
    if rho <= 0:
        raise ValueError("Nonpositive density")
    _domain(p, t)
    target = np.array([1 / rho, u])
    scale = np.array([1 / rho, max(abs(u), 1e5)])
    f, g = saturation(p)
    x = (target[0] - 1 / f.density_kg_m3) / (1 / g.density_kg_m3 - 1 / f.density_kg_m3)
    order = [4, 1, 2] if 0 <= x <= 1 else [1, 4, 2] if x < 0 else [2, 4, 1]

    def starts():
        for region in order:
            yield region, p, x if region == 4 else min(t, 623.15) if region == 1 else t
        # Global fallback seeds are physical forward energy brackets, never stock projection.
        for pp in [p, 8e6, 15.5e6, 2e5]:
            ff, gg = saturation(pp)
            xx = (target[0] - 1 / ff.density_kg_m3) / (1 / gg.density_kg_m3 - 1 / ff.density_kg_m3)
            yield 4, pp, xx if 0 <= xx <= 1 else 0.5
            for region, lo, hi in [
                (1, T_MIN, 623.15),
                (2, max(T_MIN, saturation(pp)[0].temperature_k - 60), T_MAX),
            ]:
                try:

                    def fn(tt):
                        return _raw(pp, tt, region)["u"] * 1000 - u

                    if fn(lo) * fn(hi) <= 0:
                        tt = brentq(fn, lo, hi, xtol=2e-11)
                        yield region, pp, tt
                except (ValueError, FloatingPointError):
                    pass

    for region, pp, second0 in starts():
        y = np.array([pp, second0])
        for _ in range(32):
            pressure, second = map(float, y)
            try:
                if region == 4:
                    a, b = saturation(pressure)
                    value = np.array(
                        [
                            (1 - second) / a.density_kg_m3 + second / b.density_kg_m3,
                            (1 - second) * a.internal_energy_j_kg + second * b.internal_energy_j_kg,
                        ]
                    )
                    j, _ = mixture_jacobian(pressure, second)
                else:
                    raw = _raw(pressure, second, region)
                    value = np.array([raw["v"], raw["u"] * 1000])
                    j = single_jacobian(pressure, second, region)
                residual = value - target
                norm = float(np.max(abs(residual / scale)))
                if abs(residual[0]) <= 5e-14 * target[0] and abs(residual[1]) <= 1e-7:
                    if region == 4:
                        if 0 <= second <= 1:
                            candidate = px(pressure, second)
                            if _matches(candidate, rho, u):
                                return candidate
                        for endpoint in (0.0, 1.0):
                            candidate = px(pressure, endpoint)
                            if _matches(candidate, rho, u, True):
                                return candidate
                    else:
                        ts = saturation(pressure)[0].temperature_k
                        if abs(second - ts) < 1e-8:
                            candidate = px(pressure, 0.0 if region == 1 else 1.0)
                            if _matches(candidate, rho, u, True):
                                return candidate
                        if (region == 1 and second <= ts) or (region == 2 and second >= ts):
                            if abs(second - ts) < 1e-9:
                                candidate = px(pressure, 0.0 if region == 1 else 1.0)
                                if _matches(candidate, rho, u, True):
                                    return candidate
                            candidate = _state(pressure, second, region)
                            if iapws.classify_region(
                                second, pressure * 1e-6
                            ) == region and _matches(candidate, rho, u):
                                return candidate
                    break
                step = np.linalg.solve(j, -residual)
                factor = min(1.0, 0.4 * pressure / max(abs(step[0]), 1e-30))
                if region != 4:
                    factor = min(factor, 30 / max(abs(step[1]), 1e-30))
                accepted = False
                for _ in range(18):
                    trial = y + factor * step
                    pp, ss = map(float, trial)
                    if P_MIN <= pp <= P_MAX and (
                        (-0.5 <= ss <= 1.5)
                        if region == 4
                        else T_MIN <= ss <= (623.15 if region == 1 else T_MAX)
                    ):
                        if region == 4:
                            a, b = saturation(pp)
                            probe = np.array(
                                [
                                    (1 - ss) / a.density_kg_m3 + ss / b.density_kg_m3,
                                    (1 - ss) * a.internal_energy_j_kg + ss * b.internal_energy_j_kg,
                                ]
                            )
                        else:
                            raw = _raw(pp, ss, region)
                            probe = np.array([raw["v"], raw["u"] * 1000])
                        if (
                            np.isfinite(probe).all()
                            and np.max(abs((probe - target) / scale)) < norm
                        ):
                            y = trial
                            accepted = True
                            break
                    factor *= 0.5
                if not accepted:
                    break
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                break
    # Explicit domain-face candidates are solutions only if BOTH conserved
    # coordinates satisfy the same strict forward residual, never clipped states.
    for boundary in (P_MIN, P_MAX):
        a, b = saturation(boundary)
        q = (1 / rho - 1 / a.density_kg_m3) / (1 / b.density_kg_m3 - 1 / a.density_kg_m3)
        if 0 <= q <= 1:
            candidate = px(boundary, q)
            if _matches(candidate, rho, u, True):
                return candidate
        ts = saturation(boundary)[0].temperature_k
        for region, lo, hi in ((1, T_MIN, ts), (2, ts, T_MAX)):
            try:

                def energy_residual(tt):
                    return _raw(boundary, tt, region)["u"] * 1000 - u

                if energy_residual(lo) * energy_residual(hi) > 0:
                    continue
                temperature = brentq(energy_residual, lo, hi, xtol=1e-11)
                candidate = (
                    px(boundary, 0.0 if region == 1 else 1.0)
                    if abs(temperature - ts) < 1e-9
                    else _state(boundary, temperature, region)
                )
                if _matches(candidate, rho, u, True):
                    return candidate
            except (ValueError, FloatingPointError):
                pass
    raise ValueError("Active-chart conserved inverse failed; no projection")
