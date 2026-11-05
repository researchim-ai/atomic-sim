"""Diagnostic convective/nucleate gains, separate from frozen production physics."""

import math

import numpy as np

from atomic_model_hem_qualified import eos
from atomic_model_hem_throughflow import closures
from atomic_model_workspace.common import number
from physics.materials import iapws


def terms(p, x, G, D, wall_temperature_k):
    for value, label, lo, hi in [
        (p, "pressure", 15e6, 16e6),
        (x, "quality", 0, 0.1),
        (G, "mass flux", 300, 2000),
        (D, "diameter", 0.005, 0.025),
    ]:
        number(value, lo, hi, label)
    ts, C, N = closures.coefficients(p, x, G, D)
    tw = number(wall_temperature_k, ts, ts + 10, "diagnostic wall temperature")
    delta = tw - ts
    if delta == 0:
        return np.zeros(2)
    dp = float(iapws.saturation_pressure_mpa(tw) * 1e6 - p)
    if dp <= 0:
        raise ValueError("Saturation pressure resolution")
    return np.array([C * delta, N * delta**1.24 * dp**0.75])


def pressure_tangent(t):
    n = iapws._SAT_N
    tt = complex(t, 1e-20)
    theta = tt + n[8] / (tt - n[9])
    a = theta**2 + n[0] * theta + n[1]
    b = n[2] * theta**2 + n[3] * theta + n[4]
    c = n[5] * theta**2 + n[6] * theta + n[7]
    return float(((2 * c / (-b + np.sqrt(b * b - 4 * a * c))) ** 4).imag / 1e-20 * 1e6)


def predict(p, x, G, D, q, L, gains=(1.0, 1.0)):
    a, b = [number(v, 0, 1e12, "diagnostic gain") for v in gains]
    if a == b == 0:
        raise ValueError("Zero gains cannot supply positive heat")
    scope = {k: True for k in closures.FLAGS}
    critical = closures.critical_heat_flux(p, G, x, D, L, scope=scope)
    q = number(q, 1e-9, 0.8 * critical["heat_flux_w_m2"], "heat flux")
    if q >= 0.8 * critical["heat_flux_w_m2"]:
        raise ValueError("Strict pre-CHF guard")
    ts, C, N = closures.coefficients(p, x, G, D)

    def residual(t):
        return float(np.dot([a, b], terms(p, x, G, D, t))) - q

    left, right = ts, ts + 10
    if residual(right) < 0:
        raise ValueError("Required wall temperature outside supported branch")
    for _ in range(80):
        middle = (left + right) / 2
        if middle in (left, right):
            break
        if residual(middle) < 0:
            left = middle
        else:
            right = middle
        if right - left <= 1e-9 and abs(residual((left + right) / 2)) <= 0.001:
            break
    tw = (left + right) / 2
    if abs(residual(tw)) > 0.001:
        raise ValueError("Diagnostic root underresolved; no trusted sensitivity")
    basis = terms(p, x, G, D, tw)
    delta = tw - ts
    dp = float(iapws.saturation_pressure_mpa(tw) * 1e6 - p)
    derivative = a * C + b * basis[1] / delta * (1.24 + 0.75 * delta * pressure_tangent(tw) / dp)
    if not math.isfinite(derivative) or derivative <= 0:
        raise ValueError("Diagnostic monotone derivative")
    return dict(
        wall_temperature_k=tw,
        numerical_root_bracket_k=[left, right],
        root_bracket_width_k=right - left,
        heat_flux_residual_w_m2=residual(tw),
        sensitivity_k_per_gain=(-basis / derivative).tolist(),
        logarithmic_sensitivity_k=(-basis * np.array([a, b]) / derivative).tolist(),
        diagnostic_only=True,
        physical_acceptance=False,
    )


def envelope(row):
    p, x, G, D, q = [
        row[k]
        for k in ("pressure_pa", "quality", "mass_flux_kg_m2_s", "diameter_m", "heat_flux_w_m2")
    ]
    L = row["length_over_diameter"] * D
    chf = closures.critical_heat_flux(p, G, x, D, L, scope={k: True for k in closures.FLAGS})
    qlo, qhi = row["heat_flux_w_m2_lower"], row["heat_flux_w_m2_upper"]
    if not 0 < qlo <= q <= qhi < 0.8 * chf["heat_flux_w_m2"]:
        raise ValueError("Measured heat interval leaves guard")
    ts = eos.px(p, 0.0).temperature_k
    # Intersection of OBSERVATIONAL interval with declared model domain;
    # physical state/measurements are never overwritten or projected.
    lo = max(row["wall_temperature_k_lower"], ts)
    hi = min(row["wall_temperature_k_upper"], ts + 10.0)
    if lo > hi:
        raise ValueError("Observed wall interval has no supported intersection")
    return dict(
        id=row["row_id"],
        lower_basis=terms(p, x, G, D, lo).tolist(),
        upper_basis=terms(p, x, G, D, hi).tolist(),
        observed_q_bounds=[qlo, qhi],
        measurement_wall_interval_k=[
            row["wall_temperature_k_lower"],
            row["wall_temperature_k_upper"],
        ],
        admitted_wall_interval_k=[lo, hi],
        intersection_truncated=lo != row["wall_temperature_k_lower"]
        or hi != row["wall_temperature_k_upper"],
    )
