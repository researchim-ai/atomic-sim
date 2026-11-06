"""Named Chen-INL variant with IAEA1995 tube K1/K4 corrections, not full LUT."""

from functools import lru_cache
from math import exp, sqrt

import numpy as np
from scipy.optimize import brentq

from atomic_model_hem_qualified import closures as prior
from atomic_model_hem_qualified import eos
from atomic_model_workspace.common import keys
from physics.materials import iapws

VERSION = "chen-inl-hfg025-iaea1995-k1k4/v1"
SOURCE = prior.CHF_SOURCE
FLAGS = (
    "pure_water",
    "round_tube",
    "vertical_upflow",
    "uniform_axial_heating",
    "uniform_circumferential_heating",
    "no_spacers",
)
G_AXIS = (300.0, 500.0, 1000.0, 1500.0, 2000.0)
# PDF189, printed181, TableII.I. Axes p15/16MPa,G_AXIS,x0/.05/.1; kW/m².
TABLE = np.array(
    [
        [
            [3274.0, 3111.0, 2884.0],
            [3115.0, 3003.0, 2739.0],
            [2468.0, 2251.0, 1897.0],
            [2312.0, 1955.0, 1649.0],
            [2464.0, 2071.0, 1843.0],
        ],
        [
            [2944.0, 2818.0, 2597.0],
            [2673.0, 2545.0, 2268.0],
            [2089.0, 1853.0, 1528.0],
            [1945.0, 1655.0, 1419.0],
            [2166.0, 1841.0, 1688.0],
        ],
    ]
)
TABLE.setflags(write=False)


def geometry(p, G, x, D, L, scope):
    keys(scope, FLAGS)
    if any(scope[k] is not True for k in FLAGS):
        raise ValueError("Straight uniform vertical tube scope required")
    for val, name in [
        (p, "pressure"),
        (G, "mass flux"),
        (x, "quality"),
        (D, "diameter"),
        (L, "heated distance"),
    ]:
        eos.number(val, name)
    if not (
        15e6 <= p <= 16e6
        and 300 <= G <= 2000
        and 0 <= x <= 0.1
        and 0.005 <= D <= 0.025
        and 0.2 <= L <= 10
        and L / D >= 5
    ):
        raise ValueError("Bounded15..16MPa tube CHF/Chen domain; no extrapolation")


def critical_heat_flux(p, G, x, D, L, *, scope):
    geometry(p, G, x, D, L, scope)
    i = min(int(np.searchsorted(G_AXIS, G, side="right")) - 1, len(G_AXIS) - 2)
    j = 0 if x <= 0.05 else 1
    weights = (
        (p - 15e6) / 1e6,
        (G - G_AXIS[i]) / (G_AXIS[i + 1] - G_AXIS[i]),
        (x - 0.05 * j) / 0.05,
    )
    q = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                q += TABLE[a, i + b, j + c] * np.prod(
                    [w if k else 1 - w for k, w in zip((a, b, c), weights)]
                )
    liquid, vapor = prior.transport(p)
    alpha = x * liquid.rho_kg_m3 / (x * liquid.rho_kg_m3 + (1 - x) * vapor.rho_kg_m3)
    k1 = sqrt(0.008 / D)
    k4 = exp(D / L * exp(2 * alpha))
    return dict(
        heat_flux_w_m2=float(q * 1000 * k1 * k4),
        nominal_8mm_w_m2=float(q * 1000),
        diameter_factor=k1,
        heated_length_factor=k4,
        version=VERSION,
        source=SOURCE,
        uncertainty="Uncalibrated; K1/K4 are source approximations, no assigned confidence interval",
    )


@lru_cache(maxsize=2048)
def coefficients(p, x, G, D):
    f, g = prior.transport(p)
    re = G * (1 - x) * D / f.viscosity_pa_s
    pr = f.cp_j_kg_k * f.viscosity_pa_s / f.conductivity_w_m_k
    if not 1e4 <= re <= 5e6 or not 0.7 <= pr <= 100:
        raise ValueError("Chen liquid Re/Pr envelope")
    inv = (
        0.0
        if x == 0
        else (x / (1 - x)) ** 0.9
        * (f.rho_kg_m3 / g.rho_kg_m3) ** 0.5
        * (g.viscosity_pa_s / f.viscosity_pa_s) ** 0.1
    )
    enhancement = 1.0 if inv <= 0.1 else 2.35 * (0.213 + inv) ** 0.736
    suppression = 1 / (1 + 2.53e-6 * (re * enhancement**1.25) ** 1.17)
    conv = 0.023 * re**0.8 * pr**0.4 * f.conductivity_w_m_k / D
    nb = (
        0.00122
        * f.conductivity_w_m_k**0.79
        * f.cp_j_kg_k**0.45
        * f.rho_kg_m3**0.49
        / (
            prior.surface_tension(float(f.T_k)) ** 0.5
            * f.viscosity_pa_s**0.29
            * (g.h_j_kg - f.h_j_kg) ** 0.25
            * g.rho_kg_m3**0.24
        )
    )
    return float(f.T_k), float(enhancement * conv), float(suppression * nb)


def heat_flux(p, x, G, D, wall_temperature_k, L, *, scope):
    chf = critical_heat_flux(p, G, x, D, L, scope=scope)
    ts, conv, nb = coefficients(p, x, G, D)
    tw = eos.number(wall_temperature_k, "wall temperature")
    if not 0 < tw - ts <= 10 or tw >= 647.096:
        raise ValueError("Pre-CHF wall superheat domain")
    dp = float(iapws.saturation_pressure_mpa(tw) * 1e6 - p)
    if dp <= 0:
        raise ValueError("Wall saturation pressure difference")
    q = conv * (tw - ts) + nb * (tw - ts) ** 1.24 * dp**0.75
    if q >= 0.8 * chf["heat_flux_w_m2"]:
        raise ValueError("Post-CHF/guard-limit regime unsupported")
    return float(q)


def wall_temperature(p, x, G, D, heat_flux_w_m2, L, *, scope):
    target = eos.number(heat_flux_w_m2, "heat flux")
    chf = critical_heat_flux(p, G, x, D, L, scope=scope)
    if not 0 < target < 0.8 * chf["heat_flux_w_m2"]:
        raise ValueError("Post-CHF/guard-limit regime unsupported")
    ts, _, _ = coefficients(p, x, G, D)

    # Root brackets are algebraic probes; only the returned pre-CHF state is admitted.
    def unrestricted(delta):
        _, conv, nb = coefficients(p, x, G, D)
        if delta == 0:
            return -target
        dp = float(iapws.saturation_pressure_mpa(ts + delta) * 1e6 - p)
        if dp <= 0:
            raise ValueError("Wall saturation pressure difference")
        return conv * delta + nb * delta**1.24 * dp**0.75 - target

    if unrestricted(10) < 0:
        raise ValueError("Prescribed heat flux requires unsupported wall superheat")
    delta = brentq(unrestricted, 0.0, 10.0, xtol=1e-10)
    tw = ts + delta
    q = heat_flux(p, x, G, D, tw, L, scope=scope)
    if abs(q - target) > 0.001:
        raise ValueError("Wall heat-flux inverse residual")
    return dict(
        wall_temperature_k=float(tw),
        wall_superheat_k=float(delta),
        heat_flux_residual_w_m2=q - target,
        chf_fraction=target / chf["heat_flux_w_m2"],
        chf=chf,
        version=VERSION,
        validated=False,
    )
