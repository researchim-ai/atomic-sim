"""Named, bounded pre-CHF constitutive models, SI; no post-CHF extrapolation."""

from dataclasses import dataclass
from functools import lru_cache
from math import exp

import numpy as np

from atomic_model_hem_closures import CHF1995CubeScope, chf_1995_cube, darcy_factor
from atomic_model_hem_mixture.friction import FLAGS, MODELS, churchill
from atomic_model_hem_qualified import eos
from atomic_model_workspace.common import keys
from physics.materials import iapws

VERSION = "qualified-hem-closures/1.0.0"
CHEN_SOURCE = "https://mooseframework.inl.gov/bison/source/materials/CoolantChannelMaterial.html"
CHF_SOURCE = "https://www-pub.iaea.org/MTCD/Publications/PDF/te_1203_prn.pdf"
SURFACE_SOURCE = "https://iapws.org/documents/release/Surf-H2O.download"
HEAT_FLAGS = (
    "pure_water",
    "round_tube",
    "vertical_upflow",
    "uniform_heating",
    "saturated_pre_chf",
    "negligible_hydrostatic_drop",
)
# Table II.I, PDF page189 / printed181, kW/m²; unshaded corners only.
# Axes: [15,16]MPa, [1000,1500,2000]kg/(m²s), [0,.05,.1].
HIGH_CHF = np.array(
    [
        [[2468.0, 2251.0, 1897.0], [2312.0, 1955.0, 1649.0], [2464.0, 2071.0, 1843.0]],
        [[2089.0, 1853.0, 1528.0], [1945.0, 1655.0, 1419.0], [2166.0, 1841.0, 1688.0]],
    ]
)
HIGH_CHF.setflags(write=False)


@lru_cache(maxsize=1024)
def transport(p):
    return iapws.saturation_properties(p)


def surface_tension(temperature_k):
    t = eos.number(temperature_k, "surface temperature")
    if not 273.16 <= t < 647.096:
        raise ValueError("IAPWS surface-tension domain")
    tau = 1 - t / 647.096
    return 0.2358 * tau**1.256 * (1 - 0.625 * tau)


def table_chf(p, G, x, D):
    p = eos.number(p, "CHF pressure")
    G = eos.number(G, "CHF mass flux")
    x = eos.number(x, "CHF quality")
    D = eos.number(D, "CHF diameter")
    if D != 0.008:
        raise ValueError("1995 nominal CHF table requires exactly8mm; no geometry correction")
    if 3e5 <= p <= 5e5 and 1000 <= G <= 1500 and 0 <= x <= 0.05:
        return chf_1995_cube(
            p, G, x, diameter_m=D, scope=CHF1995CubeScope(True, True, True, True, True)
        ).heat_flux_w_m2
    if not 15e6 <= p <= 16e6 or not 1000 <= G <= 2000 or not 0 <= x <= 0.1:
        raise ValueError("CHF outside admitted disconnected table boxes; no extrapolation")
    ig = 0 if G <= 1500 else 1
    ix = 0 if x <= 0.05 else 1
    weights = [(p - 15e6) / 1e6, (G - (1000 + 500 * ig)) / 500, (x - 0.05 * ix) / 0.05]
    q = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                q += HIGH_CHF[a, ig + b, ix + c] * np.prod(
                    [w if j else 1 - w for j, w in zip((a, b, c), weights)]
                )
    return float(q * 1000)


def biasi_chf(p, G, x, D, length):
    """Literal conservative intersection with INL's documented Table2 range.

    Published page says100..600 kg/(m²s); larger ranges elsewhere are not assumed.
    """
    for value, label in [
        (p, "pressure"),
        (G, "mass flux"),
        (x, "quality"),
        (D, "diameter"),
        (length, "heated length"),
    ]:
        eos.number(value, label)
    if (
        not 0.27e6 <= p <= 14e6
        or not 100 <= G <= 600
        or not 0 <= x <= 0.1
        or not 0.005 <= D <= 0.0375
        or not 0.2 <= length <= 6
    ):
        raise ValueError("Biasi-INL restricted documented domain")
    pb = p / 1e5
    n = 0.4 if D >= 0.01 else 0.6
    F = 0.7249 + 0.099 * pb * exp(-0.032 * pb)
    H = -1.159 + 0.149 * pb * exp(-0.019 * pb) + 9 * pb / (10 + pb * pb)
    q1 = 15.048e7 * (100 * D) ** (-n) * G ** (-0.6) * H * (1 - x)
    q2 = 2.764e7 * (100 * D) ** (-n) * G ** (-0.6) * (1.468 * F * G ** (-1 / 6) - x)
    q = q1 if G < 300 else max(q1, q2)
    if q <= 0:
        raise ValueError("Nonpositive CHF prediction")
    return float(q)


def friction(
    state, velocity, diameter, roughness, *, viscosity_model, scope, darcy_law="churchill"
):
    keys(scope, FLAGS)
    if any(scope[k] is not True for k in FLAGS) or viscosity_model not in MODELS:
        raise ValueError("Homogeneous friction scope/model")
    v = eos.number(velocity, "velocity")
    d = eos.number(diameter, "diameter")
    r = eos.number(roughness, "roughness")
    if not 0.005 <= d <= 0.3 or not 0 <= r / d <= 0.05:
        raise ValueError("Friction geometry")
    if state.region == 4:
        f, g = transport(state.pressure_pa)
        x = state.quality
        mu = (
            1 / ((1 - x) / f.viscosity_pa_s + x / g.viscosity_pa_s)
            if viscosity_model == "mcadams"
            else (1 - x) * f.viscosity_pa_s + x * g.viscosity_pa_s
        )
    else:
        mu = iapws.viscosity_pa_s(state.density_kg_m3, state.temperature_k)
    re = state.density_kg_m3 * abs(v) * d / mu
    if darcy_law not in ("churchill", "colebrook"):
        raise ValueError("Friction Darcy law")
    factor = (
        0.0
        if re == 0
        else churchill(float(re), r / d)
        if darcy_law == "churchill"
        else darcy_factor(float(re), r / d)
    )
    gradient = 0.0 if re == 0 else 32 * mu * v / d**2 * (factor * re / 64)
    return dict(
        force_density_n_m3=-gradient,
        entropy_production_w_m3_k=gradient * v / state.temperature_k,
        reynolds=float(re),
        viscosity_pa_s=float(mu),
    )


@dataclass(frozen=True)
class Boiling:
    coefficient_w_m2_k: float
    differential_conductance_w_m2_k: float
    heat_flux_w_m2: float
    chf_w_m2: float
    chf_fraction: float
    reynolds_liquid: float
    enhancement: float
    suppression: float
    nucleate_coefficient_w_m2_k: float
    convective_coefficient_w_m2_k: float
    version: str = VERSION
    source: str = CHEN_SOURCE
    variant: str = "chen-inl-documented-hfg025"
    uncertainty: str = "Not calibrated; no confidence interval assigned"
    validated: bool = False


def boiling(state, G, D, wall_temperature, length, *, scope, chf_model, maximum_chf_fraction=0.8):
    keys(scope, HEAT_FLAGS)
    if any(scope[k] is not True for k in HEAT_FLAGS):
        raise ValueError("Saturated vertical pre-CHF scope required")
    G = eos.number(G, "mass flux")
    D = eos.number(D, "diameter")
    tw = eos.number(wall_temperature, "wall temperature")
    length = eos.number(length, "heated length")
    limit = eos.number(maximum_chf_fraction, "CHF fraction bound")
    if (
        state.region != 4
        or not 0 <= state.quality <= 0.1
        or not 0.005 <= D <= 0.0375
        or not 0.2 <= length <= 6
        or not G > 0
        or not 0 < limit <= 0.8
    ):
        raise ValueError("Wet saturated Chen/CHF intersection domain; dryout/condensation refused")
    p = state.pressure_pa
    x = state.quality
    t = state.temperature_k
    if not 0 < tw - t <= 10 or tw >= 647.096 or state.density_kg_m3 * 9.80665 * length / p >= 0.01:
        raise ValueError("Wall superheat or neglected gravity envelope")
    f, g = transport(p)
    re = G * (1 - x) * D / f.viscosity_pa_s
    pr = f.cp_j_kg_k * f.viscosity_pa_s / f.conductivity_w_m_k
    if not 1e4 <= re <= 5e6 or not 0.7 <= pr <= 100:
        raise ValueError("Chen liquid convection Re/Pr envelope")
    invX = (
        0.0
        if x == 0
        else (x / (1 - x)) ** 0.9
        * (f.rho_kg_m3 / g.rho_kg_m3) ** 0.5
        * (g.viscosity_pa_s / f.viscosity_pa_s) ** 0.1
    )
    F = 1.0 if invX <= 0.1 else 2.35 * (0.213 + invX) ** 0.736
    S = 1 / (1 + 2.53e-6 * (re * F**1.25) ** 1.17)
    hc = 0.023 * re**0.8 * pr**0.4 * f.conductivity_w_m_k / D
    delta = tw - t
    dp = float(iapws.saturation_pressure_mpa(tw) * 1e6 - p)
    if dp <= 0:
        raise ValueError("Invalid wall saturation pressure difference")
    nb = (
        0.00122
        * f.conductivity_w_m_k**0.79
        * f.cp_j_kg_k**0.45
        * f.rho_kg_m3**0.49
        / (
            surface_tension(t) ** 0.5
            * f.viscosity_pa_s**0.29
            * (g.h_j_kg - f.h_j_kg) ** 0.25
            * g.rho_kg_m3**0.24
        )
        * delta**0.24
        * dp**0.75
    )
    h = F * hc + S * nb
    q = h * delta
    # Region4 complex-step derivative, with a strict physical bulk/superheat check above.
    n = iapws._SAT_N
    tt = complex(tw, 1e-20)
    theta = tt + n[8] / (tt - n[9])
    a = theta**2 + n[0] * theta + n[1]
    b = n[2] * theta**2 + n[3] * theta + n[4]
    c = n[5] * theta**2 + n[6] * theta + n[7]
    dp_dt = (((2 * c / (-b + np.sqrt(b * b - 4 * a * c))) ** 4).imag / 1e-20) * 1e6
    dq_dt = F * hc + S * nb * (1.24 + 0.75 * delta * dp_dt / dp)
    if chf_model == "table1995":
        chf = table_chf(p, G, x, D)
    elif chf_model == "biasi_inl":
        chf = biasi_chf(p, G, x, D, length)
    else:
        raise ValueError("Unknown explicit CHF model")
    if q >= limit * chf:
        raise ValueError("Pre-CHF limit exceeded; post-CHF is unsupported")
    return Boiling(
        float(h),
        float(dq_dt),
        float(q),
        float(chf),
        float(q / chf),
        float(re),
        float(F),
        float(S),
        float(nb),
        float(hc),
    )
