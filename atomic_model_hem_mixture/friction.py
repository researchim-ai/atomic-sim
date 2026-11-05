"""Homogeneous constitutive choices, not validated separated-flow correlations."""

from dataclasses import dataclass
from math import exp, log

import numpy as np

from atomic_model_hem import eos
from atomic_model_hem_closures import darcy_factor
from atomic_model_workspace.common import keys
from physics.materials import iapws

VERSION = "homogeneous-friction/1.0.0"
SOURCE = "https://engineering.purdue.edu/mudawar/files/articles-all/2005/2005_1.pdf"
FLAGS = (
    "homogeneous_equilibrium",
    "no_slip",
    "circular_straight",
    "fully_developed",
    "quasi_steady",
)
MODELS = ("mcadams", "cicchitti")


@dataclass(frozen=True)
class Result:
    viscosity_pa_s: float
    density_kg_m3: float
    reynolds: float
    darcy_factor: float | None
    pressure_gradient_pa_m: float
    force_density_n_m3: float
    entropy_production_w_m3_k: float
    viscosity_model: str
    darcy_law: str
    darcy_source: str
    version: str = VERSION
    source: str = SOURCE
    experimentally_validated: bool = False


def evaluate(
    state, velocity_m_s, diameter_m, roughness_m, *, viscosity_model, scope, darcy_law="colebrook"
):
    keys(scope, FLAGS)
    if any(scope[k] is not True for k in FLAGS):
        raise ValueError("Homogeneous equilibrium no-slip fully-developed scope required")
    if viscosity_model not in MODELS or not isinstance(state, eos.State):
        raise ValueError("Unknown viscosity model or thermodynamic state")
    velocity = eos.number(velocity_m_s, "velocity")
    diameter = eos.number(diameter_m, "diameter")
    roughness = eos.number(roughness_m, "roughness")
    if not 0.005 <= diameter <= 0.3 or not 0 <= roughness / diameter <= 0.05:
        raise ValueError("Homogeneous friction geometry domain")
    density = eos.number(state.density_kg_m3, "state density")
    eos.number(state.internal_energy_j_kg, "state internal energy")
    if density <= 0:
        raise ValueError("Nonpositive friction state density")
    if not 300 <= state.temperature_k <= 623.15:
        raise ValueError("Homogeneous transport temperature domain300..623.15K")
    if state.region == 4:
        canonical = eos.px(state.pressure_pa, state.quality)
        liquid, vapor = iapws.saturation_properties(state.pressure_pa)
        x = canonical.quality
        if viscosity_model == "mcadams":
            viscosity = 1 / ((1 - x) / liquid.viscosity_pa_s + x / vapor.viscosity_pa_s)
        else:
            viscosity = (1 - x) * liquid.viscosity_pa_s + x * vapor.viscosity_pa_s
    elif state.region in (1, 2) and state.quality is None:
        canonical = eos.pt(state.pressure_pa, state.temperature_k)
        viscosity = iapws.viscosity_pa_s(canonical.density_kg_m3, canonical.temperature_k)
    else:
        raise ValueError("Unsupported homogeneous phase/region")
    if (
        abs(canonical.density_kg_m3 / state.density_kg_m3 - 1) > 2e-9
        or abs(canonical.internal_energy_j_kg - state.internal_energy_j_kg) > 0.002
        or abs(canonical.temperature_k - state.temperature_k) > 1e-8
    ):
        raise ValueError("Inconsistent friction thermodynamic coordinates")
    rho = canonical.density_kg_m3
    reynolds = rho * abs(velocity) * diameter / viscosity
    if darcy_law not in ("colebrook", "churchill"):
        raise ValueError("Unknown declared Darcy law")
    factor = (
        None
        if reynolds == 0
        else (
            darcy_factor(float(reynolds), roughness / diameter)
            if darcy_law == "colebrook"
            else churchill(float(reynolds), roughness / diameter)
        )
    )
    # f*Re/64 preserves the exact zero-speed limit without squaring tiny v.
    gradient = (
        0.0 if reynolds == 0 else 32 * viscosity * velocity / diameter**2 * (factor * reynolds / 64)
    )
    force = -gradient
    return Result(
        float(viscosity),
        float(rho),
        float(reynolds),
        factor,
        float(gradient),
        float(force),
        float(-force * velocity / canonical.temperature_k),
        viscosity_model,
        darcy_law,
        "https://files.engineering.com/files/85c0f3a6-a102-4a22-9d35-f15858c0dd2b/CEM_-_Friction-factor_equation_%281977%29.pdf"
        if darcy_law == "churchill"
        else "https://doi.org/10.1680/ijoti.1939.13150",
    )


def churchill(reynolds, relative_roughness):
    """Darcy form of Churchill1977 eq18; log-domain powers, no Re clipping."""
    re = eos.number(reynolds, "Reynolds")
    rr = eos.number(relative_roughness, "roughness ratio")
    if not 1e-300 <= re <= 1e8 or not 0 <= rr <= 0.05:
        raise ValueError("Churchill computational Re/roughness domain")
    base = abs(2.457 * log(1 / ((7 / re) ** 0.9 + 0.27 * rr)))
    log_a = float("-inf") if base == 0 else 16 * log(base)
    log_b = 16 * log(37530 / re)
    log_sum = float(np.logaddexp(12 * log(8 / re), -1.5 * np.logaddexp(log_a, log_b)))
    return 8 * exp(log_sum / 12)
