"""Dimensionally explicit groups for a heated vertical round tube."""
from math import isfinite


def tube_groups(*, diameter_m, mass_flux_kg_m2_s, density_kg_m3, viscosity_pa_s,
                conductivity_w_m_k, cp_j_kg_k, expansion_per_k, heat_flux_w_m2,
                direction):
    values = (diameter_m, mass_flux_kg_m2_s, density_kg_m3, viscosity_pa_s,
              conductivity_w_m_k, cp_j_kg_k, expansion_per_k, heat_flux_w_m2)
    if any(type(v) not in (int, float) or not isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite dimensional tube inputs required')
    if type(direction) is not str or direction not in ('upward', 'downward'):
        raise ValueError('Explicit vertical direction required')
    try:
        re = mass_flux_kg_m2_s * diameter_m / viscosity_pa_s
        pr = cp_j_kg_k * viscosity_pa_s / conductivity_w_m_k
        nu = viscosity_pa_s / density_kg_m3
        grq = 9.80665 * expansion_per_k * heat_flux_w_m2 * diameter_m**4 / (conductivity_w_m_k * nu**2)
        b = (1 if direction == 'upward' else -1) * grq / (8 * re)
        boq = grq / (re**3.425 * pr**.8)
    except (OverflowError, ZeroDivisionError) as error:
        raise ValueError('Tube dimensionless groups outside floating-point range') from error
    if not all(isfinite(v) and v > 0 for v in (re, pr, grq, boq)) or not isfinite(b):
        raise ValueError('Tube dimensionless groups overflow/underflow')
    return dict(reynolds=re, prandtl=pr, grashof_heat_flux=grq, laminar_buoyancy=b,
                iaea_heat_flux_buoyancy=boq, direction=direction)
