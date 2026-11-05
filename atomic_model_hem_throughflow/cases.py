"""Inputs frozen independently of the HIPER comparison errors."""

from copy import deepcopy

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.closures import FLAGS


def baseline():
    return dict(
        schema_version=1,
        inlet_pressure_pa=15.5e6,
        inlet_enthalpy_j_kg=eos.pt(15.5e6, 600.0).enthalpy_j_kg,
        mass_flux_kg_m2_s=520.0,
        diameter_m=0.01581,
        length_m=7.1,
        roughness_m=0.0,
        gravity_m_s2=9.80665,
        heat_flux_w_m2=[200000.0],
        viscosity_model="mcadams",
        friction_scope={k: True for k in FLAGS},
        max_step_m=0.1,
    )


def matrix():
    liquid = baseline()
    liquid["heat_flux_w_m2"] = [0.0]
    boiling = baseline()
    wet = baseline()
    wet.update(
        inlet_enthalpy_j_kg=eos.px(15.5e6, 0.02).enthalpy_j_kg,
        length_m=2.0,
        heat_flux_w_m2=[20000.0],
    )
    vapor = baseline()
    vapor.update(
        inlet_pressure_pa=1e6,
        inlet_enthalpy_j_kg=eos.px(1e6, 0.9999).enthalpy_j_kg,
        mass_flux_kg_m2_s=100.0,
        diameter_m=0.02,
        length_m=1.0,
        gravity_m_s2=0.0,
        heat_flux_w_m2=[2000.0],
    )
    condensation = deepcopy(vapor)
    condensation.update(
        inlet_enthalpy_j_kg=eos.pt(1e6, eos.px(1e6, 1.0).temperature_k + 0.2).enthalpy_j_kg,
        heat_flux_w_m2=[-2000.0],
    )
    exit_liquid = deepcopy(wet)
    exit_liquid.update(
        inlet_enthalpy_j_kg=eos.px(15.5e6, 0.0001).enthalpy_j_kg,
        heat_flux_w_m2=[-2000.0],
        mass_flux_kg_m2_s=100.0,
        length_m=1.0,
    )
    exact_liquid = deepcopy(wet)
    exact_liquid.update(
        inlet_enthalpy_j_kg=eos.px(15.5e6, 0.0).enthalpy_j_kg,
        heat_flux_w_m2=[2000.0],
        mass_flux_kg_m2_s=100.0,
        length_m=1.0,
    )
    exact_vapor = deepcopy(vapor)
    exact_vapor.update(inlet_enthalpy_j_kg=eos.px(1e6, 1.0).enthalpy_j_kg, heat_flux_w_m2=[-2000.0])
    variable = baseline()
    variable["heat_flux_w_m2"] = [0.0, 200000.0, -100000.0, 0.0]
    transition = baseline()
    transition.update(
        inlet_pressure_pa=1e6,
        inlet_enthalpy_j_kg=eos.px(1e6, 0.001).enthalpy_j_kg,
        mass_flux_kg_m2_s=100.0,
        diameter_m=0.005,
        heat_flux_w_m2=[2000.0],
    )
    return [
        ("high_pressure_liquid", liquid),
        ("boiling_entry", boiling),
        ("wet_heating", wet),
        ("vapor_exit", vapor),
        ("condensation_entry", condensation),
        ("liquid_exit", exit_liquid),
        ("exact_saturated_liquid", exact_liquid),
        ("exact_saturated_vapor", exact_vapor),
        ("piecewise_heat", variable),
        ("transition_friction", transition),
    ]
