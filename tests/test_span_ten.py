"""Дарси по участкам, напор столбов, скачок амплитуды, баклинг и гелий."""

import math

import numpy as np
import pytest

from core.solvers import integrate, per_variable_atol
from core.status import DomainError
from physics.depletion.chains import (
    bateman_daughter,
    decay_constant_per_s,
    transient_equilibrium_ratio,
)
from physics.hydraulics.circulation import buoyancy_head_pa
from physics.hydraulics.closures import (
    annulus_flow_area_m2,
    annulus_hydraulic_diameter_m,
    friction_darcy,
    series_darcy_drop_pa,
)
from physics.hydraulics.network import G
from physics.materials.coolants import (
    helium_cp_j_kg_k,
    helium_cv_j_kg_k,
    helium_enthalpy_j_kg,
    helium_internal_energy_j_kg,
    helium_specific_gas_constant_j_kg_k,
    sodium_cp_j_kg_k,
    sodium_frozen_cp_gap_j_kg,
)
from physics.neutronics.diffusion import (
    absorption_reactivity_delta,
    buckling_k,
    buckling_k_slope,
    critical_buckling_m2,
    mean_free_path_m,
    reactivity_from_k,
    uncollided_fraction,
)
from physics.neutronics.point_kinetics import PointKinetics, prompt_jump_ratio


def test_series_darcy_annulus_and_columns():
    mass_flow, diameter, viscosity, density = 2.0, 0.05, 1.0e-3, 800.0
    length, roughness = 10.0, 1.0e-4
    flow_area = math.pi * diameter**2 / 4.0
    reynolds = mass_flow * diameter / (flow_area * viscosity)
    inner = (roughness / diameter) / 3.7
    friction = (1.0 / (-1.8 * math.log10(inner**1.11 + 6.9 / reynolds))) ** 2
    darcy = friction * (length / diameter) * mass_flow**2 / (density * flow_area**2) / 2.0
    assert friction_darcy(
        mass_flow, diameter, viscosity, roughness, length, density
    ) == pytest.approx(darcy, rel=1.0e-12)
    split = series_darcy_drop_pa(
        mass_flow, diameter, viscosity, [roughness, roughness], [4.0, 6.0], density
    )
    assert split == pytest.approx(darcy, rel=1.0e-12)
    mixed = series_darcy_drop_pa(
        mass_flow, diameter, viscosity, [1.0e-5, 1.0e-3], [5.0, 5.0], density
    )
    smooth = friction_darcy(mass_flow, diameter, viscosity, 1.0e-5, 10.0, density)
    rough = friction_darcy(mass_flow, diameter, viscosity, 1.0e-3, 10.0, density)
    assert smooth < mixed < rough

    outer, inner_diameter = 0.02, 0.012
    area = annulus_flow_area_m2(outer, inner_diameter)
    perimeter = math.pi * (outer + inner_diameter)
    assert 4.0 * area / perimeter == pytest.approx(
        annulus_hydraulic_diameter_m(outer, inner_diameter)
    )

    head = buoyancy_head_pa(900.0, 700.0, 8.0)
    assert head == pytest.approx((900.0 - 700.0) * G * 8.0)
    assert buoyancy_head_pa(800.0, 800.0, 8.0) == pytest.approx(0.0)
    top = 1.0e6
    assert (top + 900.0 * G * 8.0) - (top + 700.0 * G * 8.0) == pytest.approx(head)


def test_prompt_jump_buckling_and_half_life():
    beta, reactivity = 0.0065, 0.002
    ratio = prompt_jump_ratio(beta, reactivity)
    assert (reactivity - beta) * ratio + beta == pytest.approx(0.0, abs=1.0e-15)
    kinetics = PointKinetics(np.array([beta]), np.array([0.08]), 1.0e-6)
    jumped = kinetics.reference_step(kinetics.initial_state(1.0), reactivity, 0.01)
    assert jumped[0] == pytest.approx(ratio, rel=1.0e-3)
    with pytest.raises(DomainError):
        prompt_jump_ratio(beta, beta)

    diffusion, absorption, fission = 0.02, 0.1, 0.125
    buckling = critical_buckling_m2(diffusion, absorption, fission)
    assert buckling_k(diffusion, absorption, fission, buckling) == pytest.approx(1.0, rel=1.0e-12)
    assert buckling_k_slope(diffusion, absorption, fission, buckling) == pytest.approx(
        -diffusion / fission, rel=1.0e-12
    )
    with pytest.raises(DomainError):
        critical_buckling_m2(diffusion, absorption, absorption)

    k_before = fission / absorption
    added = 0.004
    k_after = fission / (absorption + added)
    delta = reactivity_from_k(k_after) - reactivity_from_k(k_before)
    assert delta == pytest.approx(absorption_reactivity_delta(added, fission), rel=1.0e-12)

    half_life = 3600.0
    assert math.exp(-decay_constant_per_s(half_life) * half_life) == pytest.approx(0.5)
    parent_lambda, daughter_lambda, time_s, parent0 = 0.01, 0.5, 40.0, 8.0
    daughter = bateman_daughter(parent0, parent_lambda, daughter_lambda, time_s)
    parent = parent0 * math.exp(-parent_lambda * time_s)
    assert daughter / parent == pytest.approx(
        transient_equilibrium_ratio(parent_lambda, daughter_lambda), rel=1.0e-8
    )
    step = 1.0e-4
    slope = (
        bateman_daughter(parent0, parent_lambda, daughter_lambda, time_s + step)
        - bateman_daughter(parent0, parent_lambda, daughter_lambda, time_s - step)
    ) / (2.0 * step)
    assert slope == pytest.approx(parent_lambda * parent - daughter_lambda * daughter, rel=1.0e-6)
    path = mean_free_path_m(2.5)
    assert uncollided_fraction(2.5, path) == pytest.approx(math.exp(-1.0))


def test_helium_sodium_gap_and_vector_tolerance():
    temperature = 600.0
    step = 1.0e-3
    slope = (
        helium_enthalpy_j_kg(temperature + step) - helium_enthalpy_j_kg(temperature - step)
    ) / (2.0 * step)
    assert slope == pytest.approx(helium_cp_j_kg_k(temperature), rel=1.0e-8)
    assert helium_cv_j_kg_k(temperature) == pytest.approx(
        helium_cp_j_kg_k(temperature) - helium_specific_gas_constant_j_kg_k()
    )
    assert helium_enthalpy_j_kg(temperature) - helium_internal_energy_j_kg(
        temperature
    ) == pytest.approx(helium_specific_gas_constant_j_kg_k() * (temperature - 300.0))

    def primitive(value: float) -> float:
        offset = value - 371.0
        return 1436.0 * offset - 0.29 * offset**2 + 4.6e-4 * offset**3 / 3.0

    start, end = 500.0, 800.0
    gap = (primitive(end) - primitive(start)) - sodium_cp_j_kg_k(start) * (end - start)
    assert sodium_frozen_cp_gap_j_kg(start, end) == pytest.approx(gap, rel=1.0e-12)
    assert gap < 0.0

    scales = np.array([1.0, 1.0e-3])
    absolute = per_variable_atol(scales, 1.0e-8)
    assert absolute / scales == pytest.approx(1.0e-8)
    result = integrate(
        lambda _time, state: -state,
        np.array([1.0, 1.0]),
        (0.0, 0.2),
        rtol=1.0e-8,
        atol=absolute,
    )
    assert result.final == pytest.approx(math.exp(-0.2), rel=1.0e-6)
