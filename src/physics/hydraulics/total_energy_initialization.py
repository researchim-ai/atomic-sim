"""Bidirectional discrete steady initialization for the total-energy pipe."""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from core.status import DomainError
from physics.hydraulics.isobaric import _finite
from physics.materials.iapws import properties_ph, properties_pt


def equilibrated_total_config(config, *, wall_offset_k=0.0):
    c = config
    n = c.cells
    offset = _finite(wall_offset_k, "wall_offset_k")
    if c.energy_formulation != "compatible_total":
        raise DomainError("Total-energy initializer requires compatible_total")
    if any(
        getattr(c, k) is not None
        for k in (
            "initial_pressure_profile_pa",
            "initial_temperature_profile_k",
            "initial_wall_temperature_profile_k",
        )
    ):
        raise ValueError("Initializer requires an unprofiled configuration")
    losses = c.junction_loss_coefficients or (c.loss_coefficient,) * (n + 1)
    if c.left_pressure_pa == c.right_pressure_pa or sum(losses) <= 0:
        raise DomainError("Nonzero pressure drive and positive resistance required")
    if c.left_pressure_pa < c.right_pressure_pa:
        # Solve in the direction of flow. Geometry and reservoir temperatures
        # must be reflected together; pressure exchange alone changes the pipe.
        reflected = replace(
            c,
            left_pressure_pa=c.right_pressure_pa,
            right_pressure_pa=c.left_pressure_pa,
            left_temperature_k=c.right_temperature_k,
            right_temperature_k=c.left_temperature_k,
            junction_lengths_m=(
                tuple(reversed(c.junction_lengths_m)) if c.junction_lengths_m else None
            ),
            junction_loss_coefficients=tuple(reversed(losses)),
        )
        forward = equilibrated_total_config(reflected, wall_offset_k=offset)
        return replace(
            c,
            initial_flow_kg_s=-forward.initial_flow_kg_s,
            initial_pressure_profile_pa=tuple(reversed(forward.initial_pressure_profile_pa)),
            initial_temperature_profile_k=tuple(reversed(forward.initial_temperature_profile_k)),
            initial_wall_temperature_profile_k=tuple(
                reversed(forward.initial_wall_temperature_profile_k)
            ),
        )
    inlet = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    estimate = c.area_m2 * np.sqrt(
        2 * inlet.rho_kg_m3 * (c.left_pressure_pa - c.right_pressure_pa) / sum(losses)
    )

    def state_at_total_h(p, total_h, flow):
        h = total_h
        for _ in range(20):
            state = properties_ph(p, h)
            if state.region != 1:
                raise DomainError("Total-enthalpy initialization left liquid region")
            target = total_h - 0.5 * (flow / (state.rho_kg_m3 * c.area_m2)) ** 2
            if abs(target - h) < 1e-8:
                return properties_ph(p, target)
            h = target
        raise DomainError("Total-enthalpy fixed point did not converge")

    def profile(flow):
        p = c.left_pressure_pa
        donor = inlet
        states = []
        total_h = inlet.h_j_kg + 0.5 * (flow / (inlet.rho_kg_m3 * c.area_m2)) ** 2
        for j, loss in enumerate(losses):
            p -= loss * flow**2 / (2 * donor.rho_kg_m3 * c.area_m2**2)
            if j < n:
                donor = state_at_total_h(p, total_h + c.heat_w * (j + 1) / (flow * n), flow)
                states.append(donor)
        return p, states

    flow = brentq(
        lambda f: profile(f)[0] - c.right_pressure_pa, 0.5 * estimate, 2 * estimate, xtol=1e-12
    )
    _, states = profile(flow)
    initialized = replace(
        c,
        initial_flow_kg_s=float(flow),
        initial_pressure_profile_pa=tuple(float(s.p_pa) for s in states),
        initial_temperature_profile_k=tuple(float(s.T_k) for s in states),
        initial_wall_temperature_profile_k=tuple(
            float(s.T_k + c.heat_w / c.total_ua_w_k + offset) for s in states
        ),
    )
    if c.enthalpy_transport == "muscl_mc":
        from physics.hydraulics.muscl_initialization import refine_total_equilibrium

        return refine_total_equilibrium(initialized, wall_offset_k=offset)
    return initialized
