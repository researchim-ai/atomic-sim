"""Explicit forward-flow steady initialization for the liquid wall channel.

No state/ledger edits: the result is a full reproducible initial configuration.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicConfig
from physics.hydraulics.isobaric import _finite
from physics.materials.iapws import properties_ph, properties_pt


def equilibrated_config(config: DynamicConfig, *, wall_offset_k=0.0):
    """Steady donor-flow state, optionally add a uniform wall temperature pulse.

    Only positive pressure drive and positive total resistance are supported.
    This is an initial-condition solver, not a feedback controller or pump law.
    """
    c = config
    n = c.cells
    if c.energy_formulation == "compatible_total":
        raise DomainError("Compatible total energy requires its own steady initializer")
    if c.enthalpy_transport != "donor" and c.heat_w != 0:
        raise DomainError("Heated MUSCL equilibrium requires a separate steady initializer")
    offset = _finite(wall_offset_k, "wall_offset_k")
    if any(
        getattr(c, name) is not None
        for name in (
            "initial_pressure_profile_pa",
            "initial_temperature_profile_k",
            "initial_wall_temperature_profile_k",
        )
    ):
        raise ValueError("Equilibrium initialization requires an unprofiled configuration")
    if c.left_pressure_pa <= c.right_pressure_pa:
        raise DomainError("Steady initializer supports forward pressure drive only")
    losses = c.junction_loss_coefficients or (c.loss_coefficient,) * (n + 1)
    if sum(losses) <= 0:
        raise DomainError("Steady initializer requires positive total resistance")
    inlet = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    estimate = c.area_m2 * np.sqrt(
        2 * inlet.rho_kg_m3 * (c.left_pressure_pa - c.right_pressure_pa) / sum(losses)
    )

    def profile(flow):
        p = c.left_pressure_pa
        h = inlet.h_j_kg
        states = []
        for j, loss in enumerate(losses):
            donor = properties_ph(p, h)
            if donor.region != 1:
                raise DomainError("Steady donor left liquid region")
            p -= loss * flow**2 / (2 * donor.rho_kg_m3 * c.area_m2**2)
            if j < n:
                h = inlet.h_j_kg + c.heat_w * (j + 1) / (flow * n)
                state = properties_ph(p, h)
                if state.region != 1:
                    raise DomainError("Steady initializer left liquid region")
                states.append(state)
        return p, states

    flow = brentq(
        lambda f: profile(f)[0] - c.right_pressure_pa, 0.5 * estimate, 2 * estimate, xtol=1e-12
    )
    _, states = profile(flow)
    return replace(
        c,
        initial_flow_kg_s=float(flow),
        initial_pressure_profile_pa=tuple(float(s.p_pa) for s in states),
        initial_temperature_profile_k=tuple(float(s.T_k) for s in states),
        initial_wall_temperature_profile_k=tuple(
            float(s.T_k + c.heat_w / c.total_ua_w_k + offset) for s in states
        ),
    )
