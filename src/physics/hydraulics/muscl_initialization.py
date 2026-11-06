"""Solve the actual steady MUSCL total-enthalpy and pressure-loss equations.

A donor profile supplies only the starting guess. Optimizer success is not
acceptance: the reconstructed model's full physical RHS is checked separately.
Forward orientation only; the public initializer handles reflection.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from core.status import DomainError
from physics.hydraulics.reconstruction import face_values
from physics.materials.iapws import properties_pt

MOMENTUM_TOLERANCE_KG_S2 = 1e-6
ENERGY_TOLERANCE_W = 1e-3


def refine_total_equilibrium(seed, *, wall_offset_k=0.0):
    from physics.hydraulics.dynamic_channel import DynamicWallChannel

    c = seed
    n = c.cells
    if c.enthalpy_transport != "muscl_mc" or c.energy_formulation != "compatible_total":
        raise DomainError("MUSCL refinement requires compatible_total and muscl_mc")
    if c.initial_flow_kg_s <= 0 or c.left_pressure_pa <= c.right_pressure_pa:
        raise DomainError("MUSCL refinement requires a forward donor seed")
    left = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    right = properties_pt(c.right_pressure_pa, c.right_temperature_k)
    flow_scale = c.initial_flow_kg_s
    pressure_scale = c.left_pressure_pa - c.right_pressure_pa
    energy_scale = max(abs(c.heat_w) / n, flow_scale * 1000.0)
    p_seed = np.array(c.initial_pressure_profile_pa)
    t_seed = np.array(c.initial_temperature_profile_k)
    losses = np.array(c.junction_loss_coefficients)

    def decode(x):
        p = p_seed + pressure_scale * x[:n]
        temperature = t_seed + x[n : 2 * n]
        flow = flow_scale * x[-1]
        states = [properties_pt(float(pi), float(ti)) for pi, ti in zip(p, temperature)]
        if any(s.region != 1 for s in states):
            raise DomainError("MUSCL steady solve left liquid IF97 region 1")
        return p, temperature, flow, states

    def residual(x):
        p, _, flow, states = decode(x)
        rho = np.r_[left.rho_kg_m3, [s.rho_kg_m3 for s in states]]
        enthalpy = face_values(
            [s.h_j_kg for s in states],
            np.full(n + 1, flow),
            left.h_j_kg,
            right.h_j_kg,
            scheme="muscl_mc",
        )
        # Both adjacent pressures are checked by DynamicWallChannel on acceptance.
        flux = flow * (enthalpy + 0.5 * (flow / (rho * c.area_m2)) ** 2)
        pressure = np.r_[c.left_pressure_pa, p, c.right_pressure_pa]
        momentum = np.diff(-pressure) - losses * flow**2 / (2 * rho * c.area_m2**2)
        energy = flux[:-1] - flux[1:] + c.heat_w / n
        return np.r_[momentum / pressure_scale, energy / energy_scale]

    initial = np.r_[np.zeros(2 * n), 1.0]
    lower = np.r_[(611.657 - p_seed) / pressure_scale, 273.15 - t_seed, 0.01]
    upper = np.r_[(100e6 - p_seed) / pressure_scale, 623.15 - t_seed, 100.0]
    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        jac="3-point",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    if not result.success or not np.all(np.isfinite(result.fun)):
        raise DomainError("MUSCL steady nonlinear solve did not converge")
    p, temperature, flow, states = decode(result.x)
    cfg = replace(
        c,
        initial_flow_kg_s=float(flow),
        initial_pressure_profile_pa=tuple(float(v) for v in p),
        initial_temperature_profile_k=tuple(float(v) for v in temperature),
        initial_wall_temperature_profile_k=tuple(
            float(v + c.heat_w / c.total_ua_w_k) for v in temperature
        ),
    )
    # Independent acceptance uses the production RHS including inverse EOS,
    # reconstructed-face guards, Mach limit, wall coupling and kinetic work.
    model = DynamicWallChannel(cfg)
    rhs = model.derivatives(0.0, model._state)
    if (
        max(abs(rhs[2 * n : 3 * n + 1])) > MOMENTUM_TOLERANCE_KG_S2
        or max(abs(rhs[1 : 2 * n : 2])) > ENERGY_TOLERANCE_W
        or max(abs(rhs[3 * n + 1 : 4 * n + 1])) > 1e-9
    ):
        raise DomainError("MUSCL steady solution failed production physical RHS tolerances")
    # Validate interface applicability with the same production model, above;
    # no clipping/projection is performed to force liquid acceptance.
    if wall_offset_k:
        cfg = replace(
            cfg,
            initial_wall_temperature_profile_k=tuple(
                float(v + wall_offset_k) for v in cfg.initial_wall_temperature_profile_k
            ),
        )
        perturbed = DynamicWallChannel(cfg)
        perturbed.derivatives(0.0, perturbed._state)
    return cfg
