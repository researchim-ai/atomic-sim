"""Sparse piecewise analytic Jacobian of the pressure-driven liquid channel."""

import numpy as np
from scipy.sparse import lil_matrix

from physics.hydraulics.liquid_derivatives import conserved_derivatives
from physics.hydraulics.reconstruction import face_value_jacobian


def channel_jacobian(model, t, state):
    """Exact local chain rule on a fixed flow/limiter branch, CSC output."""
    model.rates(t, state)  # Full cell/face/Mach applicability, no projection.
    c = model.config
    n = c.cells
    size = len(state)
    volumes = list(model.network.volumes.values())
    therm = [
        conserved_derivatives(v.thermo, state[2 * i], state[2 * i + 1], v.volume_m3)
        for i, v in enumerate(volumes)
    ]
    left, right = model.network.boundaries["left"], model.network.boundaries["right"]
    flows = np.asarray(state[2 * n : 3 * n + 1])
    h = np.array([v.thermo.h_j_kg for v in volumes])
    face, face_h = face_value_jacobian(
        h, flows, left.enthalpy_j_kg, right.enthalpy_j_kg, scheme=c.enthalpy_transport
    )
    flux_grad = np.zeros((n + 1, 2 * n))
    for i in range(n):
        flux_grad[:, 2 * i : 2 * i + 2] = flows[:, None] * face_h[:, i, None] * therm[i].enthalpy
    jac = lil_matrix((size, size))
    for i in range(n):
        mrow, urow = 2 * i, 2 * i + 1
        wrow = 3 * n + 1 + i
        jac[mrow, 2 * n + i] = 1.0
        jac[mrow, 2 * n + i + 1] = -1.0
        jac[urow, : 2 * n] = flux_grad[i] - flux_grad[i + 1]
        jac[urow, 2 * n + i] = face[i]
        jac[urow, 2 * n + i + 1] = -face[i + 1]
        dq = -c.total_ua_w_k / n * therm[i].temperature
        for k in range(2):
            jac[urow, 2 * i + k] += dq[k]
            jac[wrow, 2 * i + k] = -dq[k] / (c.total_heat_capacity_j_k / n)
        jac[urow, wrow] = c.total_ua_w_k / n
        jac[wrow, wrow] = -c.total_ua_w_k / c.total_heat_capacity_j_k
    for j, junction in enumerate(model.network.junctions.values()):
        row = 2 * n + j
        a = junction.area_m2
        length = junction.length_m
        loss = junction.loss_coefficient
        flow = flows[j]
        if j > 0:
            for k in range(2):
                jac[row, 2 * (j - 1) + k] += a / length * therm[j - 1].pressure[k]
        if j < n:
            for k in range(2):
                jac[row, 2 * j + k] -= a / length * therm[j].pressure[k]
        donor = j - 1 if flow >= 0 else j
        if 0 <= donor < n:
            rho = (
                state[2 * donor] / volumes[donor].volume_m3
                if c.energy_formulation == "compatible_total"
                else volumes[donor].thermo.rho_kg_m3
            )
            for k in range(2):
                jac[row, 2 * donor + k] += (
                    loss * flow * abs(flow) / (2 * a * length * rho**2) * therm[donor].density[k]
                )
        else:
            rho = left.density_kg_m3 if donor < 0 else right.density_kg_m3
        jac[row, 2 * n + j] = -loss * abs(flow) / (rho * a * length)
    jac[-2, 2 * n] = 1.0
    jac[-2, 3 * n] = -1.0
    jac[-1, : 2 * n] = flux_grad[0] - flux_grad[-1]
    jac[-1, 2 * n] = face[0]
    jac[-1, 3 * n] = -face[-1]
    if c.energy_formulation == "compatible_total":
        from physics.hydraulics.mechanical_energy import mechanical_jacobian_correction

        jac = mechanical_jacobian_correction(model, state, jac, therm)
    result = jac.tocsc()
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("Nonfinite analytic channel Jacobian")
    return result
