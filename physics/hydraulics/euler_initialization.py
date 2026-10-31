"""Local steady HLLC solve using the actual conservative production residual."""

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from core.status import DomainError
from physics.hydraulics.euler_channel import EulerLiquidChannel
from physics.materials.iapws import _region1_raw, properties_pt


def steady_contact_config(config):
    c = config
    n = c.cells
    if c.numerical_flux != "hllc" or c.boundary != "reservoir" or c.total_ua_w_k <= 0:
        raise ValueError("Steady contact initializer requires HLLC exterior states and positive UA")
    profile_keys = [k for k in c.__dataclass_fields__ if "profile" in k]
    if any(getattr(c, k) is not None for k in profile_keys):
        raise ValueError("Unprofiled seed required")
    if c.left_pressure_pa == c.right_pressure_pa:
        raise DomainError("Nonzero pressure drive required")
    if c.left_velocity_m_s != 0 or c.right_velocity_m_s != 0:
        raise ValueError("Initializer currently supports resting exterior states only")
    if c.left_pressure_pa < c.right_pressure_pa:
        forward = steady_contact_config(
            replace(
                c,
                left_pressure_pa=c.right_pressure_pa,
                right_pressure_pa=c.left_pressure_pa,
                left_temperature_k=c.right_temperature_k,
                right_temperature_k=c.left_temperature_k,
            )
        )
        return replace(
            c,
            initial_pressure_profile_pa=tuple(reversed(forward.initial_pressure_profile_pa)),
            initial_temperature_profile_k=tuple(reversed(forward.initial_temperature_profile_k)),
            initial_velocity_profile_m_s=tuple(
                -u for u in reversed(forward.initial_velocity_profile_m_s)
            ),
            initial_wall_temperature_profile_k=tuple(
                reversed(forward.initial_wall_temperature_profile_k)
            ),
        )
    inlet = properties_pt(c.left_pressure_pa, c.left_temperature_k)
    dp = c.left_pressure_pa - c.right_pressure_pa
    velocity = dp / (2 * c.wave_speed_bound_m_s * inlet.rho_kg_m3)
    flow = velocity * inlet.rho_kg_m3 * c.area_m2
    rise = c.heat_w / (flow * inlet.cp_j_kg_k)
    temperatures = c.left_temperature_k + rise * (np.arange(n) + 1) / n
    pressure = 0.5 * (c.left_pressure_pa + c.right_pressure_pa)
    model = EulerLiquidChannel(c)
    scales = np.tile([flow, max(dp * c.area_m2, 1.0), max(c.heat_w, 1.0)], n)
    reference_h = inlet.h_j_kg

    def evaluate(x, need_jac=False):
        p = pressure + dp * x[:n]
        t = temperatures + x[n : 2 * n]
        u = velocity * x[2 * n :]
        y = model._state.copy()
        transform = lil_matrix((len(y), 3 * n)) if need_jac else None
        for i in range(n):
            q = model.inventory(p[i], t[i], u[i])
            y[3 * i : 3 * i + 3] = q
            if need_jac:
                s = properties_pt(float(p[i]), float(t[i]))
                raw = _region1_raw(float(t[i]), float(p[i] / 1e6))
                mass = q[0]
                volume = model.volume
                dm_dp = -volume * raw["dv_dp_mpa"] / 1e6 / raw["v"] ** 2
                dm_dt = -volume * raw["dv_dt"] / raw["v"] ** 2
                total = s.u_j_kg + 0.5 * u[i] ** 2
                transform[3 * i : 3 * i + 3, i] = (
                    np.array([dm_dp, dm_dp * u[i], dm_dp * total + mass * raw["du_dp_mpa"] * 0.001])
                    * dp
                )
                transform[3 * i : 3 * i + 3, n + i] = np.array(
                    [dm_dt, dm_dt * u[i], dm_dt * total + mass * raw["du_dt"] * 1000]
                )
                transform[3 * i : 3 * i + 3, 2 * n + i] = (
                    np.array([0, mass, mass * u[i]]) * velocity
                )
                transform[3 * n + i, n + i] = 1
        y[3 * n : 4 * n] = t + c.heat_w / c.total_ua_w_k
        if need_jac:
            jac = (model.analytic_jacobian(0, y)[: 3 * n, :] @ transform.tocsc()).toarray()
            jac[2::3] -= reference_h * jac[0::3]
            return jac / scales[:, None], y
        residual = model.derivatives(0, y)[: 3 * n].copy()
        residual[2::3] -= reference_h * residual[0::3]
        return residual / scales, y

    x0 = np.r_[np.zeros(2 * n), np.ones(n)]
    result = least_squares(
        lambda x: evaluate(x)[0],
        x0,
        jac=lambda x: evaluate(x, True)[0],
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=100,
    )
    if not result.success:
        raise DomainError("Steady HLLC nonlinear solve failed")
    _, y = evaluate(result.x)
    cells = [model.cell(q) for q in y[: 3 * n].reshape(n, 3)]
    cfg = replace(
        c,
        initial_pressure_profile_pa=tuple(s[0].p_pa for s in cells),
        initial_temperature_profile_k=tuple(s[0].T_k for s in cells),
        initial_velocity_profile_m_s=tuple(s[1] for s in cells),
        initial_wall_temperature_profile_k=tuple(float(t) for t in y[3 * n : 4 * n]),
    )
    check = EulerLiquidChannel(cfg)
    rhs = check.derivatives(0, check._state)[: 3 * n].reshape(n, 3)
    if (
        np.max(abs(rhs[:, 0])) > 1e-8
        or np.max(abs(rhs[:, 1])) > 1e-6
        or np.max(abs(rhs[:, 2])) > 1e-3
    ):
        raise DomainError("Steady HLLC solve failed physical residual acceptance")
    return cfg
