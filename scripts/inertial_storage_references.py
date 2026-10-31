"""Independent temperature-based dynamics and nonlinear lossless period quadrature."""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from scripts.storage_references import coefficients


def reference(spec):
    nested = dict(model=spec["model"]["storage_configuration"])
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(nested)
    signs = np.array([-1.0, 1.0])
    ua = array("gas_liquid_ua_w_k")
    ambient = array("gas_ambient_ua_w_k")
    heater = array("liquid_heater_w")
    inertia = spec["model"]["line"]["inertance_m_inv"]

    def rhs(t, z, resistance):
        x = z[0]
        tl = z[1:3]
        tg = z[3:5]
        m = z[8]
        mass = m0 + signs * x
        vg = volume - mass / rho
        pressure = p0 * (tg / tg0) * (vg0 / vg)
        donor = 0 if m >= 0 else 1
        heat = ua * (tg - tl)
        ext = ambient * (model["ambient_temperature_k"] - tg)
        work = signs * pressure * m / rho
        dissipated = np.zeros(2)
        dissipated[1 - donor] = resistance * abs(m) ** 3 / rho
        liquid_rate = (signs * m * cv * (tl[donor] - tl) + dissipated + heat + heater) / (mass * cv)
        return np.r_[
            m,
            liquid_rate,
            (work - heat + ext) / cg,
            work,
            sum(heater + ext),
            (pressure[0] - pressure[1] - resistance * m * abs(m)) / inertia,
        ]

    initial = np.r_[0.0, tl0, tg0, np.zeros(3), spec["model"]["initial_mass_flow_kg_s"]]
    state = initial.copy()
    pieces = []
    events = []
    start = 0.0
    resistance = spec["model"]["line"]["resistance_pa_s2_kg2"]
    for command in spec["timeline"] + [dict(t=spec["duration_s"], resistance_pa_s2_kg2=resistance)]:
        end = command["t"]
        if end > start:
            result = solve_ivp(
                lambda t, z: rhs(t, z, resistance),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-13,
                atol=1e-12,
                max_step=0.01,
                dense_output=True,
            )
            if not result.success:
                raise ValueError("Independent inertial storage reference failed")
            for i in range(len(result.t) - 1):
                a, b = result.y[8, i : i + 2]
                direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                if direction:
                    root = (
                        float(result.t[i + 1])
                        if b == 0
                        else brentq(
                            lambda t: result.sol(t)[8], result.t[i], result.t[i + 1], xtol=1e-12
                        )
                    )
                    events.append(dict(kind="flow_zero", t=root, direction=direction))
            pieces.append((start, end, result.sol))
            state = result.y[:, -1]
        start = end
        resistance = command["resistance_pa_s2_kg2"]

    def at(t):
        z = initial.copy() if t == 0 else next(sol(t) for a, b, sol in pieces if a <= t <= b)
        vg = vg0 - signs * z[0] / rho
        return dict(
            transfer_mass_kg=float(z[0]),
            mass_flow_kg_s=float(z[8]),
            pressure=p0 * (z[3:5] / tg0) * (vg0 / vg),
            liquid_temperature=z[1:3],
            gas_temperature=z[3:5],
            gas_boundary_work=z[5:7],
            external_heat_j=float(z[7]),
        )

    return at, events


def lossless_half_period(spec):
    nested = dict(model=spec["model"]["storage_configuration"])
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(nested)
    if (
        spec["timeline"]
        or spec["model"]["initial_mass_flow_kg_s"] != 0
        or spec["model"]["line"]["resistance_pa_s2_kg2"] != 0
        or any(
            np.any(array(k)) for k in ("gas_liquid_ua_w_k", "gas_ambient_ua_w_k", "liquid_heater_w")
        )
    ):
        raise ValueError("Period quadrature requires adiabatic, lossless start at a turning point")
    dp = p0[0] - p0[1]
    if dp == 0:
        raise ValueError("No finite-amplitude oscillation at equilibrium")
    direction = 1 if dp > 0 else -1
    signs = np.array([-1.0, 1.0])
    u0 = p0 * vg0 / (gamma - 1)
    inertia = spec["model"]["line"]["inertance_m_inv"]

    def gas_change(u):
        vg = vg0 - signs * direction * u / rho
        return float(sum(u0 * np.expm1((gamma - 1) * np.log(vg0 / vg))))

    bound = min(m0[0], rho * vg0[1]) if direction > 0 else min(m0[1], rho * vg0[0])
    turn = brentq(gas_change, bound * 1e-8, bound * (1 - 1e-8), xtol=1e-13)
    pturn = p0 * (vg0 / (vg0 - signs * direction * turn / rho)) ** gamma

    def integrand(theta):
        if theta < 1e-4:
            return np.sqrt(2 * inertia * turn / abs(dp))
        if np.pi / 2 - theta < 1e-4:
            return np.sqrt(2 * inertia * turn / abs(pturn[0] - pturn[1]))
        u = turn * np.sin(theta) ** 2
        speed = np.sqrt(-2 * rho * gas_change(u) / inertia)
        return 2 * turn * np.sin(theta) * np.cos(theta) / speed

    return float(
        quad(integrand, 0, np.pi / 2, epsabs=1e-9, epsrel=1e-10, points=[1e-4, np.pi / 2 - 1e-4])[0]
    )
