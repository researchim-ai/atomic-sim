"""Independent gas-storage references: adiabatic quadrature and T-based DOP853."""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq


def coefficients(spec):
    model = spec["model"]
    v = model["vessels"]
    rho = model["fluid"]["density_kg_m3"]
    cv = model["fluid"]["liquid_cv_j_kg_k"]

    def array(k):
        return np.array([a[k] for a in v], dtype=float)

    m0 = array("initial_liquid_mass_kg")
    volume = array("total_volume_m3")
    vg0 = volume - m0 / rho
    gamma = array("gas_gamma")
    p0 = array("initial_gas_pressure_pa")
    tg0 = array("initial_gas_temperature_k")
    tl0 = array("initial_liquid_temperature_k")
    cg = p0 * vg0 / ((gamma - 1) * tg0)
    return model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array


def conductance_clock(spec, t):
    c = spec["model"]["connection"]["conductance_m3_s_pa"]
    clock = 0.0
    previous = 0.0
    for command in spec["timeline"]:
        if command["t"] > t:
            break
        clock += c * (command["t"] - previous)
        previous = command["t"]
        c = command["conductance_m3_s_pa"]
    return clock + c * (t - previous)


def adiabatic_at(spec, t):
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(spec)
    if any(
        np.any(array(k)) for k in ("gas_liquid_ua_w_k", "gas_ambient_ua_w_k", "liquid_heater_w")
    ):
        raise ValueError("Adiabatic reference excludes all heat transfers")
    connection = model["connection"]
    crack = connection["cracking_pressure_pa"]
    dp0 = p0[0] - p0[1]
    direction = 1 if dp0 > 0 else -1
    clock = conductance_clock(spec, t)
    signs = np.array([-1.0, 1.0])

    def pressure(x):
        return p0 * (vg0 / (vg0 - signs * x / rho)) ** gamma

    def driving(u):
        p = pressure(direction * u)
        dp = p[0] - p[1]
        return dp - crack if connection["kind"] == "check" else direction * dp

    x = 0.0
    if clock > 0 and dp0 != 0 and not (connection["kind"] == "check" and dp0 <= crack):
        upper = min(m0[0], rho * vg0[1]) if direction > 0 else min(m0[1], rho * vg0[0])
        equilibrium = brentq(driving, 0, upper * (1 - 1e-10), xtol=1e-13)

        def mobility(u):
            return quad(lambda z: 1 / (rho * driving(z)), 0, u, epsabs=1e-18, epsrel=2e-11)[0]

        ceiling = equilibrium * 0.5
        while mobility(ceiling) < clock:
            ceiling = (ceiling + equilibrium) / 2
            if equilibrium - ceiling < equilibrium * 1e-6:
                raise ValueError(
                    "Adiabatic quadrature is not admitted in the final near-equilibrium tail"
                )
        x = direction * brentq(lambda u: mobility(u) - clock, 0, ceiling, xtol=1e-13)

    vg = vg0 - signs * x / rho
    gas_temperature = tg0 * (vg0 / vg) ** (gamma - 1)
    gas_delta = cg * (gas_temperature - tg0)
    liquid_delta = np.zeros(2)
    donor = 0 if x >= 0 else 1
    liquid_delta[donor] = signs[donor] * x * cv * tl0[donor]
    liquid_delta[1 - donor] = -liquid_delta[donor] - sum(gas_delta)
    liquid_temperature = (m0 * cv * tl0 + liquid_delta) / ((m0 + signs * x) * cv)
    return dict(
        transfer_mass_kg=float(x),
        pressure=pressure(x),
        liquid_temperature=liquid_temperature,
        gas_temperature=gas_temperature,
        gas_boundary_work=gas_delta,
    )


def thermal_reference(spec):
    model, rho, cv, m0, volume, vg0, gamma, p0, tg0, tl0, cg, array = coefficients(spec)
    signs = np.array([-1.0, 1.0])
    ua = array("gas_liquid_ua_w_k")
    ambient = array("gas_ambient_ua_w_k")
    heater = array("liquid_heater_w")

    def evaluate(z, c):
        x = z[0]
        tl = z[1:3]
        tg = z[3:5]
        mass = m0 + signs * x
        vg = volume - mass / rho
        pressure = p0 * (tg / tg0) * (vg0 / vg)
        dp = pressure[0] - pressure[1]
        q = c * (
            max(dp - model["connection"]["cracking_pressure_pa"], 0.0)
            if model["connection"]["kind"] == "check"
            else dp
        )
        donor = 0 if q >= 0 else 1
        heat = ua * (tg - tl)
        ext = ambient * (model["ambient_temperature_k"] - tg)
        work = signs * pressure * q
        liquid_rate = (
            signs * rho * q * cv * (tl[donor] - tl)
            + signs * q * (pressure[donor] - pressure)
            + heat
            + heater
        ) / (mass * cv)
        gas_rate = (work - heat + ext) / cg
        return np.r_[rho * q, liquid_rate, gas_rate, work, sum(heater + ext)]

    initial = np.r_[0.0, tl0, tg0, np.zeros(3)]
    state = initial.copy()
    pieces = []
    start = 0.0
    c = model["connection"]["conductance_m3_s_pa"]
    events = []

    def margin(z):
        vg = vg0 - signs * z[0] / rho
        p = p0 * (z[3:5] / tg0) * (vg0 / vg)
        return p[0] - p[1] - model["connection"]["cracking_pressure_pa"]

    for command in spec["timeline"] + [dict(t=spec["duration_s"], conductance_m3_s_pa=c)]:
        end = command["t"]
        if end > start:
            result = solve_ivp(
                lambda t, z: evaluate(z, c),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-13,
                atol=1e-12,
                max_step=0.02,
                dense_output=True,
            )
            if not result.success:
                raise ValueError("Independent storage reference failed")
            if model["connection"]["kind"] == "check":
                for i in range(len(result.t) - 1):
                    a, b = margin(result.y[:, i]), margin(result.y[:, i + 1])
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        root = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: margin(result.sol(t)),
                                result.t[i],
                                result.t[i + 1],
                                xtol=1e-12,
                            )
                        )
                        events.append(dict(kind="check_threshold", t=root, direction=direction))
            pieces.append((start, end, result.sol))
            state = result.y[:, -1]
        c = command["conductance_m3_s_pa"]
        start = end

    def at(t):
        z = initial.copy() if t == 0 else next(sol(t) for a, b, sol in pieces if a <= t <= b)
        vg = vg0 - signs * z[0] / rho
        return dict(
            transfer_mass_kg=float(z[0]),
            pressure=p0 * (z[3:5] / tg0) * (vg0 / vg),
            liquid_temperature=z[1:3],
            gas_temperature=z[3:5],
            gas_boundary_work=z[5:7],
            external_heat_j=float(z[7]),
        )

    return at, events
