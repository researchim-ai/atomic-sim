"""Absolute-inventory semidiscrete reference: separate loop flux, shared IF97 EOS.

This verifies temporal evolution and coordinate transformation, not an
independent EOS or a different spatial discretization.
"""

import numpy as np
from scipy.integrate import solve_ivp

from atomic_model_hem import eos


def initial(spec):
    states = []
    vel = []
    for point in spec["initial"]:
        if point["kind"] == "pt":
            s = eos.pt(point["pressure_pa"], point["temperature_k"])
        elif point["kind"] == "px":
            s = eos.px(point["pressure_pa"], point["quality"])
        else:
            s = eos.rho_u(
                point["density_kg_m3"],
                point["internal_energy_j_kg"],
                pressure_seed_pa=point["pressure_seed_pa"],
                temperature_seed_k=point["temperature_seed_k"],
            )
        states.append(s)
        vel.append(point["velocity_m_s"])
    n = len(states)
    volume = spec["geometry"]["area_m2"] * spec["geometry"]["length_m"] / n
    m = np.array([s.density_kg_m3 * volume for s in states])
    v = np.array(vel)
    return states, np.r_[
        m,
        m * v,
        np.array([mass * s.internal_energy_j_kg for mass, s in zip(m, states)]) + 0.5 * m * v * v,
        0.0,
        0.0,
    ]


def solve(spec, times, method):
    seeds, y0 = initial(spec)
    n = len(seeds)
    area = spec["geometry"]["area_m2"]
    volume = area * spec["geometry"]["length_m"] / n
    heaters = spec["heat_power_w"]
    wave = spec["wave_speed_bound_m_s"]

    def thermos(y):
        states = []
        for i in range(n):
            m, P, E = map(float, [y[i], y[n + i], y[2 * n + i]])
            states.append(
                eos.rho_u(
                    m / volume,
                    E / m - 0.5 * (P / m) ** 2,
                    pressure_seed_pa=seeds[i].pressure_pa,
                    temperature_seed_k=seeds[i].temperature_k,
                )
            )
        return states

    def rhs(t, y):
        states = thermos(y)
        fluxes = []
        result = np.zeros_like(y)
        for left in range(n):
            right = (left + 1) % n

            def flux(i):
                m, P, E = y[i], y[n + i], y[2 * n + i]
                v = P / m
                p = states[i].pressure_pa
                w = np.array([m, P, E]) / volume
                return w, np.array([P / volume, P * v / volume + p, v * (E / volume + p)])

            wl, fl = flux(left)
            wr, fr = flux(right)
            fluxes.append((fl + fr) / 2 - wave * (wr - wl) / 2)
        for i in range(n):
            f = area * (fluxes[(i - 1) % n] - fluxes[i])
            for j in range(3):
                result[j * n + i] = f[j]
            result[2 * n + i] += heaters[i]
        result[-2] = sum(heaters)
        result[-1] = sum(q / s.temperature_k for q, s in zip(heaters, states))
        return result

    solved = solve_ivp(
        rhs,
        (0.0, times[-1]),
        y0,
        method=method,
        rtol=2e-11,
        atol=1e-10,
        max_step=2e-4,
        t_eval=times,
    )
    if not solved.success:
        raise ValueError("HEM reference IVP failed")
    observations = []
    for time, y in zip(solved.t, solved.y.T):
        states = thermos(y)
        observations.append(
            dict(
                time_s=float(time),
                cells=[
                    dict(
                        state.as_dict(),
                        mass_kg=float(y[i]),
                        momentum_kg_m_s=float(y[n + i]),
                        total_energy_j=float(y[2 * n + i]),
                    )
                    for i, state in enumerate(states)
                ],
                external_heat_j=float(y[-2]),
                external_heat_entropy_j_k=float(y[-1]),
            )
        )
    return observations
