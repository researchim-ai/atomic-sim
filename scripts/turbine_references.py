"""Independent speed-variable DOP853 and IF97 expansion reference; no runtime train calls."""

import numpy as np
from scipy.integrate import solve_ivp

from physics.materials.iapws import saturation_properties


def specific_work(spec):
    cfg = spec["model"]
    _, inlet = saturation_properties(cfg["inlet_pressure_pa"])
    liquid, vapor = saturation_properties(cfg["outlet_pressure_pa"])
    entropy_fraction = (inlet.s_j_kg_k - liquid.s_j_kg_k) / (vapor.s_j_kg_k - liquid.s_j_kg_k)
    his = liquid.h_j_kg + entropy_fraction * (vapor.h_j_kg - liquid.h_j_kg)
    return float(cfg["isentropic_efficiency"] * (inlet.h_j_kg - his))


def reference(spec):
    cfg = spec["model"]
    work = specific_work(spec)
    initial = np.array([spec["initial_omega_rad_s"], 0.0, 0.0, 0.0, 0.0])
    state = initial.copy()
    segments = []
    active = spec["controls"]
    start = 0.0
    boundary = None
    for command in [*spec["timeline"], dict(t=spec["duration_s"], controls=None)]:
        end = command["t"]
        if end > start:
            controls = dict(active)

            def rhs(t, y):
                power = controls["mass_flow_kg_s"] * work
                g = (
                    controls["generator_conductance_nm_s_rad"]
                    if controls["generator_connected"]
                    else 0.0
                )
                generator = g * y[0] ** 2
                bearing = cfg["bearing_drag_nm_s_rad"] * y[0] ** 2
                return [
                    (power - generator - bearing) / (cfg["inertia_kg_m2"] * y[0]),
                    power,
                    cfg["generator_efficiency"] * generator,
                    (1 - cfg["generator_efficiency"]) * generator,
                    bearing,
                ]

            def lower(t, y):
                return 1.0 if stationary_lower else y[0] - cfg["min_omega_rad_s"]

            def upper(t, y):
                return 1.0 if stationary_upper else cfg["max_omega_rad_s"] - y[0]

            initial_rate = rhs(start, state)[0]
            stationary_lower = state[0] == cfg["min_omega_rad_s"] and initial_rate == 0
            stationary_upper = state[0] == cfg["max_omega_rad_s"] and initial_rate == 0
            if (
                state[0] == cfg["min_omega_rad_s"]
                and initial_rate < 0
                or state[0] == cfg["max_omega_rad_s"]
                and initial_rate > 0
            ):
                boundary = dict(t=start, limit="lower" if initial_rate < 0 else "upper")
                break
            lower.terminal = upper.terminal = True
            lower.direction = upper.direction = -1
            result = solve_ivp(
                rhs,
                (start, end),
                state,
                method="DOP853",
                rtol=1e-12,
                atol=1e-10,
                max_step=0.1,
                dense_output=True,
                events=[lower, upper],
            )
            if not result.success:
                raise ValueError(result.message)
            if result.t[-1] > start:
                segments.append((start, result.t[-1], result.sol))
            state = result.y[:, -1]
            for index, events in enumerate(result.t_events):
                if len(events):
                    boundary = dict(t=float(events[0]), limit="lower" if index == 0 else "upper")
            if boundary:
                break
        start, active = end, command["controls"]

    def at(stamp):
        if stamp == 0:
            return initial.copy()
        for start, end, solution in segments:
            if start <= stamp <= end + 1e-8:
                return solution(min(end, stamp))
        raise ValueError("Turbine reference time outside accepted domain")

    return at, boundary
