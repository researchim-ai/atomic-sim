"""Independent power-loop/DOP853 and continuum references, no runtime model calls."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def powers(spec, temperature, boundary):
    cfg = spec["model"]
    n = cfg["cells"]
    t = np.asarray(temperature).reshape(3, n)
    q = np.zeros((3, n))
    external = np.zeros(4)  # stream a, stream b, heater, ambient (inward)
    capacity = np.array(
        [
            np.array(cfg["a_cell_mass_kg"]) * cfg["a_cp_j_kg_k"],
            np.array(cfg["b_cell_mass_kg"]) * cfg["b_cp_j_kg_k"],
            cfg["wall_capacity_j_k"],
        ]
    )
    for side_index, side in enumerate(("a", "b")):
        flow = boundary[f"{side}_mass_flow_kg_s"]
        for i in range(n):
            upstream = i - 1 if flow >= 0 else i + 1
            inlet = (
                boundary[f"{side}_inlet_temperature_k"]
                if not 0 <= upstream < n
                else t[side_index, upstream]
            )
            advection = abs(flow) * cfg[f"{side}_cp_j_kg_k"] * (inlet - t[side_index, i])
            q[side_index, i] += advection
            external[side_index] += advection
            film = cfg[f"{side}_wall_ua_w_k"][i] * (t[side_index, i] - t[2, i])
            q[side_index, i] -= film
            q[2, i] += film
    for i in range(n):
        ambient = cfg["ambient_ua_w_k"][i] * (cfg["ambient_temperature_k"] - t[2, i])
        q[2, i] += cfg["wall_power_w"][i] + ambient
        external[2] += cfg["wall_power_w"][i]
        external[3] += ambient
    return (q / capacity).ravel(), q.ravel(), external


def initial_temperature(spec):
    if spec["initialization"]["mode"] == "given":
        return np.concatenate(
            [spec["initialization"]["temperature_k"][k] for k in ("a", "b", "wall")]
        )
    n = spec["model"]["cells"]
    guess = np.full(
        3 * n,
        (spec["controls"]["a_inlet_temperature_k"] + spec["controls"]["b_inlet_temperature_k"]) / 2,
    )
    result = root(lambda t: powers(spec, t, spec["controls"])[1], guess, tol=1e-10)
    if max(abs(powers(spec, result.x, spec["controls"])[1])) > 1e-6:
        raise ValueError("Independent steady reference did not converge")
    return result.x


def reference(spec):
    n = spec["model"]["cells"]
    temperature = initial_temperature(spec)
    state = np.concatenate((temperature, np.zeros(3 * n + 4)))
    segments = []
    boundary = spec["controls"]
    start = 0.0
    for command in [*spec["timeline"], dict(t=spec["duration_s"], controls=None)]:
        end = command["t"]
        if end > start:
            active = dict(boundary)

            def rhs(_time, y, active=active):
                rate, net, ports = powers(spec, y[: 3 * n], active)
                return np.concatenate((rate, net, ports))

            result = solve_ivp(
                rhs,
                (start, end),
                state,
                method="DOP853",
                rtol=2e-12,
                atol=1e-10,
                max_step=0.5,
                dense_output=True,
            )
            if not result.success:
                raise ValueError(result.message)
            segments.append((start, end, result.sol))
            state = result.y[:, -1]
        start, boundary = end, command["controls"]

    def at(stamp):
        if stamp == 0:
            return np.concatenate((temperature, np.zeros(3 * n + 4)))
        for start, end, solution in segments:
            if start <= stamp <= end:
                return solution(stamp)
        raise ValueError("Reference time outside horizon")

    return at


def cocurrent_equal_capacity_outlets(spec):
    """Continuous 1D steady solution; both positive flows and equal m*cp."""
    cfg, boundary = spec["model"], spec["controls"]
    a = boundary["a_mass_flow_kg_s"] * cfg["a_cp_j_kg_k"]
    b = boundary["b_mass_flow_kg_s"] * cfg["b_cp_j_kg_k"]
    if a <= 0 or abs(a - b) > 1e-10 or any(cfg["ambient_ua_w_k"]) or any(cfg["wall_power_w"]):
        raise ValueError(
            "Continuum reference requires equal positive heat capacity rates and no sources"
        )
    ua = sum(1 / (1 / x + 1 / y) for x, y in zip(cfg["a_wall_ua_w_k"], cfg["b_wall_ua_w_k"]))
    ta, tb = boundary["a_inlet_temperature_k"], boundary["b_inlet_temperature_k"]
    mean, half_difference = (ta + tb) / 2, (ta - tb) / 2 * np.exp(-2 * ua / a)
    return np.array([mean + half_difference, mean - half_difference])
