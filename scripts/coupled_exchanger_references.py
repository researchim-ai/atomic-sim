"""Independent absolute-temperature DOP853 reference for two coupled pump cycles.

No production model, Jacobian, snapshots or integration helpers are called.
The reference integrates external heat/work and both internal heat ports itself.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def coefficients(spec):
    result = []
    offset = 0
    for side in ("a", "b"):
        raw = spec["loops"][side]
        model = raw["loop"]["model"]
        nodes = {node["id"]: node["parameters"] for node in model["topology"]["nodes"]}
        successors = {
            c["from"].split(".")[0]: c["to"].split(".")[0] for c in model["topology"]["connections"]
        }
        names = [model["reference"]["segment"]]
        while len(names) < len(nodes):
            names.append(successors[names[-1]])
        rho = model["fluid"]["density_kg_m3"]
        cp = model["fluid"]["heat_capacity_j_kg_k"]
        area = model["fluid"]["area_m2"]

        def values(key):
            return np.array([nodes[name][key] for name in names])

        n = len(names)
        result.append(
            dict(
                side=side,
                raw=raw,
                model=model,
                pump=raw["pump"],
                names=names,
                n=n,
                offset=offset,
                rho=rho,
                cp=cp,
                inertia=sum(values("length_m")) / area,
                resistance=values("loss_coefficient") / (2 * rho * area**2),
                capacity=rho * area * values("length_m") * cp,
                wall_capacity=values("wall_capacity_j_k"),
                ua=values("fluid_wall_ua_w_k"),
                ambient=values("ambient_ua_w_k"),
                heater=values("wall_power_w"),
                pump_index=names.index(raw["pump"]["segment"]),
            )
        )
        offset += 2 + 2 * n + 5
    return result, offset


def equations(spec, coeffs, wall_start, state, boundary):
    result = np.zeros_like(state)
    wall = state[wall_start : wall_start + len(spec["exchanger"]["cells"])]
    cell_power = np.array(
        [
            cell["wall_power_w"]
            + cell["ambient_ua_w_k"] * (spec["exchanger"]["ambient_temperature_k"] - wall[i])
            for i, cell in enumerate(spec["exchanger"]["cells"])
        ]
    )
    external_wall = float(sum(cell_power))
    heat_totals = []
    for data in coeffs:
        n, o = data["n"], data["offset"]
        flow, omega = state[o : o + 2]
        tf, tw = state[o + 2 : o + 2 + n], state[o + 2 + n : o + 2 + 2 * n]
        p = data["pump"]
        torque = boundary[f"{data['side']}_motor_torque_nm"]
        head = (p["displacement_m3_rad"] * omega - flow / data["rho"]) / p["leakage_m3_s_pa"]
        leak = p["leakage_m3_s_pa"] * head**2
        bearing = p["drag_nm_s_rad"] * omega**2
        fluid = np.zeros(n)
        casing = np.zeros(n)
        for i in range(n):
            donor = (i - 1 if flow >= 0 else i + 1) % n
            transfer = data["ua"][i] * (tw[i] - tf[i])
            fluid[i] = (
                data["cp"] * abs(flow) * (tf[donor] - tf[i])
                + data["resistance"][i] * abs(flow) ** 3 / data["rho"]
                + transfer
            )
            casing[i] = (
                data["heater"][i]
                - transfer
                + data["ambient"][i] * (data["model"]["ambient_temperature_k"] - tw[i])
            )
        casing[data["pump_index"]] += leak
        total = 0.0
        for i, cell in enumerate(spec["exchanger"]["cells"]):
            index = data["names"].index(cell[f"{data['side']}_segment"])
            heat = cell[f"{data['side']}_ua_w_k"] * (wall[i] - tf[index])
            fluid[index] += heat
            cell_power[i] -= heat
            total += heat
        heat_totals.append(total)
        result[o] = (head - sum(data["resistance"]) * flow * abs(flow)) / data["inertia"]
        result[o + 1] = (torque - p["displacement_m3_rad"] * head - p["drag_nm_s_rad"] * omega) / p[
            "inertia_kg_m2"
        ]
        result[o + 2 : o + 2 + n] = fluid / data["capacity"]
        result[o + 2 + n : o + 2 + 2 * n] = casing / data["wall_capacity"]
        result[o + 2 + 2 * n : o + 2 + 2 * n + 5] = [
            torque * omega,
            bearing,
            leak,
            head * flow / data["rho"],
            sum(data["heater"] + data["ambient"] * (data["model"]["ambient_temperature_k"] - tw)),
        ]
    for i, cell in enumerate(spec["exchanger"]["cells"]):
        result[wall_start + i] = cell_power[i] / cell["wall_capacity_j_k"]
    result[-3:] = [*heat_totals, external_wall]
    return result


def reference(spec):
    coeffs, wall_start = coefficients(spec)
    nwall = len(spec["exchanger"]["cells"])
    initial = np.zeros(wall_start + nwall + 3)
    dynamic = []
    for data in coeffs:
        n, o = data["n"], data["offset"]
        raw = data["raw"]
        init = raw["loop"]["initialization"]
        initial[o : o + 2] = [init["mass_flow_kg_s"], raw["initial_omega_rad_s"]]
        initial[o + 2 : o + 2 + n] = [init["fluid_temperature_k"][name] for name in data["names"]]
        initial[o + 2 + n : o + 2 + 2 * n] = [
            init["wall_temperature_k"][name] for name in data["names"]
        ]
        dynamic.extend(range(o, o + 2 + 2 * n))
    initial[wall_start : wall_start + nwall] = [
        cell["initial_wall_temperature_k"] for cell in spec["exchanger"]["cells"]
    ]
    dynamic.extend(range(wall_start, wall_start + nwall))
    boundary = {f"{d['side']}_motor_torque_nm": d["raw"]["motor_torque_nm"] for d in coeffs}
    if spec["initialization"]["mode"] == "steady":

        def residual(values):
            trial = initial.copy()
            trial[dynamic] = values
            return equations(spec, coeffs, wall_start, trial, boundary)[dynamic]

        solution = root(residual, initial[dynamic], tol=1e-11)
        if max(abs(residual(solution.x))) > 1e-10:
            raise ValueError("Independent coupled steady solve failed")
        initial[dynamic] = solution.x
    state = initial.copy()
    segments = []
    start = 0.0
    for command in [*spec["timeline"], dict(t=spec["duration_s"], controls=None)]:
        end = command["t"]
        if end > start:
            active = dict(boundary)
            result = solve_ivp(
                lambda t, y: equations(spec, coeffs, wall_start, y, active),
                (start, end),
                state,
                method="DOP853",
                rtol=2e-12,
                atol=1e-10,
                max_step=0.02,
                dense_output=True,
            )
            if not result.success:
                raise ValueError(result.message)
            segments.append((start, end, result.sol))
            state = result.y[:, -1]
        start, boundary = end, command["controls"]

    def at(stamp):
        if stamp == 0:
            return initial.copy()
        for start, end, solution in segments:
            if start <= stamp <= end:
                return solution(stamp)
        raise ValueError("Coupled reference outside horizon")

    return at, coeffs, wall_start
