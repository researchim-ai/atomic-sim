"""Separate absolute-stock IVP with loop-assembled face fluxes and thermal equations.

Shares the strict IF97 adapter, not the production RHS, parser, reflection
implementation, ledger layout or shifted fluid/wall coordinates.
"""

import numpy as np
from scipy.integrate import solve_ivp

from atomic_model_hem import eos


def solve(spec, program, method):
    channel = spec["channel"]
    n = channel["geometry"]["cells"] + 2
    area = channel["geometry"]["area_m2"]
    wave = channel["wave_speed_bound_m_s"]
    walls = spec["walls"]
    w = len(walls)
    names = ["left"] + ["c" + str(i) for i in range(n - 2)] + ["right"]
    points = (
        [spec["terminals"]["left"]["initial"]]
        + channel["initial"]
        + [spec["terminals"]["right"]["initial"]]
    )
    volumes = np.array(
        [spec["terminals"]["left"]["volume_m3"]]
        + [area * channel["geometry"]["length_m"] / (n - 2)] * (n - 2)
        + [spec["terminals"]["right"]["volume_m3"]]
    )
    seeds = []
    vel = []
    for point in points:
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
        seeds.append(s)
        vel.append(point["velocity_m_s"])
    m0 = np.array([s.density_kg_m3 * volume for s, volume in zip(seeds, volumes)])
    v = np.array(vel)
    y = np.r_[
        m0,
        m0 * v,
        np.array([m * s.internal_energy_j_kg for m, s in zip(m0, seeds)]) + 0.5 * m0 * v * v,
        [wall["initial_temperature_k"] for wall in walls],
        np.zeros(5),
    ]
    ports = [names.index(wall["cell"]) for wall in walls]
    controls = dict(
        fluid={
            "left": spec["terminals"]["left"]["heat_power_w"],
            **{"c" + str(i): p for i, p in enumerate(channel["heat_power_w"])},
            "right": spec["terminals"]["right"]["heat_power_w"],
        },
        walls={wall["id"]: wall["heat_power_w"] for wall in walls},
    )

    def thermos(value):
        result = []
        for i in range(n):
            m, P, E = map(float, [value[i], value[n + i], value[2 * n + i]])
            result.append(
                eos.rho_u(
                    float(m / volumes[i]),
                    E / m - 0.5 * (P / m) ** 2,
                    pressure_seed_pa=seeds[i].pressure_pa,
                    temperature_seed_k=seeds[i].temperature_k,
                )
            )
        return result

    def rhs(t, value):
        states = thermos(value)
        flux = []
        result = np.zeros_like(value)
        for edge in range(n + 1):
            if edge in (0, n):
                i = 0 if edge == 0 else n - 1
                rho = value[i] / volumes[i]
                speed = value[n + i] / value[i]
                force = (
                    rho * speed * speed
                    + states[i].pressure_pa
                    + (-1 if edge == 0 else 1) * wave * rho * speed
                )
                flux.append(np.array([0.0, force, 0.0]))
            else:
                left, right = edge - 1, edge

                def face(i):
                    m, P, E = value[i], value[n + i], value[2 * n + i]
                    speed = P / m
                    p = states[i].pressure_pa
                    return np.array([m, P, E]) / volumes[i], np.array(
                        [P / volumes[i], P * speed / volumes[i] + p, speed * (E / volumes[i] + p)]
                    )

                wl, fl = face(left)
                wr, fr = face(right)
                flux.append((fl + fr - wave * (wr - wl)) / 2)
        for i in range(n):
            change = area * (flux[i] - flux[i + 1])
            for j in range(3):
                result[j * n + i] = change[j]
            result[2 * n + i] += controls["fluid"][names[i]]
        total_wall_heat = 0.0
        external_entropy = sum(
            controls["fluid"][name] / states[i].temperature_k for i, name in enumerate(names)
        )
        production = 0.0
        for j, wall in enumerate(walls):
            tw = value[3 * n + j]
            tf = states[ports[j]].temperature_k
            exchange = wall["conductance_w_k"] * (tw - tf)
            ambient = wall["ambient_ua_w_k"] * (wall["ambient_temperature_k"] - tw)
            external = controls["walls"][wall["id"]] + ambient
            result[2 * n + ports[j]] += exchange
            result[3 * n + j] = (external - exchange) / wall["capacity_j_k"]
            total_wall_heat += external
            external_entropy += (
                controls["walls"][wall["id"]] / tw + ambient / wall["ambient_temperature_k"]
            )
            production += wall["conductance_w_k"] * (tw - tf) ** 2 / (tw * tf) + wall[
                "ambient_ua_w_k"
            ] * (wall["ambient_temperature_k"] - tw) ** 2 / (tw * wall["ambient_temperature_k"])
        result[-5:] = [
            sum(controls["fluid"].values()),
            total_wall_heat,
            area * (flux[0][1] - flux[-1][1]),
            external_entropy,
            production,
        ]
        return result

    def observation(time, value):
        states = thermos(value)
        return dict(
            time_s=float(time),
            cells=[
                dict(
                    s.as_dict(),
                    mass_kg=float(value[i]),
                    momentum_kg_m_s=float(value[n + i]),
                    total_energy_j=float(value[2 * n + i]),
                )
                for i, s in enumerate(states)
            ],
            wall_temperatures_k=list(map(float, value[3 * n : 3 * n + w])),
            global_integrals=dict(
                zip(
                    [
                        "external_fluid_heat_j",
                        "external_wall_heat_j",
                        "endwall_impulse_kg_m_s",
                        "external_entropy_j_k",
                        "thermal_entropy_production_j_k",
                    ],
                    map(float, value[-5:]),
                )
            ),
        )

    time = 0.0
    observations = [observation(time, y)]
    for call in program:
        if call["command"] is not None:
            controls = call["command"]["controls"]
        target = time + call["duration_s"]
        solved = solve_ivp(
            rhs, (time, target), y, method=method, rtol=2e-11, atol=1e-10, max_step=1e-4
        )
        if not solved.success:
            raise ValueError("Finite-system reference IVP failed")
        y = solved.y[:, -1]
        time = target
        observations.append(observation(time, y))
    return observations
