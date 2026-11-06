"""Finite-volume/endwall/thermal-wall numerical qualification fixtures."""

from copy import deepcopy

from atomic_model_hem_flow.cases import uniform


def baseline(kind="mixture"):
    channel = uniform(kind=kind)
    point = deepcopy(channel["initial"][0])
    terminals = {
        side: dict(volume_m3=0.025, initial=deepcopy(point), heat_power_w=0.0)
        for side in ["left", "right"]
    }
    temperature = 453.0356323914666 if kind == "mixture" else 350.0 if kind == "liquid" else 600.0
    walls = [
        dict(
            id="w" + str(i),
            cell="c" + str(i),
            capacity_j_k=10000.0,
            initial_temperature_k=temperature,
            conductance_w_k=50.0,
            heat_power_w=0.0,
            ambient_ua_w_k=0.0,
            ambient_temperature_k=temperature,
        )
        for i in range(4)
    ]
    return dict(schema_version=1, channel=channel, terminals=terminals, walls=walls)


def matrix():
    yield "uniform_rest", baseline()
    s = baseline()
    s["terminals"]["left"]["initial"]["pressure_pa"] = 1.001e6
    s["terminals"]["right"]["initial"]["pressure_pa"] = 0.999e6
    yield "pressure_transfer", s
    s = baseline()
    s["terminals"]["left"]["initial"]["pressure_pa"] = 0.999e6
    s["terminals"]["right"]["initial"]["pressure_pa"] = 1.001e6
    yield "reverse_transfer", s
    s = baseline("liquid")
    for p in s["channel"]["initial"]:
        p["velocity_m_s"] = 0.001
    for p in s["terminals"].values():
        p["initial"]["velocity_m_s"] = 0.001
    for w in s["walls"]:
        w["conductance_w_k"] = 0.0
    yield "reflected_momentum", s
    s = baseline()
    s["terminals"]["left"]["volume_m3"] = 0.01
    s["terminals"]["right"]["volume_m3"] = 0.04
    s["terminals"]["left"]["initial"]["pressure_pa"] = 1.001e6
    yield "unequal_volumes", s
    s = baseline()
    for p in s["channel"]["initial"]:
        p["quality"] = 0.999
    for p in s["terminals"].values():
        p["initial"]["quality"] = 0.999
    for w in s["walls"]:
        w.update(initial_temperature_k=600.0, conductance_w_k=500.0)
    yield "wall_driven_evaporation", s
    s = baseline()
    for w in s["walls"]:
        w.update(heat_power_w=1000.0, ambient_ua_w_k=20.0, ambient_temperature_k=400.0)
    yield "wall_heater_ambient", s
    s = baseline()
    s["channel"]["heat_power_w"] = [1000.0, 2000.0, 3000.0, 4000.0]
    s["terminals"]["left"]["heat_power_w"] = -1000.0
    yield "mixed_external_heat", s


def program(spec, kind=None):
    fluid = {
        "left": spec["terminals"]["left"]["heat_power_w"],
        **{"c" + str(i): p for i, p in enumerate(spec["channel"]["heat_power_w"])},
        "right": spec["terminals"]["right"]["heat_power_w"],
    }
    walls = {w["id"]: w["heat_power_w"] for w in spec["walls"]}
    calls = [dict(duration_s=dt, command=None) for dt in [0.003, 0.006, 0.009]]
    if kind:
        for revision in [1, 2]:
            controls = dict(fluid=deepcopy(fluid), walls=deepcopy(walls))
            if revision == 1:
                if kind == "walls":
                    controls["walls"] = {name: 4000.0 for name in walls}
                else:
                    controls["fluid"] = {name: 500.0 for name in fluid}
            calls[revision]["command"] = dict(
                controls=controls, expected_revision=revision, controller_state={"tick": revision}
            )
    return calls


def programs():
    for name, spec in matrix():
        yield name, spec, program(spec)
    for kind in ["walls", "fluid"]:
        spec = baseline()
        yield "online_" + kind + "_power", spec, program(spec, kind)
