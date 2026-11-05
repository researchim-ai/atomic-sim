"""Frozen synthetic inputs for active-chart endpoints, phase changes and pre-CHF."""

from copy import deepcopy

from atomic_model_hem_mixture.cases import baseline_request
from atomic_model_hem_qualified import closures, eos


def baseline(quality=0.5, velocity=0.001, cells=2, pressure=1e6):
    old = baseline_request(quality=quality, velocity=velocity)
    spec = old["model"]
    spec["max_step_s"] = 2e-5
    for node in spec["nodes"]:
        node["initial"]["pressure_pa"] = pressure
    for edge in spec["edges"]:
        edge["cells"] = cells
        edge["initial"] = [
            dict(kind="px", pressure_pa=pressure, quality=quality, velocity_m_s=velocity)
            for _ in range(cells)
        ]
        edge["heat_power_w"] = [0.0] * cells
    for wall in spec["walls"]:
        wall["initial_temperature_k"] = eos.px(pressure, quality).temperature_k
        wall["ambient_temperature_k"] = wall["initial_temperature_k"]
    laws = old["mixture_friction"]
    for law in laws.values():
        law["darcy_law"] = "churchill"
    return dict(schema_version=1, model=spec, friction=laws)


def matrix():
    liquid = baseline(0.0)
    vapor = baseline(1.0, -0.001)
    hot = baseline(0.999)
    hot["model"]["walls"][0].update(initial_temperature_k=600.0, conductance_w_k=500.0)
    liquid_exit = baseline(1e-8)
    liquid_exit["model"]["walls"][0].update(initial_temperature_k=600.0, conductance_w_k=500.0)
    condense = baseline(1.0)
    p = 1e6
    t = eos.px(p, 1.0).temperature_k + 0.001
    for node in condense["model"]["nodes"]:
        node["initial"] = dict(kind="pt", pressure_pa=p, temperature_k=t, velocity_m_s=0.0)
    for edge in condense["model"]["edges"]:
        edge["initial"] = [
            dict(kind="pt", pressure_pa=p, temperature_k=t, velocity_m_s=0.001)
            for _ in edge["initial"]
        ]
    condense["model"]["walls"][0].update(initial_temperature_k=350.0, conductance_w_k=500.0)
    high = baseline(0.0, pressure=15.5e6)
    for node in high["model"]["nodes"]:
        node["initial"] = dict(kind="pt", pressure_pa=15.5e6, temperature_k=600.0, velocity_m_s=0.0)
    for edge in high["model"]["edges"]:
        edge["initial"] = [
            dict(kind="pt", pressure_pa=15.5e6, temperature_k=600.0, velocity_m_s=0.001)
            for _ in edge["initial"]
        ]
    high["model"]["walls"][0]["initial_temperature_k"] = 600.0
    heatlow = boiling_case(1e6, 500.0, "biasi_inl")
    heathigh = boiling_case(15.5e6, 1500.0, "table1995")
    return [
        ("moving_saturated_liquid", liquid),
        ("moving_saturated_vapor", vapor),
        ("vapor_phase_exit", hot),
        ("liquid_phase_exit", liquid_exit),
        ("condensation_entry", condense),
        ("pwr_pressure_liquid", high),
        ("chen_biasi", heatlow),
        ("chen_table_high_pressure", heathigh),
    ]


def boiling_case(p, G, chf_model):
    request = baseline(0.05, pressure=p)
    s = eos.px(p, 0.05)
    for edge in request["model"]["edges"]:
        edge["diameter_m"] = 0.008
        for point in edge["initial"]:
            point["velocity_m_s"] = G / s.density_kg_m3
    request["model"]["walls"][0].update(
        initial_temperature_k=s.temperature_k + 2,
        conductance_w_k=2000.0,
        boiling=dict(
            scope={k: True for k in closures.HEAT_FLAGS},
            chf_model=chf_model,
            maximum_chf_fraction=0.8,
        ),
    )
    request["model"]["max_step_s"] = 2e-6
    return request


def program(name):
    durations = [2e-5, 3e-5, 5e-5] if name.startswith("chen") else [0.0002, 0.0003, 0.0005]
    return [dict(duration_s=d, command=None) for d in durations]


def programs():
    result = [(name, spec, program(name)) for name, spec in matrix()]
    request = baseline()
    calls = program("online")
    names = [n["id"] for n in request["model"]["nodes"]] + [
        e["id"] + "/" + str(i) for e in request["model"]["edges"] for i in range(e["cells"])
    ]
    on = dict(fluid={n: 100.0 for n in names}, walls={"w0": 1000.0})
    off = dict(fluid={n: 0.0 for n in names}, walls={"w0": 0.0})
    calls[1]["command"] = dict(controls=on, expected_revision=1, controller_state={"tick": 1})
    calls[2]["command"] = dict(controls=off, expected_revision=2, controller_state={"tick": 2})
    result.append(("forced_online_entropy", deepcopy(request), calls))
    return result
