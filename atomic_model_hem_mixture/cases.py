"""Synthetic model-choice and phase-endpoint programs, not experimental data."""

from copy import deepcopy

from atomic_model_hem_mixture.friction import FLAGS
from atomic_model_hem_network.cases import baseline


def baseline_request(model="mcadams", quality=0.5, velocity=0.001, topology="parallel"):
    graph = baseline(topology, "mixture")
    for node in graph["nodes"]:
        node["initial"]["quality"] = quality
    for edge in graph["edges"]:
        for p in edge["initial"]:
            p.update(quality=quality, velocity_m_s=velocity)
    return dict(
        schema_version=1,
        model=graph,
        mixture_friction={
            e["id"]: dict(viscosity_model=model, scope={k: True for k in FLAGS})
            for e in graph["edges"]
        },
    )


def matrix():
    a = baseline_request()
    b = baseline_request("cicchitti")
    reverse = baseline_request(velocity=-0.001, topology="branch")
    liquid = baseline_request(quality=0, velocity=0.0)
    vapor = baseline_request(quality=1, velocity=0.0)
    turbulent = baseline_request(velocity=2)
    alternative = baseline_request("cicchitti", velocity=2)
    transition = baseline_request(velocity=0.15)
    for law in transition["mixture_friction"].values():
        law["darcy_law"] = "churchill"
    heated = baseline_request(quality=0.999)
    for law in heated["mixture_friction"].values():
        law["darcy_law"] = "churchill"
    heated["model"]["walls"][0].update(initial_temperature_k=600.0, conductance_w_k=500.0)
    return [
        ("mixture_mcadams", a),
        ("mixture_cicchitti", b),
        ("reverse_branch", reverse),
        ("saturated_liquid", liquid),
        ("saturated_vapor", vapor),
        ("turbulent_mcadams", turbulent),
        ("turbulent_cicchitti", alternative),
        ("heated_phase_exit", heated),
        ("transition_churchill", transition),
    ]


def program(name):
    durations = (
        [0.00005, 0.00005, 0.0001] if name.startswith("turbulent") else [0.0005, 0.0005, 0.001]
    )
    return [dict(duration_s=d, command=None) for d in durations]


def online():
    request = baseline_request()
    calls = program("online")
    from atomic_model_hem_mixture.model import MixtureHEMNetwork

    m = MixtureHEMNetwork(request)
    mask = m.control()
    mask["walls"]["w0"] = 1000.0
    calls[1]["command"] = dict(controls=mask, expected_revision=1, controller_state={"tick": 1})
    calls[2]["command"] = dict(
        controls=m.control(), expected_revision=2, controller_state={"tick": 2}
    )
    return "online_power", deepcopy(request), calls
