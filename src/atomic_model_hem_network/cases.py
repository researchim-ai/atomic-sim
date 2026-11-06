"""Small closed graph qualification requests; synthetic, not plant geometry."""

from copy import deepcopy


def point(kind="liquid", velocity=0.0):
    if kind == "liquid":
        return dict(kind="pt", pressure_pa=3e6, temperature_k=350.0, velocity_m_s=velocity)
    return dict(kind="px", pressure_pa=1e6, quality=0.5, velocity_m_s=velocity)


def baseline(topology="branch", kind="liquid"):
    pairs = {
        "branch": [("a", "b"), ("b", "c"), ("b", "d")],
        "loop": [("a", "b"), ("b", "c"), ("c", "a")],
        "parallel": [("a", "b"), ("a", "b")],
    }[topology]
    names = list(dict.fromkeys(x for pair in pairs for x in pair))
    nodes = [dict(id=n, volume_m3=0.02, initial=point(kind), heat_power_w=0.0) for n in names]
    edges = [
        dict(
            id="e" + str(i),
            from_node=a,
            to_node=b,
            cells=2,
            length_m=2.0,
            diameter_m=0.05,
            roughness_m=0.0,
            friction="none",
            initial=[point(kind), point(kind)],
            heat_power_w=[0.0, 0.0],
        )
        for i, (a, b) in enumerate(pairs)
    ]
    temp = 350.0 if kind == "liquid" else 453.0356323914666
    walls = [
        dict(
            id="w0",
            cell="e0/0",
            capacity_j_k=10000.0,
            initial_temperature_k=temp,
            conductance_w_k=50.0,
            heat_power_w=0.0,
            ambient_ua_w_k=0.0,
            ambient_temperature_k=temp,
        )
    ]
    return dict(
        schema_version=1,
        nodes=nodes,
        edges=edges,
        walls=walls,
        wave_speed_bound_m_s=2000.0,
        max_step_s=5e-5,
        cfl=0.3,
    )


def matrix():
    rest = baseline()
    pressure = deepcopy(rest)
    pressure["nodes"][0]["initial"]["pressure_pa"] += 10
    asymmetric = deepcopy(pressure)
    asymmetric["edges"][2]["diameter_m"] = 0.04
    loop = baseline("loop")
    loop["nodes"][0]["initial"]["pressure_pa"] += 10
    parallel = baseline("parallel")
    parallel["nodes"][0]["initial"]["pressure_pa"] += 10
    friction = baseline("parallel")
    for edge in friction["edges"]:
        edge["friction"] = "darcy"
        edge["friction_scope"] = dict(
            circular_straight=True,
            fully_developed=True,
            constant_properties=True,
            forced_convection=True,
            negligible_compressibility=True,
        )
        for p in edge["initial"]:
            p["velocity_m_s"] = 0.001
    mixture = baseline("branch", "mixture")
    mixture["nodes"][0]["initial"]["pressure_pa"] += 10
    heat = baseline("loop", "mixture")
    heat["walls"][0]["initial_temperature_k"] = 500.0
    heat["walls"][0]["conductance_w_k"] = 500.0
    ambient = deepcopy(rest)
    ambient["walls"][0].update(
        heat_power_w=1000.0, ambient_ua_w_k=20.0, ambient_temperature_k=400.0
    )
    laminar_heat = baseline("parallel")
    for edge in laminar_heat["edges"]:
        for p in edge["initial"]:
            p["velocity_m_s"] = 0.001
    laminar_heat["walls"][0].update(
        initial_temperature_k=400.0,
        conductance_w_k=100.0,
        correlation=dict(
            scope=dict(
                circular_straight=True,
                fully_developed=True,
                constant_properties=True,
                forced_convection=True,
                negligible_compressibility=True,
            ),
            constant_wall_temperature=True,
            thermally_developed=True,
        ),
    )
    turbulent_heat = deepcopy(laminar_heat)
    turbulent_heat["walls"][0]["conductance_w_k"] = 1e5
    for edge in turbulent_heat["edges"]:
        for p in edge["initial"]:
            p["velocity_m_s"] = 1.0
    return [
        ("branch_rest", rest),
        ("branch_pressure", pressure),
        ("unequal_branches", asymmetric),
        ("loop_pressure", loop),
        ("parallel_channels", parallel),
        ("laminar_drag", friction),
        ("mixture_branch", mixture),
        ("wall_exchange", heat),
        ("ambient_source", ambient),
        ("laminar_heat_transfer", laminar_heat),
        ("turbulent_heat_transfer", turbulent_heat),
    ]
