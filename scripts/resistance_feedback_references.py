"""Independent simultaneous pressure/flow/enthalpy roots and exact fixtures."""

import math
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares, root

from scripts.hydraulic_graph_references import reference as hydraulic_reference
from scripts.thermal_graph_references import cases as thermal_cases


def cases():
    for name, base, temperatures in thermal_cases():
        selected = []
        if name in {str(n) + "_" + mode for n in (3, 8, 16, 32) for mode in ("linear", "mixed")}:
            selected = [-0.004, 0.004]
        if name == "3_linear":
            selected.append(0.0)
        if name == "3_near_zero":
            selected = [-0.004]
        if name == "two_supplies":
            selected = [-0.004, 0.004]
        for beta in selected:
            value = deepcopy(base)
            if name == "two_supplies":
                # This is a new explicit fixture: the internal mixer has zero
                # external mass injection, not a fixed-pressure reservoir.
                mixer = next(n for n in value["hydraulic"]["nodes"] if n["id"] == "mix")
                mixer.pop("pressure_pa")
                mixer["injection_kg_s"] = 0.0
            laws = {}
            for edge in value["hydraulic"]["edges"]:
                mean = (
                    temperatures["temperatures_k"][edge["from"]]
                    + temperatures["temperatures_k"][edge["to"]]
                ) * 0.5
                edge["linear_pa_s_kg"] /= math.exp(beta * (mean - 330.0))
                laws[edge["id"]] = dict(
                    kind="exponential_endpoint_mean_v1",
                    reference_temperature_k=330.0,
                    temperature_coefficient_per_k=beta,
                    minimum_multiplier=0.1,
                    maximum_multiplier=10.0,
                )
            exact = dict(**hydraulic_reference(base["hydraulic"]), **temperatures)
            yield (
                name + ("_negative" if beta < 0 else "_positive" if beta > 0 else "_constant"),
                dict(
                    format="atomic-resistance-feedback-steady",
                    schema_version=1,
                    steady_input=value,
                    resistance_laws=laws,
                    initial_temperatures_k={n: 335.0 for n in temperatures["temperatures_k"]},
                ),
                exact,
            )


def reference(spec):
    base = spec["steady_input"]
    graph = base["hydraulic"]
    cfg = base["thermal"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    names = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)
    m = len(edges)
    free = [i for i, v in enumerate(nodes) if "pressure_pa" not in v]
    k = len(free)
    anchor = next(v["pressure_pa"] for v in nodes if "pressure_pa" in v)
    incidence = np.zeros((n, m))
    source = []
    target = []
    linear = []
    quadratic = []
    gains = []
    beta = []
    tref = []
    for j, e in enumerate(edges):
        a, b = names[e["from"]], names[e["to"]]
        incidence[a, j] = 1
        incidence[b, j] = -1
        source.append(a)
        target.append(b)
        linear.append(e["linear_pa_s_kg"])
        quadratic.append(e["quadratic_pa_s2_kg2"])
        gains.append(
            e["pressure_gain_pa"]
            - graph["density_kg_m3"]
            * graph["gravity_m_s2"]
            * (nodes[b]["elevation_m"] - nodes[a]["elevation_m"])
        )
        law = spec["resistance_laws"][e["id"]]
        beta.append(law["temperature_coefficient_per_k"])
        tref.append(law["reference_temperature_k"])
    source = np.array(source)
    target = np.array(target)
    linear = np.array(linear)
    quadratic = np.array(quadratic)
    beta = np.array(beta)
    tref = np.array(tref)
    rho = graph["density_kg_m3"]
    cp = cfg["heat_capacity_j_kg_k"]
    href = cfg["enthalpy_reference_temperature_k"]
    supply = np.array([cfg["nodes"][v["id"]].get("supply_temperature_k", href) for v in nodes])
    heat = np.array([cfg["nodes"][v["id"]]["heat_power_w"] for v in nodes])
    ua = np.array([cfg["nodes"][v["id"]]["ambient_ua_w_k"] for v in nodes])
    ambient = np.array([cfg["nodes"][v["id"]]["ambient_temperature_k"] for v in nodes])
    fixed_pressure = np.array([v.get("pressure_pa", anchor) for v in nodes])
    injection = np.array([nodes[i]["injection_kg_s"] for i in free])

    def unpack(x):
        p = fixed_pressure.copy()
        p[free] = anchor + x[:k] * 1e5
        q = x[k : k + m] * 10
        temperature = 330.0 + x[k + m :] * 50
        return p, q, temperature

    def residual(x):
        p, q, t = unpack(x)
        effective = linear * np.exp(beta * ((t[source] + t[target]) * 0.5 - tref))
        mass = incidence[free] @ q - injection
        momentum = incidence.T @ p + gains - effective * q - quadratic * q * np.abs(q)
        donor = np.where(q >= 0, source, target)
        receiver = np.where(q >= 0, target, source)
        rate = np.abs(q)
        energy = np.zeros(n)
        loss = (effective * q * q + quadratic * np.abs(q) ** 3) / rho
        np.add.at(energy, donor, -rate * (cp * (t[donor] - href) + p[donor] / rho))
        np.add.at(energy, receiver, rate * (cp * (t[donor] - href) + p[receiver] / rho) + loss)
        external = incidence @ q
        external[free] = injection
        energy += (
            external * (cp * (np.where(external > 0, supply, t) - href) + p / rho)
            + heat
            - ua * (t - ambient)
        )
        return np.r_[mass / 10, momentum / 1e5, energy / 1e5]

    seed = hydraulic_reference(graph)
    x = np.r_[
        (np.array([seed["pressures_pa"][nodes[i]["id"]] for i in free]) - anchor) / 1e5,
        np.array([seed["flows_kg_s"][e["id"]] for e in edges]) / 10,
        np.full(n, 0.4),
    ]
    lower = np.r_[np.full(k + m, -np.inf), np.full(n, (cfg["min_temperature_k"] - 330.0) / 50)]
    upper = np.r_[np.full(k + m, np.inf), np.full(n, (cfg["max_temperature_k"] - 330.0) / 50)]
    solved = least_squares(
        residual,
        x,
        jac="3-point",
        bounds=(lower, upper),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=1000,
    )
    # Least-squares gradient stopping alone does not imply dimensional accuracy,
    # especially at the donor change near zero flow. Refine the full mixed root.
    polished = root(residual, solved.x, method="hybr", options=dict(xtol=1e-11, maxfev=10000))
    if np.linalg.norm(residual(polished.x)) < np.linalg.norm(residual(solved.x)):
        solved.x = polished.x
    errors = residual(solved.x)
    mass = float(np.max(np.abs(errors[:k]), initial=0)) * 10
    pressure = float(np.max(np.abs(errors[k : k + m]))) * 1e5
    energy = float(np.max(np.abs(errors[k + m :]))) * 1e5
    if not np.all(np.isfinite(solved.x)) or mass > 1e-9 or pressure > 1e-6 or energy > 1e-5:
        raise ValueError(
            f"Independent simultaneous feedback reference failed: mass={mass}, pressure={pressure}, energy={energy}"
        )
    p, q, t = unpack(solved.x)
    return dict(
        pressures_pa={v["id"]: float(p[i]) for i, v in enumerate(nodes)},
        flows_kg_s={v["id"]: float(q[i]) for i, v in enumerate(edges)},
        temperatures_k={v["id"]: float(t[i]) for i, v in enumerate(nodes)},
        mass_residual_kg_s=mass,
        pressure_residual_pa=pressure,
        energy_residual_w=energy,
    )
