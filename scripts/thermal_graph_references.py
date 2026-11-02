"""Independent port-flux energy residuals and manufactured thermal graphs."""

from copy import deepcopy

import numpy as np
from scipy.optimize import root

from scripts.hydraulic_graph_references import cases as hydraulic_cases
from scripts.hydraulic_graph_references import reference as hydraulic_reference


def reference(spec):
    hydraulic = hydraulic_reference(spec["hydraulic"])
    cfg = spec["thermal"]
    cp = cfg["heat_capacity_j_kg_k"]
    ref = cfg["enthalpy_reference_temperature_k"]
    nodes = spec["hydraulic"]["nodes"]
    names = {n["id"]: i for i, n in enumerate(nodes)}
    rho = spec["hydraulic"]["density_kg_m3"]
    balances = {name: 0.0 for name in names}
    for edge in spec["hydraulic"]["edges"]:
        q = hydraulic["flows_kg_s"][edge["id"]]
        balances[edge["from"]] += q
        balances[edge["to"]] -= q
    for node in nodes:
        if "injection_kg_s" in node:
            balances[node["id"]] = node["injection_kg_s"]

    def residual(theta):
        energy = np.zeros(len(nodes))
        for edge in spec["hydraulic"]["edges"]:
            q = hydraulic["flows_kg_s"][edge["id"]]
            donor, receiver = (edge["from"], edge["to"]) if q >= 0 else (edge["to"], edge["from"])
            pressure = hydraulic["pressures_pa"]
            loss = (
                edge["linear_pa_s_kg"] * q * q + edge["quadratic_pa_s2_kg2"] * abs(q) ** 3
            ) / rho
            energy[names[donor]] -= abs(q) * (cp * theta[names[donor]] + pressure[donor] / rho)
            energy[names[receiver]] += (
                abs(q) * (cp * theta[names[donor]] + pressure[receiver] / rho) + loss
            )
        for name, i in names.items():
            node = cfg["nodes"][name]
            q = balances[name]
            incoming = node["supply_temperature_k"] - ref if q > 0 else theta[i]
            energy[i] += (
                q * (cp * incoming + hydraulic["pressures_pa"][name] / rho)
                + node["heat_power_w"]
                - node["ambient_ua_w_k"] * (theta[i] + ref - node["ambient_temperature_k"])
            )
        return energy

    solved = root(
        lambda x: residual(x) / 1e5,
        np.full(len(nodes), 20.0),
        method="hybr",
        options=dict(xtol=1e-11, maxfev=10000),
    )
    errors = residual(solved.x)
    if not np.all(np.isfinite(solved.x)) or np.max(np.abs(errors)) > 1e-5:
        raise ValueError("Independent conservative enthalpy reference failed its energy residual")
    return dict(
        temperatures_k={name: float(solved.x[i] + ref) for name, i in names.items()},
        energy_residual_w=float(np.max(np.abs(errors))),
    )


def cases():
    for name, hydraulic, exact in hydraulic_cases():
        cp, ref, rho = 4200.0, 300.0, hydraulic["density_kg_m3"]
        temperatures = {
            node["id"]: 350.0 + 10 * np.sin(i) for i, node in enumerate(hydraulic["nodes"])
        }
        thermal = {
            node["id"]: dict(
                heat_power_w=0.0,
                ambient_ua_w_k=100.0 + i,
                ambient_temperature_k=300.0,
                supply_temperature_k=320.0 + i % 5,
            )
            for i, node in enumerate(hydraulic["nodes"])
        }
        balances = {node["id"]: 0.0 for node in hydraulic["nodes"]}
        energies = deepcopy(balances)
        for edge in hydraulic["edges"]:
            m = exact["flows_kg_s"][edge["id"]]
            balances[edge["from"]] += m
            balances[edge["to"]] -= m
            donor, receiver = (edge["from"], edge["to"]) if m >= 0 else (edge["to"], edge["from"])
            loss = (
                edge["linear_pa_s_kg"] * m * m + edge["quadratic_pa_s2_kg2"] * abs(m) ** 3
            ) / rho
            energies[donor] -= abs(m) * (
                cp * (temperatures[donor] - ref) + exact["pressures_pa"][donor] / rho
            )
            energies[receiver] += (
                abs(m) * (cp * (temperatures[donor] - ref) + exact["pressures_pa"][receiver] / rho)
                + loss
            )
        for node in hydraulic["nodes"]:
            k = node["id"]
            m = node.get("injection_kg_s", balances[k])
            cfg = thermal[k]
            inlet = cfg["supply_temperature_k"] if m > 0 else temperatures[k]
            energies[k] += m * (cp * (inlet - ref) + exact["pressures_pa"][k] / rho) - cfg[
                "ambient_ua_w_k"
            ] * (temperatures[k] - cfg["ambient_temperature_k"])
            cfg["heat_power_w"] = -float(energies[k])
        yield (
            name,
            dict(
                format="atomic-thermal-hydraulic-steady",
                schema_version=1,
                hydraulic=hydraulic,
                thermal=dict(
                    medium="manufactured-liquid",
                    heat_capacity_j_kg_k=cp,
                    enthalpy_reference_temperature_k=ref,
                    min_temperature_k=250.0,
                    max_temperature_k=600.0,
                    nodes=thermal,
                ),
            ),
            dict(temperatures_k={k: float(v) for k, v in temperatures.items()}),
        )
    yield from limiting_cases()


def limiting_cases():
    for mode in ("driven_cycle", "zero_flow"):
        hydraulic = dict(
            format="atomic-hydraulic-steady",
            schema_version=1,
            density_kg_m3=1000,
            gravity_m_s2=9.81,
            nodes=[dict(id=n, elevation_m=0, pressure_pa=200000) for n in ("a", "b", "c")],
            edges=[
                dict(
                    id="e" + str(i),
                    **{"from": a, "to": b},
                    linear_pa_s_kg=1000,
                    quadratic_pa_s2_kg2=0,
                    pressure_gain_pa=3000 if mode == "driven_cycle" else 0,
                )
                for i, (a, b) in enumerate([("a", "b"), ("b", "c"), ("c", "a")])
            ],
        )
        thermal = dict(
            medium="limiting-liquid",
            heat_capacity_j_kg_k=1000,
            enthalpy_reference_temperature_k=300,
            min_temperature_k=250,
            max_temperature_k=500,
            nodes={
                n: dict(
                    heat_power_w=0,
                    ambient_ua_w_k=10,
                    ambient_temperature_k=300,
                    supply_temperature_k=300,
                )
                for n in ("a", "b", "c")
            },
        )
        yield (
            mode,
            dict(
                format="atomic-thermal-hydraulic-steady",
                schema_version=1,
                hydraulic=hydraulic,
                thermal=thermal,
            ),
            dict(
                temperatures_k={
                    n: 300.9 if mode == "driven_cycle" else 300.0 for n in ("a", "b", "c")
                }
            ),
        )
    hydraulic = dict(
        format="atomic-hydraulic-steady",
        schema_version=1,
        density_kg_m3=1000,
        gravity_m_s2=9.81,
        nodes=[
            dict(id=n, elevation_m=0, pressure_pa=p)
            for n, p in [("hot", 201200), ("cold", 201200), ("mix", 200000), ("outlet", 198800)]
        ],
        edges=[
            dict(
                id="e" + str(i),
                **{"from": a, "to": b},
                linear_pa_s_kg=r,
                quadratic_pa_s2_kg2=0,
                pressure_gain_pa=0,
            )
            for i, (a, b, r) in enumerate(
                [("hot", "mix", 1200), ("cold", "mix", 600), ("mix", "outlet", 400)]
            )
        ],
    )
    thermal = dict(
        medium="mixing-liquid",
        heat_capacity_j_kg_k=1000,
        enthalpy_reference_temperature_k=300,
        min_temperature_k=250,
        max_temperature_k=500,
        nodes={
            n: dict(
                heat_power_w=0,
                ambient_ua_w_k=0,
                ambient_temperature_k=300,
                supply_temperature_k=400 if n == "hot" else 300,
            )
            for n in ("hot", "cold", "mix", "outlet")
        },
    )
    mix = (400 + 2 * 300) / 3 + 0.0012
    yield (
        "two_supplies",
        dict(
            format="atomic-thermal-hydraulic-steady",
            schema_version=1,
            hydraulic=hydraulic,
            thermal=thermal,
        ),
        dict(temperatures_k=dict(hot=400, cold=300, mix=mix, outlet=mix + 0.0012)),
    )
