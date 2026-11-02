"""Independent conservative port-flux IVPs; no production matrix exponential."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.hydraulic_graph_references import reference as hydraulic_reference
from scripts.thermal_graph_references import cases as steady_cases


def cases():
    for name, steady, exact in steady_cases():
        names = list(steady["thermal"]["nodes"])
        yield (
            name,
            dict(
                format="atomic-thermal-graph-transient",
                schema_version=1,
                steady_input=steady,
                volumes_m3={name: 1.0 + 0.25 * (i % 3) for i, name in enumerate(names)},
                initial_temperatures_k={
                    name: exact["temperatures_k"][name] + 5 * np.cos(i)
                    for i, name in enumerate(names)
                },
                max_step_s=1.0,
            ),
        )
    name, steady, exact = next(c for c in steady_cases() if c[0] == "driven_cycle")
    steady = deepcopy(steady)
    for node in steady["thermal"]["nodes"].values():
        node["ambient_ua_w_k"] = 0
    yield (
        "adiabatic_cycle",
        dict(
            format="atomic-thermal-graph-transient",
            schema_version=1,
            steady_input=steady,
            volumes_m3={n: 1.0 for n in exact["temperatures_k"]},
            initial_temperatures_k={n: 310.0 for n in exact["temperatures_k"]},
            max_step_s=1.0,
        ),
    )


def reference(spec, method):
    graph = spec["steady_input"]["hydraulic"]
    cfg = spec["steady_input"]["thermal"]
    hyd = hydraulic_reference(graph)
    names = [n["id"] for n in graph["nodes"]]
    index = {n: i for i, n in enumerate(names)}
    n = len(names)
    rho = graph["density_kg_m3"]
    cp = cfg["heat_capacity_j_kg_k"]
    ref = cfg["enthalpy_reference_temperature_k"]
    gravity = graph["gravity_m_s2"]
    capacity = np.array([rho * cp * spec["volumes_m3"][k] for k in names])
    scale = capacity.sum()
    initial = np.array([spec["initial_temperatures_k"][k] for k in names])
    pressure = np.array([hyd["pressures_pa"][k] for k in names])
    donor = []
    receiver = []
    rates = []
    loss = []
    actual = np.zeros(n)
    mechanical = 0.0
    for edge in graph["edges"]:
        q = hyd["flows_kg_s"][edge["id"]]
        a = index[edge["from"]]
        b = index[edge["to"]]
        donor.append(a if q >= 0 else b)
        receiver.append(b if q >= 0 else a)
        rates.append(abs(q))
        loss.append(
            (edge["linear_pa_s_kg"] * q * q + edge["quadratic_pa_s2_kg2"] * abs(q) ** 3) / rho
        )
        actual[a] += q
        actual[b] -= q
        mechanical += q * edge["pressure_gain_pa"] / rho + gravity * q * (
            graph["nodes"][a]["elevation_m"] - graph["nodes"][b]["elevation_m"]
        )
    donor = np.array(donor)
    receiver = np.array(receiver)
    rates = np.array(rates)
    loss = np.array(loss)
    external = np.array(
        [node.get("injection_kg_s", actual[i]) for i, node in enumerate(graph["nodes"])]
    )
    ua = np.array([cfg["nodes"][k]["ambient_ua_w_k"] for k in names])
    heat = np.array([cfg["nodes"][k]["heat_power_w"] for k in names])
    ambient = np.array([cfg["nodes"][k]["ambient_temperature_k"] for k in names])
    supply = np.array([cfg["nodes"][k].get("supply_temperature_k", ref) for k in names])

    def rhs(t, state):
        temperature = initial + state[:n]
        energy = np.zeros(n)
        sensible = cp * (temperature[donor] - ref)
        np.add.at(energy, donor, -rates * (sensible + pressure[donor] / rho))
        np.add.at(energy, receiver, rates * (sensible + pressure[receiver] / rho) + loss)
        exchange = external * (
            cp * (np.where(external > 0, supply, temperature) - ref) + pressure / rho
        )
        boundary = exchange + heat - ua * (temperature - ambient)
        return np.r_[(energy + boundary) / capacity, (float(boundary.sum()) + mechanical) / scale]

    jac = np.zeros((n + 1, n + 1))
    for a, b, q in zip(donor, receiver, rates):
        jac[a, a] -= q * cp / capacity[a]
        jac[b, a] += q * cp / capacity[b]
    diag = np.minimum(external, 0) * cp - ua
    jac[np.arange(n), np.arange(n)] += diag / capacity
    jac[-1, :n] = diag / scale
    times = [0.0, 30.0, 60.0, 120.0]
    options = dict(method=method, rtol=1e-11, atol=1e-12, max_step=0.5, t_eval=times)
    if method == "Radau":
        options["jac"] = jac
    result = solve_ivp(rhs, (0.0, 120.0), np.zeros(n + 1), **options)
    if not result.success or result.t[-1] != 120.0 or not np.all(np.isfinite(result.y)):
        raise ValueError("Independent dynamic port-flux IVP failed")
    return dict(
        times_s=times,
        temperatures_k=(initial[:, None] + result.y[:n]).T.tolist(),
        integrated_external_energy_j=(result.y[-1] * scale).tolist(),
        energy_residual_j=(capacity @ result.y[:n] - result.y[-1] * scale).tolist(),
    )
