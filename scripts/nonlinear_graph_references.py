"""Independent mixed hydraulics in two transient IVP solvers; no production RK."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.hydraulic_graph_references import reference as hydraulic_reference
from scripts.resistance_feedback_references import cases as feedback_cases
from scripts.thermal_graph_references import limiting_cases


def pipe_spec(beta=0.01, reverse=False):
    steady = dict(
        format="atomic-thermal-hydraulic-steady",
        schema_version=1,
        hydraulic=dict(
            format="atomic-hydraulic-steady",
            schema_version=1,
            density_kg_m3=1000.0,
            gravity_m_s2=9.81,
            nodes=[
                dict(id="a", elevation_m=0.0, pressure_pa=200000.0),
                dict(id="b", elevation_m=0.0, pressure_pa=201000.0 if reverse else 199000.0),
            ],
            edges=[
                dict(
                    id="pipe",
                    **{"from": "a", "to": "b"},
                    linear_pa_s_kg=1000.0,
                    quadratic_pa_s2_kg2=0.0,
                    pressure_gain_pa=0.0,
                )
            ],
        ),
        thermal=dict(
            medium="test-liquid",
            heat_capacity_j_kg_k=1000.0,
            enthalpy_reference_temperature_k=300.0,
            min_temperature_k=250.0,
            max_temperature_k=500.0,
            nodes={
                k: dict(
                    heat_power_w=0.0,
                    ambient_ua_w_k=0.0,
                    ambient_temperature_k=300.0,
                    supply_temperature_k=350.0,
                )
                for k in ("a", "b")
            },
        ),
    )
    return dict(
        format="atomic-nonlinear-thermal-graph",
        schema_version=1,
        steady_input=steady,
        resistance_laws=dict(pipe=law(beta, 300.0)),
        volumes_m3=dict(a=0.5, b=0.5),
        initial_temperatures_k=dict(a=310.0, b=320.0),
        max_step_s=0.5,
    )


def law(beta, tref=330.0):
    return dict(
        kind="exponential_endpoint_mean_v1",
        reference_temperature_k=tref,
        temperature_coefficient_per_k=beta,
        minimum_multiplier=0.1,
        maximum_multiplier=10.0,
    )


def cycle_spec(beta=0.01, adiabatic=False):
    _, steady, _ = next(c for c in limiting_cases() if c[0] == "driven_cycle")
    steady = deepcopy(steady)
    for node in steady["hydraulic"]["nodes"][1:]:
        node.pop("pressure_pa")
        node["injection_kg_s"] = 0.0
    if adiabatic:
        for node in steady["thermal"]["nodes"].values():
            node["ambient_ua_w_k"] = 0.0
    names = list(steady["thermal"]["nodes"])
    return dict(
        format="atomic-nonlinear-thermal-graph",
        schema_version=1,
        steady_input=steady,
        resistance_laws={e["id"]: law(beta, 300.0) for e in steady["hydraulic"]["edges"]},
        volumes_m3={k: 0.5 for k in names},
        initial_temperatures_k={k: 310.0 for k in names},
        max_step_s=0.5,
    )


def bridge_spec():
    spec = pipe_spec()
    steady = spec["steady_input"]
    graph = steady["hydraulic"]
    graph["nodes"] = [
        dict(id="supply", elevation_m=0.0, pressure_pa=200000.0),
        dict(id="a", elevation_m=0.0, injection_kg_s=0.0),
        dict(id="b", elevation_m=0.0, injection_kg_s=0.0),
        dict(id="outlet", elevation_m=0.0, pressure_pa=190000.0),
    ]
    data = [
        ("sa", "supply", "a", 500.0),
        ("ao", "a", "outlet", 1000.0),
        ("sb", "supply", "b", 1000.0),
        ("bo", "b", "outlet", 1000.0),
        ("bridge", "a", "b", 1000.0),
    ]
    graph["edges"] = [
        dict(
            id=name,
            **{"from": a, "to": b},
            linear_pa_s_kg=linear,
            quadratic_pa_s2_kg2=0.0,
            pressure_gain_pa=0.0,
        )
        for name, a, b, linear in data
    ]
    names = [n["id"] for n in graph["nodes"]]
    steady["thermal"]["nodes"] = {
        k: dict(
            heat_power_w=600000.0 if k == "a" else 0.0,
            ambient_ua_w_k=0.0,
            ambient_temperature_k=300.0,
            supply_temperature_k=300.0,
        )
        for k in names
    }
    spec.update(
        resistance_laws={
            e["id"]: law(0.02 if e["id"] == "sa" else 0.0, 300.0) for e in graph["edges"]
        },
        volumes_m3={k: 0.5 for k in names},
        initial_temperatures_k={k: 300.0 for k in names},
    )
    return spec


def cases():
    for name, beta, reverse in [
        ("pipe_negative", -0.01, False),
        ("pipe_constant", 0.0, False),
        ("pipe_positive", 0.01, False),
        ("pipe_reverse", 0.01, True),
    ]:
        yield name, pipe_spec(beta, reverse)
    yield "cycle_cooling", cycle_spec(-0.01)
    yield "cycle_adiabatic", cycle_spec(0.01, True)
    idle = pipe_spec()
    idle["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    for node in idle["steady_input"]["thermal"]["nodes"].values():
        node["heat_power_w"] = 1000.0
    yield "idle_heating", idle
    for name in ("two_supplies_positive", "8_mixed_negative"):
        _, point, exact = next(c for c in feedback_cases() if c[0] == name)
        names = list(point["initial_temperatures_k"])
        yield (
            name,
            dict(
                format="atomic-nonlinear-thermal-graph",
                schema_version=1,
                steady_input=point["steady_input"],
                resistance_laws=point["resistance_laws"],
                volumes_m3={k: 5.0 for k in names},
                initial_temperatures_k={
                    k: exact["temperatures_k"][k] + 3 * np.cos(i) for i, k in enumerate(names)
                },
                max_step_s=0.5,
            ),
        )
    yield "bridge_reversal", bridge_spec()


class IndependentSystem:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.base = spec["steady_input"]
        self.graph = self.base["hydraulic"]
        self.cfg = self.base["thermal"]
        self.names = [n["id"] for n in self.graph["nodes"]]
        self.index = {k: i for i, k in enumerate(self.names)}
        self.cp = self.cfg["heat_capacity_j_kg_k"]
        self.rho = self.graph["density_kg_m3"]
        self.href = self.cfg["enthalpy_reference_temperature_k"]
        self.capacity = np.array([self.rho * self.cp * spec["volumes_m3"][k] for k in self.names])
        self.scale = self.capacity.sum()
        self.initial = np.array([spec["initial_temperatures_k"][k] for k in self.names])

    def hydraulic(self, temperature):
        graph = deepcopy(self.graph)
        for e in graph["edges"]:
            law = self.spec["resistance_laws"][e["id"]]
            mean = (temperature[self.index[e["from"]]] + temperature[self.index[e["to"]]]) * 0.5
            factor = np.exp(
                law["temperature_coefficient_per_k"] * (mean - law["reference_temperature_k"])
            )
            if not law["minimum_multiplier"] <= factor <= law["maximum_multiplier"]:
                raise ValueError("Independent law domain exceeded")
            e["linear_pa_s_kg"] *= float(factor)
        return graph, hydraulic_reference(graph)

    def rhs(self, time, state):
        temperature = self.initial + state[:-1]
        graph, hyd = self.hydraulic(temperature)
        energy = np.zeros(len(temperature))
        actual = np.zeros(len(temperature))
        mechanical = 0.0
        for e in graph["edges"]:
            q = hyd["flows_kg_s"][e["id"]]
            a, b = self.index[e["from"]], self.index[e["to"]]
            actual[a] += q
            actual[b] -= q
            donor, receiver = (a, b) if q >= 0 else (b, a)
            rate = abs(q)
            p = hyd["pressures_pa"]
            loss = (e["linear_pa_s_kg"] * q * q + e["quadratic_pa_s2_kg2"] * rate**3) / self.rho
            sensible = self.cp * (temperature[donor] - self.href)
            energy[donor] -= rate * (sensible + p[self.names[donor]] / self.rho)
            energy[receiver] += rate * (sensible + p[self.names[receiver]] / self.rho) + loss
            mechanical += q * e["pressure_gain_pa"] / self.rho + self.graph["gravity_m_s2"] * q * (
                self.graph["nodes"][a]["elevation_m"] - self.graph["nodes"][b]["elevation_m"]
            )
        external = 0.0
        for i, node in enumerate(self.graph["nodes"]):
            cfg = self.cfg["nodes"][node["id"]]
            q = node.get("injection_kg_s", actual[i])
            inlet = cfg["supply_temperature_k"] if q > 0 else temperature[i]
            value = (
                q * (self.cp * (inlet - self.href) + hyd["pressures_pa"][node["id"]] / self.rho)
                + cfg["heat_power_w"]
                - cfg["ambient_ua_w_k"] * (temperature[i] - cfg["ambient_temperature_k"])
            )
            energy[i] += value
            external += value
        return np.r_[energy / self.capacity, (external + mechanical) / self.scale]


def reference(spec, method, times=(0.0, 30.0, 60.0, 120.0)):
    system = IndependentSystem(spec)
    solved = solve_ivp(
        system.rhs,
        (0.0, times[-1]),
        np.zeros(len(system.names) + 1),
        method=method,
        rtol=1e-11,
        atol=1e-12,
        max_step=0.5,
        t_eval=times,
    )
    if not solved.success or solved.t[-1] != times[-1] or not np.all(np.isfinite(solved.y)):
        raise ValueError("Independent nonlinear IVP failed")
    temperature = system.initial[:, None] + solved.y[:-1]
    flows = [system.hydraulic(t)[1]["flows_kg_s"] for t in temperature.T]
    return dict(
        times_s=list(times),
        temperatures_k=temperature.T.tolist(),
        flows_kg_s=flows,
        integrated_external_energy_j=(solved.y[-1] * system.scale).tolist(),
        energy_residual_j=(system.capacity @ solved.y[:-1] - solved.y[-1] * system.scale).tolist(),
    )
