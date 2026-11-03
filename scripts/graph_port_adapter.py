"""Typed nonlinear graph residual; no time integration or equipment synthesis.

Pressure and edge flows remain explicit algebraic variables in residual calls.
Only initialize solves the pre-existing quasi-steady hydraulic problem. Endpoint
flows are positive into the owning component, with explicit enthalpy convention.
"""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.network import NetworkError, PortSpec, RowSpec, StateSpec, number
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.nonlinear_thermal_graph import signature as kernel_signature
from scripts.solve_hydraulic_graph import sha
from scripts.solve_resistance_feedback import effective_input

BUDGET = Path("validation/ai05_port_adapter_budget_v1.json")


def signature():
    value = kernel_signature()
    value["drivers"]["scripts/graph_port_adapter.py"] = sha(__file__)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class GraphPortAdapter:
    def __init__(self, spec):
        self._kernel = NonlinearThermalGraph(deepcopy(spec))
        self._input = deepcopy(spec)
        self.identity = digest(spec)
        self.captured = signature()
        self.budget = json.loads(BUDGET.read_text())
        k = self._kernel
        self._names = tuple(k.names)
        self._edges = deepcopy(k.graph["edges"])
        self._n, self._m = len(k.names), len(self._edges)
        self._size = 2 * self._n + self._m + 1
        self._port = PortSpec(
            "fluid",
            "hydraulic",
            medium=k.cfg["medium"],
            energy_convention="specific_enthalpy_transport",
        )
        self._variables, self._rows = [], []
        for name in self._names:
            self._variable("node." + name, StateSpec("temperature_k", "K", True, 300.0))
            self._row(
                "node." + name,
                RowSpec("energy", "W", 1e6, self.budget["maximum_energy_residual_w"]),
            )
        for node in k.graph["nodes"]:
            self._variable("node." + node["id"], StateSpec("pressure_pa", "Pa", False, 1e6))
            fixed = "pressure_pa" in node
            self._row(
                "node." + node["id"],
                RowSpec(
                    "pressure" if fixed else "mass",
                    "Pa" if fixed else "kg/s",
                    1e6 if fixed else 1.0,
                    self.budget["maximum_pressure_residual_pa"]
                    if fixed
                    else self.budget["maximum_mass_residual_kg_s"],
                ),
            )
        for edge in self._edges:
            self._variable("edge." + edge["id"], StateSpec("mass_flow_kg_s", "kg/s", False, 1.0))
            self._row(
                "edge." + edge["id"],
                RowSpec("momentum", "Pa", 1e6, self.budget["maximum_pressure_residual_pa"]),
            )
        self._variable("audit", StateSpec("external_energy_j", "J", True, 1e6))
        self._row("audit", RowSpec("energy", "W", 1e6, self.budget["maximum_energy_residual_w"]))

    def _variable(self, owner, spec):
        self._variables.append(dict(path=owner + "." + spec.name, **asdict(spec)))

    def _row(self, owner, spec):
        self._rows.append(dict(path=owner + "." + spec.name, **asdict(spec)))

    def contract(self):
        k = self._kernel
        components = [
            dict(id="node." + name, kind="finite_thermal_stock", capacity_j_k=float(k.capacity[i]))
            for i, name in enumerate(self._names)
        ]
        components.extend(
            dict(
                id="edge." + e["id"],
                kind="monotone_resistance_with_prescribed_pressure_gain",
                parameters=deepcopy(e),
            )
            for e in self._edges
        )
        components.extend(
            dict(
                id="boundary." + n["id"],
                kind="fixed_pressure" if "pressure_pa" in n else "fixed_mass_injection",
                parameters=deepcopy(n),
            )
            for n in k.graph["nodes"]
        )
        links = []
        for edge in self._edges:
            for side in ("from", "to"):
                links.append(
                    dict(
                        id="edge." + edge["id"] + "." + side,
                        left="edge." + edge["id"] + "." + side,
                        right="node." + edge[side] + ".edge_" + edge["id"],
                    )
                )
        links.extend(
            dict(
                id="external." + name,
                left="boundary." + name + ".fluid",
                right="node." + name + ".external",
            )
            for name in self._names
        )
        ports = {
            endpoint: asdict(
                PortSpec(
                    endpoint.rsplit(".", 1)[1],
                    "hydraulic",
                    medium=self._port.medium,
                    energy_convention=self._port.energy_convention,
                )
            )
            for link in links
            for endpoint in (link["left"], link["right"])
        }
        value = dict(
            schema_version=1,
            input_sha256=self.identity,
            variables=deepcopy(self._variables),
            rows=deepcopy(self._rows),
            components=components,
            links=links,
            ports=ports,
            units={field: unit for field, unit, _, _ in self._port.fields},
            flow_convention="positive_into_owner",
            algebraic_count=self._n + self._m,
            general_component_backend=False,
        )
        return dict(**value, contract_sha256=digest(value))

    def mass_matrix(self):
        diagonal = np.zeros(self._size)
        diagonal[: self._n] = self._kernel.capacity
        diagonal[-1] = 1.0
        return np.diag(diagonal)

    def _vector(self, value, label):
        try:
            array = np.asarray(value, dtype=object)
        except (TypeError, ValueError) as error:
            raise NetworkError("Invalid " + label + " vector") from error
        if array.shape != (self._size,):
            raise NetworkError("Invalid " + label + " shape")
        return np.array([number(v, label) for v in array], dtype=float)

    def _evaluate(self, state):
        k = self._kernel
        n = self._n
        t, p, q = state[:n], state[n : 2 * n], state[2 * n : -1]
        if np.any(t <= k._lower) or np.any(t >= k._upper):
            raise NetworkError("Adapter temperature exceeds open domain")
        for i, node in enumerate(k.graph["nodes"]):
            if (
                not node.get("min_pressure_pa", 0.0)
                < p[i]
                < node.get("max_pressure_pa", float("inf"))
            ):
                raise NetworkError("Adapter pressure exceeds open domain")
        effective, _ = effective_input(k.base, k.graph, dict(zip(self._names, t)), k.laws)
        energy = np.zeros(n)
        outgoing = k._a @ q
        injection = np.array(
            [
                outgoing[i] if "pressure_pa" in node else node["injection_kg_s"]
                for i, node in enumerate(k.graph["nodes"])
            ]
        )
        hydraulic = np.array(
            [
                p[i] - node["pressure_pa"]
                if "pressure_pa" in node
                else outgoing[i] - node["injection_kg_s"]
                for i, node in enumerate(k.graph["nodes"])
            ]
        )
        momentum = []
        ports = {}
        mechanical = 0.0

        def pair(left, right, pressure, mass, power, stream):
            for endpoint, sign in ((left, 1.0), (right, -1.0)):
                ports[endpoint] = dict(
                    pressure_pa=float(pressure),
                    mass_flow_kg_s=float(sign * mass),
                    energy_flow_w=float(sign * power),
                    stream_temperature_k=None if stream is None else float(stream),
                )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for j, edge in enumerate(effective["hydraulic"]["edges"]):
                    a, b = k.index[edge["from"]], k.index[edge["to"]]
                    flow = q[j]
                    linear = edge["linear_pa_s_kg"]
                    quadratic = edge["quadratic_pa_s2_kg2"]
                    loss = (linear * flow**2 + quadratic * abs(flow) ** 3) / k.rho
                    donor = a if flow >= 0 else b
                    outlet = t[donor] + (linear * abs(flow) + quadratic * flow**2) / (k.rho * k.cp)
                    if flow != 0 and not k._lower < outlet < k._upper:
                        raise NetworkError("Adapter stream exceeds open domain")
                    h = k.cp * (t[donor] - k.ref)
                    left = flow * (h + p[a] / k.rho) - (loss if flow < 0 else 0.0)
                    right = -flow * (h + p[b] / k.rho) - (loss if flow >= 0 else 0.0)
                    energy[a] -= left
                    energy[b] -= right
                    for side, i, mass, power in [("from", a, flow, left), ("to", b, -flow, right)]:
                        stream = None if flow == 0 else t[donor] if i == donor else outlet
                        pair(
                            "edge." + edge["id"] + "." + side,
                            "node." + edge[side] + ".edge_" + edge["id"],
                            p[i],
                            mass,
                            power,
                            stream,
                        )
                    gain = edge["pressure_gain_pa"] - k.rho * k.graph["gravity_m_s2"] * (
                        k.graph["nodes"][b]["elevation_m"] - k.graph["nodes"][a]["elevation_m"]
                    )
                    momentum.append(
                        p[a] - p[b] + gain - linear * flow - quadratic * flow * abs(flow)
                    )
                    mechanical += flow * gain / k.rho
                external = 0.0
                for i, name in enumerate(self._names):
                    cfg = k.cfg["nodes"][name]
                    mass = injection[i]
                    supply = cfg["supply_temperature_k"] if mass > 0 else t[i]
                    power = mass * (k.cp * (supply - k.ref) + p[i] / k.rho)
                    net = (
                        power
                        + cfg["heat_power_w"]
                        - cfg["ambient_ua_w_k"] * (t[i] - cfg["ambient_temperature_k"])
                    )
                    energy[i] += net
                    external += net
                    pair(
                        "node." + name + ".external",
                        "boundary." + name + ".fluid",
                        p[i],
                        mass,
                        power,
                        None if mass == 0 else supply,
                    )
                rhs = np.r_[energy, -hydraulic, -np.array(momentum), external + mechanical]
            except (FloatingPointError, OverflowError) as error:
                raise NetworkError("Nonfinite adapter arithmetic") from error
        if not np.all(np.isfinite(rhs)):
            raise NetworkError("Nonfinite adapter residual")
        return rhs, ports

    def residual(self, state, rate):
        y = self._vector(state, "state")
        dy = self._vector(rate, "rate")
        rhs, _ = self._evaluate(y)
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.mass_matrix() @ dy - rhs
        if not np.all(np.isfinite(result)):
            raise NetworkError("Nonfinite adapter residual")
        return result

    def ports(self, state):
        _, ports = self._evaluate(self._vector(state, "state"))
        return ports

    def initialize(self, temperatures_k=None):
        if signature() != self.captured:
            raise NetworkError("Adapter source/budget drift")
        k = self._kernel
        rates = k.rates(temperatures_k)
        temperatures = (
            dict(zip(self._names, k.offset)) if temperatures_k is None else temperatures_k
        )
        state = np.array(
            [temperatures[name] for name in self._names]
            + [rates["pressures_pa"][name] for name in self._names]
            + [rates["flows_kg_s"][e["id"]] for e in self._edges]
            + [0.0]
        )
        rate = np.zeros(self._size)
        rate[: self._n] = [rates["temperature_rates_k_s"][name] for name in self._names]
        rate[-1] = rates["external_power_w"]
        return self._package(state, rate, 0.0)

    def algebraic_jacobian(self, state):
        y = self._vector(state, "state")
        self._evaluate(y)
        k = self._kernel
        n = self._n
        effective, _ = effective_input(k.base, k.graph, dict(zip(self._names, y[:n])), k.laws)
        jacobian = np.zeros((n + self._m, n + self._m))
        for i, node in enumerate(k.graph["nodes"]):
            if "pressure_pa" in node:
                jacobian[i, i] = 1.0
            else:
                jacobian[i, n:] = k._a[i]
        for j, edge in enumerate(effective["hydraulic"]["edges"]):
            jacobian[n + j, k.index[edge["from"]]] = 1.0
            jacobian[n + j, k.index[edge["to"]]] = -1.0
            jacobian[n + j, n + j] = -edge["linear_pa_s_kg"] - 2.0 * edge[
                "quadratic_pa_s2_kg2"
            ] * abs(y[2 * n + j])
        return jacobian

    def lift_checkpoint(self, checkpoint):
        if signature() != self.captured:
            raise NetworkError("Adapter source/budget drift")
        kernel = NonlinearThermalGraph(deepcopy(self._input))
        kernel.restore(deepcopy(checkpoint))
        snapshot = kernel.snapshot()
        state = np.array(
            [snapshot["temperatures_k"][name] for name in self._names]
            + [snapshot["pressures_pa"][name] for name in self._names]
            + [snapshot["flows_kg_s"][edge["id"]] for edge in self._edges]
            + [snapshot["integrated_external_energy_j"]]
        )
        rhs, _ = self._evaluate(state)
        rate = np.zeros(self._size)
        rate[: self._n] = rhs[: self._n] / kernel.capacity
        rate[-1] = rhs[-1]
        return self._package(state, rate, snapshot["time_s"])

    def _package(self, state, rate, time):
        residual = self.residual(state, rate)
        if any(abs(v) > r["tolerance"] for v, r in zip(residual, self._rows)):
            raise NetworkError("Adapter initialization residual exceeds dimensional budget")
        jacobian = self.algebraic_jacobian(state)
        columns = np.array([v["nominal"] for v in self._variables[self._n : -1]])
        rows = np.array([r["nominal"] for r in self._rows[self._n : -1]])
        condition = float(np.linalg.cond(jacobian * columns / rows[:, None]))
        if (
            not np.isfinite(condition)
            or condition > self.budget["maximum_scaled_algebraic_condition"]
        ):
            raise NetworkError("Adapter algebraic block is numerically unresolved")
        result = dict(
            state=state.tolist(),
            rate=rate.tolist(),
            residual=residual.tolist(),
            ports=self.ports(state),
            input_sha256=self.identity,
            contract_sha256=self.contract()["contract_sha256"],
            time_s=time,
            maximum_scaled_algebraic_condition=condition,
            warmup_steps=0,
            commands_changed=False,
            algebraic_derivatives_required=False,
        )
        if signature() != self.captured:
            raise NetworkError("Adapter sources changed during initialization")
        return result
