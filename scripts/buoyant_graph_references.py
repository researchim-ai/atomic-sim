"""Closed thermosyphon fixtures and independent full-flow bordered equations."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from scripts.inertial_graph_references import boundary, graph, junction, pipe
from scripts.thermal_inertial_references import (
    IndependentThermalInertial,
)
from scripts.thermal_inertial_references import (
    request as thermal_request,
)


def loop(
    nodes=6,
    *,
    mirror=False,
    heated=True,
    beta=0.001,
    initial_flow=0.0,
    hot_start=False,
    gravity=9.81,
    branch=False,
):
    elevations = [0, 5, 10, 10, 5, 0] if nodes == 6 else [0, 3, 7, 10, 10, 7, 3, 0]
    ns = [boundary("n0", 200000, height=elevations[0])] + [
        junction("n" + str(i), height=elevations[i]) for i in range(1, nodes)
    ]
    es = [
        pipe("e" + str(i), "n" + str(i), "n" + str((i + 1) % nodes), 10000, 1000, 0)
        for i in range(nodes)
    ]
    if branch:
        es.append(pipe("bypass", "n1", "n4", 5000, 2000, 0))
    q = {e["id"]: initial_flow for e in es}
    if branch:
        q["bypass"] = 0.0
    h = graph(ns, es, q, gravity)
    base = thermal_request(
        h, beta=0.002, recovery=1.0, heat=0, volume=0.002, initial=300.0, supply=300.0
    )
    for t in base["thermal"]["nodes"].values():
        t["ambient_ua_w_k"] = 0
        t["ambient_temperature_k"] = 300
        t["supply_temperature_k"] = 300
    heater, cooler = (nodes - 1, nodes // 2 - 1) if mirror else (0, nodes // 2)
    if heated:
        base["thermal"]["nodes"]["n" + str(heater)]["heat_power_w"] = 1000.0
        base["thermal"]["nodes"]["n" + str(cooler)]["ambient_ua_w_k"] = 100.0
    if hot_start:
        for i in range(nodes // 2):
            base["thermal"]["nodes"]["n" + str(i)]["initial_temperature_k"] = 310.0
    return dict(
        format="atomic-buoyant-inertial-graph",
        schema_version=1,
        base=base,
        buoyancy=dict(
            expansion_per_k=beta,
            reference_temperature_k=300.0,
            maximum_relative_density_change=0.05,
        ),
    )


def cases():
    yield "hydrostatic_rest", loop(heated=False), []
    r = loop(heated=False)
    for n in r["base"]["thermal"]["nodes"].values():
        n["heat_power_w"] = 1000.0
    yield "uniform_heating", r, []
    yield "heated_startup", loop(), []
    yield "mirror_startup", loop(mirror=True), []
    yield "opposed_hot_start", loop(initial_flow=-0.01, hot_start=True), []
    yield "gravity_off", loop(gravity=0), []
    yield "expansion_off", loop(beta=0), []
    yield "eight_nodes", loop(nodes=8), []
    yield (
        "branched_gauge",
        loop(branch=True),
        [(10.0, dict(n0=210000.0)), (25.0, dict(n0=200000.0))],
    )


class IndependentBuoyant(IndependentThermalInertial):
    def __init__(self, request):
        super().__init__(request["base"])
        self.buoyancy = deepcopy(request["buoyancy"])
        self.initial = np.r_[self.initial, 0.0]

    def force(self, T):
        return np.array(
            [
                self.h.rho
                * self.h.g
                * self.buoyancy["expansion_per_k"]
                * (
                    0.5 * (T[self.h.names.index(e["from"])] + T[self.h.names.index(e["to"])])
                    - self.buoyancy["reference_temperature_k"]
                )
                * (self.h.z[self.h.names.index(e["to"])] - self.h.z[self.h.names.index(e["from"])])
                for e in self.h.edges
            ]
        )

    def rhs(self, time, y, boundaries):
        h = self.h
        q = y[: self.m]
        T = self.initialT + y[self.m : self.m + self.n]
        p = np.zeros(self.n)
        p[h.fixed] = [boundaries[h.names[i]] for i in h.fixed]
        buoyancy_force = self.force(T)
        buoyancy_work = float(q @ buoyancy_force / h.rho)
        loss = np.empty(self.m)
        heat = np.zeros(self.n)
        fluid = 0.0
        bath = 0.0
        for j, e in enumerate(h.edges):
            a = h.names.index(e["from"])
            b = h.names.index(e["to"])
            law = self.spec["edge_laws"][e["id"]]
            factor = np.exp(
                law["temperature_coefficient_per_k"]
                * ((T[a] + T[b]) / 2 - law["reference_temperature_k"])
            )
            loss[j] = factor * (h.r[j] * q[j] + h.k[j] * q[j] * abs(q[j]))
            upstream, downstream = (a, b) if q[j] >= 0 else (b, a)
            energy = abs(q[j]) * self.cp * (T[upstream] - self.ref)
            heat[upstream] -= energy
            heat[downstream] += energy
            dissipated = q[j] * loss[j] / h.rho
            recovered = dissipated * law["fluid_heat_fraction"]
            heat[a] -= q[j] * buoyancy_force[j] / h.rho / 2
            heat[b] -= q[j] * buoyancy_force[j] / h.rho / 2
            heat[a] += recovered / 2
            heat[b] += recovered / 2
            fluid += recovered
            bath += dissipated - recovered
        forcing = buoyancy_force - h.incidence.T @ (p + h.rho * h.g * h.z) - loss
        solution = np.linalg.solve(h.matrix, np.r_[forcing, np.zeros(len(h.free))])
        p[h.free] = solution[self.m :]
        external = -h.incidence @ q
        P = sum(p[i] * external[i] / h.rho for i in range(self.n))
        G = sum(h.g * h.z[i] * external[i] for i in range(self.n))
        H = 0.0
        Q = 0.0
        for i, name in enumerate(h.names):
            node = self.spec["thermal"]["nodes"][name]
            mass = external[i] if i in h.fixed else h.nodes[i]["injection_kg_s"]
            value = (
                mass * self.cp * ((node["supply_temperature_k"] if mass >= 0 else T[i]) - self.ref)
            )
            source = node["heat_power_w"] - node["ambient_ua_w_k"] * (
                T[i] - node["ambient_temperature_k"]
            )
            heat[i] += value + source
            H += value
            Q += source
        return np.r_[
            solution[: self.m], heat / self.capacity, [P, G, Q, H, fluid, bath, buoyancy_work]
        ], p

    def integrate(self, horizon, schedule, method, times):
        state = self.initial.copy()
        boundaries = deepcopy(self.h.boundaries)
        current = 0.0
        rows = []
        events = {t: None for t in times}
        events[horizon] = None
        events.update(dict(schedule))
        evaluations = 0

        def fun(t, y):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 200000:
                raise ValueError("Independent coupled RHS budget exhausted")
            return self.rhs(t, y, boundaries)[0]

        def jac(t, y):
            result = np.zeros((len(y), len(y)))
            # Ledger columns are exactly zero; do not adapt perturbations there.
            for j in range(self.m + self.n):
                h = 1e-6 * max(1.0, abs(y[j]))
                up, down = y.copy(), y.copy()
                up[j] += h
                down[j] -= h
                result[:, j] = (fun(t, up) - fun(t, down)) / (2 * h)
            return result

        for end, change in sorted(events.items()):
            sol = solve_ivp(
                fun,
                (current, end),
                state,
                method=method,
                rtol=2e-12,
                atol=np.r_[np.full(self.m + self.n, 1e-12), np.full(7, 1e-8)],
                max_step=0.1,
                **({"jac": jac} if method == "Radau" else {}),
            )
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise ValueError("Independent thermal inertial IVP failed")
            state = sol.y[:, -1]
            current = end
            if change is not None:
                boundaries = deepcopy(change)
            _, p = self.rhs(current, state, boundaries)
            rows.append(
                dict(
                    time_s=end,
                    flows_kg_s=dict(
                        zip([e["id"] for e in self.h.edges], map(float, state[: self.m]))
                    ),
                    pressures_pa=dict(zip(self.h.names, map(float, p))),
                    temperatures_k=dict(
                        zip(
                            self.h.names,
                            map(float, self.initialT + state[self.m : self.m + self.n]),
                        )
                    ),
                    ledgers=state[-7:].tolist(),
                )
            )
        return rows

    def stationary(self, guess_flow=0.13):
        """Full steady momentum+mass+thermal solve, with an explicit branch seed."""
        h = self.h
        f = len(h.free)
        reference = h.boundaries[h.names[h.fixed[0]]] - h.rho * h.g * (h.z - h.z[h.fixed[0]])
        heat = sum(n["heat_power_w"] for n in self.spec["thermal"]["nodes"].values())
        cooling = [n for n in self.spec["thermal"]["nodes"].values() if n["ambient_ua_w_k"] > 0]
        cold = 300 + heat / sum(n["ambient_ua_w_k"] for n in cooling)
        T = np.full(self.n, cold)
        if guess_flow > 0:
            T[: self.n // 2] += heat / (self.cp * guess_flow)
        else:
            T[self.n // 2 :] += heat / (self.cp * abs(guess_flow))
        initial = np.r_[np.full(self.m, guess_flow), np.zeros(f), T - self.initialT]

        def equations(x):
            q = x[: self.m]
            temp = self.initialT + x[self.m + f :]
            p = reference.copy()
            p[h.free] += 1000 * x[self.m : self.m + f]
            loss = np.array(
                [
                    np.exp(
                        self.spec["edge_laws"][e["id"]]["temperature_coefficient_per_k"]
                        * (
                            (temp[h.names.index(e["from"])] + temp[h.names.index(e["to"])]) / 2
                            - self.spec["edge_laws"][e["id"]]["reference_temperature_k"]
                        )
                    )
                    * (h.r[j] * q[j] + h.k[j] * q[j] * abs(q[j]))
                    for j, e in enumerate(h.edges)
                ]
            )
            momentum = h.incidence.T @ (p + h.rho * h.g * h.z) + loss - self.force(temp)
            rate, _ = self.rhs(0, np.r_[q, temp - self.initialT, np.zeros(7)], h.boundaries)
            return np.r_[
                momentum / 1000,
                h.incidence[h.free] @ q,
                rate[self.m : self.m + self.n] * self.capacity / 1000,
            ]

        result = root(equations, initial, tol=1e-10)
        residual = equations(result.x)
        if not result.success or max(abs(residual)) > 1e-8:
            raise ValueError("Independent nontrivial stationary branch solve failed")
        q = result.x[: self.m]
        T = self.initialT + result.x[self.m + f :]
        p = reference.copy()
        p[h.free] += 1000 * result.x[self.m : self.m + f]
        return dict(
            flows_kg_s=dict(zip([e["id"] for e in h.edges], map(float, q))),
            temperatures_k=dict(zip(h.names, map(float, T))),
            pressures_pa=dict(zip(h.names, map(float, p))),
            maximum_scaled_residual=float(max(abs(residual))),
            branch_seed_kg_s=guess_flow,
        )
