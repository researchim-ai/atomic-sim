"""Independent full-flow bordered IVPs with conservative thermal transport."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.inertial_graph_references import IndependentInertial
from scripts.inertial_graph_references import cases as hydraulic_cases


def request(
    hydraulic, *, beta=0.01, recovery=0.6, heat=2000.0, volume=0.02, initial=300.0, supply=330.0
):
    names = [n["id"] for n in hydraulic["nodes"]]
    return dict(
        format="atomic-thermal-inertial-graph",
        schema_version=1,
        hydraulic=deepcopy(hydraulic),
        thermal=dict(
            heat_capacity_j_kg_k=1000.0,
            reference_temperature_k=300.0,
            min_temperature_k=250.0,
            max_temperature_k=500.0,
            nodes={
                name: dict(
                    volume_m3=volume,
                    initial_temperature_k=initial,
                    heat_power_w=heat if i == 0 else 0.0,
                    ambient_ua_w_k=10.0,
                    ambient_temperature_k=295.0,
                    supply_temperature_k=supply if i == 0 else 290.0,
                )
                for i, name in enumerate(names)
            },
        ),
        edge_laws={
            e["id"]: dict(
                reference_temperature_k=300.0,
                temperature_coefficient_per_k=beta,
                minimum_multiplier=0.1,
                maximum_multiplier=10.0,
                fluid_heat_fraction=recovery,
            )
            for e in hydraulic["edges"]
        },
    )


def cases():
    for name, spec, schedule in hydraulic_cases():
        r = request(spec)
        if name == "single_pipe":
            r = request(spec, beta=0, recovery=0, heat=0, initial=300, supply=300)
            for n in r["thermal"]["nodes"].values():
                n["ambient_ua_w_k"] = 0
                n["supply_temperature_k"] = 300
        yield name, r, schedule


class IndependentThermalInertial:
    def __init__(self, spec):
        self.h = IndependentInertial(spec["hydraulic"])
        self.spec = deepcopy(spec)
        self.n = len(self.h.names)
        self.m = len(self.h.edges)
        t = spec["thermal"]
        self.cp = t["heat_capacity_j_kg_k"]
        self.ref = t["reference_temperature_k"]
        self.initialT = np.array(
            [t["nodes"][name]["initial_temperature_k"] for name in self.h.names]
        )
        self.capacity = np.array(
            [t["nodes"][name]["volume_m3"] * self.h.rho * self.cp for name in self.h.names]
        )
        self.initial = np.r_[self.h.initial[: self.m], np.zeros(self.n + 6)]

    def rhs(self, time, y, boundaries):
        h = self.h
        q = y[: self.m]
        T = self.initialT + y[self.m : self.m + self.n]
        p = np.zeros(self.n)
        p[h.fixed] = [boundaries[h.names[i]] for i in h.fixed]
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
            heat[a] += recovered / 2
            heat[b] += recovered / 2
            fluid += recovered
            bath += dissipated - recovered
        forcing = -h.incidence.T @ (p + h.rho * h.g * h.z) - loss
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
        return np.r_[solution[: self.m], heat / self.capacity, [P, G, Q, H, fluid, bath]], p

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
                atol=np.r_[np.full(self.m + self.n, 1e-12), np.full(6, 1e-8)],
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
                    ledgers=state[-6:].tolist(),
                )
            )
        return rows
