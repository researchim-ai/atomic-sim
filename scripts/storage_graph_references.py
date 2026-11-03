"""Synthetic fixtures and independent absolute-(m,U) conservation IVPs."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp


def fluid(shift=100000.0, offset=0.0):
    return dict(
        kind="stiffened_caloric_v1",
        cv_j_kg_k=1000.0,
        gamma=1.4,
        pressure_shift_pa=shift,
        energy_offset_j_kg=offset,
        entropy_reference_temperature_k=300.0,
        entropy_reference_density_kg_m3=1.0,
    )


def node(name, p=200000.0, T=300.0, V=0.1, heat=0.0, velocity=0.0, UA=0.0, ambient=300.0):
    return dict(
        id=name,
        kind="compressible_inventory_v1",
        volume_m3=V,
        volume_rate_m3_s=velocity,
        min_volume_m3=0.5 * V,
        max_volume_m3=2 * V,
        initial_pressure_pa=p,
        initial_temperature_k=T,
        heat_power_w=heat,
        ambient_ua_w_k=UA,
        ambient_temperature_k=ambient,
    )


def link(name, a, b, r=1e6, k=0.0, heat=20.0):
    return dict(
        id=name,
        kind="resistive_thermal_link_v1",
        **{"from": a, "to": b},
        linear_pa_s_kg=r,
        quadratic_pa_s2_kg2=k,
        thermal_conductance_w_k=heat,
    )


def request(nodes, edges=(), shift=100000.0, offset=0.0):
    return dict(
        format="atomic-compressible-storage-graph",
        schema_version=1,
        fluid=fluid(shift, offset),
        domain=dict(
            min_temperature_k=100.0,
            max_temperature_k=1000.0,
            min_pressure_pa=1000.0,
            max_pressure_pa=1e9,
            min_density_kg_m3=0.001,
            max_density_kg_m3=5000.0,
        ),
        nodes=nodes,
        edges=list(edges),
    )


def cases():
    yield "sealed_heat", request([node("a", heat=100.0)]), []
    yield "adiabatic_expansion", request([node("a", velocity=0.001)]), []
    yield "adiabatic_compression", request([node("a", velocity=-0.001)]), []
    yield (
        "mass_energy_equalization",
        request([node("a", 250000, 340), node("b", 150000, 280)], [link("ab", "a", "b")]),
        [],
    )
    yield (
        "stiffened_storage",
        request(
            [node("a", 250000, 305, V=0.01, heat=200), node("b", 150000, 300, V=0.01)],
            [link("ab", "a", "b", r=1e8, heat=10)],
            shift=1e8,
        ),
        [],
    )
    yield (
        "moving_connected",
        request(
            [node("a", velocity=0.001), node("b", velocity=-0.0005)],
            [link("ab", "a", "b", r=1e7, k=1e7, heat=5)],
        ),
        [],
    )
    yield (
        "ambient_conduction",
        request(
            [node("a", T=340, UA=10, ambient=290), node("b", T=280, UA=10, ambient=310)],
            [link("ab", "a", "b", heat=40)],
        ),
        [],
    )
    yield (
        "heater_reversal",
        request([node("a"), node("b")], [link("ab", "a", "b", r=1e7, heat=10)]),
        [(5.0, dict(a=100.0, b=0.0)), (12.0, dict(a=0.0, b=100.0))],
    )
    ns = [
        node("n" + str(i), p=200000.0 + 1000 * i, T=300.0 + i, heat=20.0 if i == 0 else 0.0)
        for i in range(8)
    ]
    es = [
        link("e" + str(i), "n" + str(i), "n" + str((i + 1) % 8), r=5e6, k=1e7, heat=5.0)
        for i in range(8)
    ]
    yield "eight_volume_ring", request(ns, es), []


class IndependentStorage:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        f = spec["fluid"]
        self.cv = f["cv_j_kg_k"]
        self.gamma = f["gamma"]
        self.R = (self.gamma - 1) * self.cv
        self.shift = f["pressure_shift_pa"]
        self.offset = f["energy_offset_j_kg"]
        self.cp = self.gamma * self.cv
        self.Tref = f["entropy_reference_temperature_k"]
        self.rref = f["entropy_reference_density_kg_m3"]
        self.nodes = spec["nodes"]
        self.names = [n["id"] for n in self.nodes]
        self.n = len(self.nodes)
        self.edges = spec["edges"]
        self.V0 = np.array([n["volume_m3"] for n in self.nodes])
        self.velocity = np.array([n["volume_rate_m3_s"] for n in self.nodes])
        self.m0 = np.array(
            [
                (n["initial_pressure_pa"] + self.shift)
                * n["volume_m3"]
                / (self.R * n["initial_temperature_k"])
                for n in self.nodes
            ]
        )
        U = np.array(
            [
                self.m0[i] * (self.cv * n["initial_temperature_k"] + self.offset)
                + self.shift * self.V0[i]
                for i, n in enumerate(self.nodes)
            ]
        )
        self.initial = np.r_[self.m0, U, np.zeros(4)]
        self.heaters = {n["id"]: n["heat_power_w"] for n in self.nodes}

    def properties(self, t, y):
        m = y[: self.n]
        U = y[self.n : 2 * self.n]
        V = self.V0 + self.velocity * t
        T = (U - self.offset * m - self.shift * V) / (m * self.cv)
        rho = m / V
        p = (self.gamma - 1) * (U - self.offset * m) / V - self.gamma * self.shift
        h = (U + p * V) / m
        s = self.cv * np.log(T / self.Tref) - self.R * np.log(rho / self.rref)
        return m, U, V, T, rho, p, h, s

    def rhs(self, t, y, heaters):
        m, U, V, T, rho, p, h, s = self.properties(t, y)
        dm = np.zeros(self.n)
        dU = np.zeros(self.n)
        generation = 0.0
        for e in self.edges:
            a = self.names.index(e["from"])
            b = self.names.index(e["to"])
            drop = p[a] - p[b]
            r = e["linear_pa_s_kg"]
            k = e["quadratic_pa_s2_kg2"]
            # Solve the scalar monotone drop relation independently.
            magnitude = (
                abs(drop) / r
                if k == 0
                else (2 * abs(drop)) / (r + np.sqrt(r * r + 4 * k * abs(drop)))
            )
            upstream, downstream = (a, b) if drop >= 0 else (b, a)
            q = magnitude if drop >= 0 else -magnitude
            power = q * h[upstream] + e["thermal_conductance_w_k"] * (T[a] - T[b])
            dm[a] -= q
            dm[b] += q
            dU[a] -= power
            dU[b] += power
            generation += magnitude * (
                s[downstream] - s[upstream] + (h[upstream] - h[downstream]) / T[downstream]
            )
            generation += e["thermal_conductance_w_k"] * (T[a] - T[b]) * (1 / T[b] - 1 / T[a])
        heat = np.array([heaters[name] for name in self.names])
        ambient = np.array(
            [
                n["ambient_ua_w_k"] * (n["ambient_temperature_k"] - T[i])
                for i, n in enumerate(self.nodes)
            ]
        )
        Q = heat + ambient
        dU += Q - p * self.velocity
        entropy_out = sum(heat / T) + sum(
            ambient[i] / n["ambient_temperature_k"] for i, n in enumerate(self.nodes)
        )
        generation += sum(
            ambient[i] * (1 / T[i] - 1 / n["ambient_temperature_k"])
            for i, n in enumerate(self.nodes)
        )
        return np.r_[dm, dU, [sum(Q), p @ self.velocity, entropy_out, generation]]

    def integrate(self, horizon, schedule, method, times):
        y = self.initial.copy()
        heaters = deepcopy(self.heaters)
        time = 0.0
        rows = []
        events = {t: None for t in times}
        events[horizon] = None
        events.update(dict(schedule))
        evaluations = 0

        def fun(t, z):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 200000:
                raise ValueError("Independent storage RHS budget exhausted")
            return self.rhs(t, z, heaters)

        def jac(t, z):
            result = np.zeros((len(z), len(z)))
            for j in range(2 * self.n):
                step = 1e-6 * max(1.0, abs(z[j]))
                up = z.copy()
                down = z.copy()
                up[j] += step
                down[j] -= step
                result[:, j] = (fun(t, up) - fun(t, down)) / (2 * step)
            return result

        for end, change in sorted(events.items()):
            sol = solve_ivp(
                fun,
                (time, end),
                y,
                method=method,
                rtol=2e-12,
                atol=np.r_[
                    np.full(self.n, 1e-12), np.full(self.n, 1e-7), [1e-7, 1e-7, 1e-10, 1e-10]
                ],
                max_step=0.05,
                **({"jac": jac} if method == "Radau" else {}),
            )
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise ValueError("Independent storage IVP failed")
            y = sol.y[:, -1]
            time = end
            if change is not None:
                heaters = deepcopy(change)
            m, U, V, T, rho, p, h, s = self.properties(time, y)
            rows.append(
                dict(
                    time_s=time,
                    masses_kg=dict(zip(self.names, map(float, m))),
                    temperatures_k=dict(zip(self.names, map(float, T))),
                    pressures_pa=dict(zip(self.names, map(float, p))),
                    ledgers=y[-4:].tolist(),
                )
            )
        return rows
