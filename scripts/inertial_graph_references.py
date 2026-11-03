"""Manufactured inertial networks and independent bordered mixed solves.

The reference integrates full edge flows and solves for acceleration plus
junction pressure together. It does not use the production null basis,
pressure projection, residual, rates, parser or dispatcher.
"""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp


def boundary(name, pressure=200000.0, height=0.0):
    return dict(id=name, kind="pressure_boundary_v1", pressure_pa=pressure, elevation_m=height)


def junction(name, injection=0.0, height=0.0):
    return dict(id=name, kind="junction_v1", injection_kg_s=injection, elevation_m=height)


def pipe(name, a, b, inertance=10000.0, linear=1000.0, quadratic=0.0):
    return dict(
        id=name,
        kind="inertial_pipe_v1",
        **{"from": a, "to": b},
        inertance_pa_s2_kg=inertance,
        linear_pa_s_kg=linear,
        quadratic_pa_s2_kg2=quadratic,
    )


def graph(nodes, edges, initial=None, gravity=0.0):
    return dict(
        format="atomic-inertial-pipe-graph",
        schema_version=1,
        density_kg_m3=1000.0,
        gravity_m_s2=gravity,
        nodes=nodes,
        edges=edges,
        initial_flows_kg_s=initial or {e["id"]: 0.0 for e in edges},
        min_pressure_pa=1000.0,
        max_pressure_pa=2000000.0,
        max_abs_flow_kg_s=100.0,
    )


def cases():
    yield "single_pipe", graph([boundary("a", 202000), boundary("b")], [pipe("ab", "a", "b")]), []
    yield (
        "series_junction",
        graph(
            [boundary("a", 203000), junction("j"), boundary("b")],
            [pipe("aj", "a", "j"), pipe("jb", "j", "b", 20000, 2000)],
        ),
        [],
    )
    yield (
        "parallel_loop",
        graph(
            [boundary("a", 203000), junction("j"), boundary("b")],
            [
                pipe("aj1", "a", "j", 12000, 2000, 500),
                pipe("aj2", "a", "j", 8000, 1000, 200),
                pipe("jb", "j", "b", 3000, 500, 100),
            ],
        ),
        [],
    )
    yield (
        "free_cycle_decay",
        graph(
            [boundary("a"), junction("b"), junction("c")],
            [pipe("ab", "a", "b"), pipe("bc", "b", "c"), pipe("ca", "c", "a")],
            dict(ab=2.0, bc=2.0, ca=2.0),
        ),
        [],
    )
    yield (
        "injection_tree",
        graph([boundary("a"), junction("j", 1.0)], [pipe("aj", "a", "j")], dict(aj=-1.0)),
        [],
    )
    yield (
        "hydrostatic_column",
        graph(
            [boundary("a", 298100), junction("j", height=5), boundary("b", 200000, height=10)],
            [pipe("aj", "a", "j"), pipe("jb", "j", "b")],
            gravity=9.81,
        ),
        [],
    )
    yield (
        "gravity_throughflow",
        graph(
            [boundary("a", 200000, height=10), junction("j", height=5), boundary("b")],
            [pipe("aj", "a", "j", 10000, 10000, 1000), pipe("jb", "j", "b", 10000, 10000, 1000)],
            gravity=9.81,
        ),
        [],
    )
    yield (
        "boundary_reversal",
        graph(
            [boundary("a", 202000), junction("j"), boundary("b")],
            [pipe("aj", "a", "j", 4000, 1000, 200), pipe("jb", "j", "b", 6000, 1000, 200)],
        ),
        [(10.0, dict(a=198000.0, b=200000.0)), (25.0, dict(a=202000.0, b=200000.0))],
    )
    nodes = (
        [boundary("n0", 210000)] + [junction("n" + str(i)) for i in range(1, 7)] + [boundary("n7")]
    )
    edges = [
        pipe(
            "e" + str(i),
            "n" + str(i),
            "n" + str(i + 1),
            inertance=5000 + 100 * i,
            linear=1000 + 50 * i,
            quadratic=100,
        )
        for i in range(7)
    ]
    yield "eight_node_chain", graph(nodes, edges), []


class IndependentInertial:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.nodes = spec["nodes"]
        self.edges = spec["edges"]
        self.names = [n["id"] for n in self.nodes]
        self.fixed = [i for i, n in enumerate(self.nodes) if n["kind"] == "pressure_boundary_v1"]
        self.free = [i for i, n in enumerate(self.nodes) if n["kind"] == "junction_v1"]
        m = len(self.edges)
        n = len(self.nodes)
        self.incidence = np.zeros((n, m))
        for j, e in enumerate(self.edges):
            self.incidence[self.names.index(e["from"]), j] = -1
            self.incidence[self.names.index(e["to"]), j] = 1
        self.inertia = np.array([e["inertance_pa_s2_kg"] for e in self.edges])
        self.r = np.array([e["linear_pa_s_kg"] for e in self.edges])
        self.k = np.array([e["quadratic_pa_s2_kg2"] for e in self.edges])
        self.rho = spec["density_kg_m3"]
        self.g = spec["gravity_m_s2"]
        self.z = np.array([n["elevation_m"] for n in self.nodes])
        f = self.incidence[self.free]
        self.matrix = np.block(
            [[np.diag(self.inertia), f.T], [f, np.zeros((len(self.free), len(self.free)))]]
        )
        self.initial = np.r_[
            [spec["initial_flows_kg_s"][e["id"]] for e in self.edges], [0.0, 0.0, 0.0]
        ]
        self.boundaries = {
            n["id"]: n["pressure_pa"] for n in self.nodes if n["kind"] == "pressure_boundary_v1"
        }

    def evaluate(self, y, boundaries):
        m = len(self.edges)
        q = y[:m]
        loss = self.r * q + self.k * q * np.abs(q)
        pressure = np.zeros(len(self.nodes))
        pressure[self.fixed] = [boundaries[self.names[i]] for i in self.fixed]
        forcing = -self.incidence.T @ (pressure + self.rho * self.g * self.z) - loss
        solved = np.linalg.solve(self.matrix, np.r_[forcing, np.zeros(len(self.free))])
        acceleration = solved[:m]
        pressure[self.free] = solved[m:]
        external = -self.incidence @ q
        power = [pressure @ external / self.rho, self.g * self.z @ external, q @ loss / self.rho]
        return np.r_[acceleration, power], pressure

    def integrate(self, horizon, schedule, method, sample_times=()):
        state = self.initial.copy()
        current = 0.0
        boundaries = deepcopy(self.boundaries)
        samples = []
        events = {float(t): None for t in sample_times if 0 < t <= horizon}
        events[horizon] = None
        events.update(dict(schedule))
        for when, changes in sorted(events.items()):
            solution = solve_ivp(
                lambda t, y: self.evaluate(y, boundaries)[0],
                (current, when),
                state,
                method=method,
                rtol=2e-12,
                atol=1e-12,
                max_step=0.25,
            )
            if not solution.success or not np.all(np.isfinite(solution.y)):
                raise ValueError("Independent inertial solve failed")
            current = when
            state = solution.y[:, -1]
            if changes is not None:
                boundaries = deepcopy(changes)
            rate, p = self.evaluate(state, boundaries)
            samples.append(
                dict(
                    time_s=when,
                    flows_kg_s=dict(
                        zip([e["id"] for e in self.edges], map(float, state[: len(self.edges)]))
                    ),
                    pressures_pa=dict(zip(self.names, map(float, p))),
                    pressure_work_j=float(state[-3]),
                    gravity_work_j=float(state[-2]),
                    exported_heat_j=float(state[-1]),
                )
            )
        return samples
