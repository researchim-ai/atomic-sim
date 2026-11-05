"""Finite-plenum graph with active-chart EOS, conservative drag and pre-CHF walls."""

import hashlib
import json
from copy import deepcopy
from math import pi
from pathlib import Path

import numpy as np

from atomic_model_hem_mixture.model import signature as mixture_signature
from atomic_model_hem_network.cases import baseline
from atomic_model_hem_network.model import GraphHEMNetwork, identifier
from atomic_model_hem_network.model import signature as graph_signature
from atomic_model_hem_qualified import closures, eos
from atomic_model_hem_system.model import MAX_CALLS, MAX_STEPS, FiniteHEMSystem
from atomic_model_workspace.common import digest, keys


def signature():
    return dict(
        graph=graph_signature(),
        mixture=mixture_signature(),
        qualified={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def initial(point):
    kind = point.get("kind")
    if kind == "pt":
        keys(point, ("kind", "pressure_pa", "temperature_k", "velocity_m_s"))
        state = eos.pt(point["pressure_pa"], point["temperature_k"])
    elif kind == "px":
        keys(point, ("kind", "pressure_pa", "quality", "velocity_m_s"))
        state = eos.px(point["pressure_pa"], point["quality"])
    elif kind == "rho_u":
        keys(
            point,
            (
                "kind",
                "density_kg_m3",
                "internal_energy_j_kg",
                "pressure_seed_pa",
                "temperature_seed_k",
                "velocity_m_s",
            ),
        )
        state = eos.rho_u(
            point["density_kg_m3"],
            point["internal_energy_j_kg"],
            pressure_seed_pa=point["pressure_seed_pa"],
            temperature_seed_k=point["temperature_seed_k"],
        )
    else:
        raise ValueError("Initial state kind")
    return state, eos.number(point["velocity_m_s"], "velocity")


class QualifiedHEMNetwork(GraphHEMNetwork):
    def __init__(self, request):
        keys(request, ("schema_version", "model", "friction"))
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Qualified schema1")
        spec = request["model"]
        keys(
            spec,
            (
                "schema_version",
                "nodes",
                "edges",
                "walls",
                "wave_speed_bound_m_s",
                "max_step_s",
                "cfl",
            ),
        )
        if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
            raise ValueError("Graph schema1")
        nodes, edges = spec["nodes"], spec["edges"]
        if (
            not isinstance(nodes, list)
            or not 2 <= len(nodes) <= 32
            or not isinstance(edges, list)
            or not 1 <= len(edges) <= 32
        ):
            raise ValueError("Node/edge budget")
        names = []
        states = []
        velocities = []
        volumes = []
        powers = []
        faces = []
        areas = []
        indices = []
        edgeids = []
        for node in nodes:
            keys(node, ("id", "volume_m3", "initial", "heat_power_w"))
            name = identifier(node["id"])
            if name in names:
                raise ValueError("Duplicate node")
            state, v = initial(node["initial"])
            volume = eos.number(node["volume_m3"], "node volume")
            if v != 0 or not 1e-5 <= volume <= 1:
                raise ValueError("Stationary finite node domain")
            names.append(name)
            states.append(state)
            velocities.append(v)
            volumes.append(volume)
            powers.append(node["heat_power_w"])
        incident = np.zeros(len(nodes))
        for edge in edges:
            keys(
                edge,
                (
                    "id",
                    "from_node",
                    "to_node",
                    "cells",
                    "length_m",
                    "diameter_m",
                    "roughness_m",
                    "friction",
                    "initial",
                    "heat_power_w",
                ),
            )
            name = identifier(edge["id"])
            count = edge["cells"]
            length = eos.number(edge["length_m"], "length")
            diam = eos.number(edge["diameter_m"], "diameter")
            rough = eos.number(edge["roughness_m"], "roughness")
            if (
                name in edgeids
                or name in names
                or edge["from_node"] not in names[: len(nodes)]
                or edge["to_node"] not in names[: len(nodes)]
                or edge["from_node"] == edge["to_node"]
            ):
                raise ValueError("Graph edge ids/ports")
            if (
                type(count) is not int
                or not 2 <= count <= 64
                or not 0.1 <= length <= 50
                or not 0.005 <= diam <= 0.3
                or not 0 <= rough / diam <= 0.05
                or edge["friction"] != "none"
            ):
                raise ValueError("Edge domain; friction must be selected separately")
            if (
                not isinstance(edge["initial"], list)
                or len(edge["initial"]) != count
                or not isinstance(edge["heat_power_w"], list)
                or len(edge["heat_power_w"]) != count
            ):
                raise ValueError("Edge profiles")
            a, b = names.index(edge["from_node"]), names.index(edge["to_node"])
            area = pi * diam**2 / 4
            ix = []
            for j, point in enumerate(edge["initial"]):
                state, v = initial(point)
                ix.append(len(names))
                names.append(name + "/" + str(j))
                states.append(state)
                velocities.append(v)
                volumes.append(area * length / count)
                powers.append(edge["heat_power_w"][j])
            chain = [a, *ix, b]
            faces.extend(zip(chain[:-1], chain[1:]))
            areas.extend([area] * (count + 1))
            indices.append(ix)
            edgeids.append(name)
            incident[a] += area
            incident[b] += area
        if not 4 <= len(names) <= 256 or np.any(incident == 0):
            raise ValueError("Fluid budget/isolated node")
        seen = {0}
        while True:
            enlarged = (
                seen | {b for a, b in faces if a in seen} | {a for a, b in faces if b in seen}
            )
            if enlarged == seen:
                break
            seen = enlarged
        if len(seen) != len(names):
            raise ValueError("Disconnected assembly")
        if not isinstance(request["friction"], dict) or not set(request["friction"]) <= set(
            edgeids
        ):
            raise ValueError("Friction edge selection")
        for law in request["friction"].values():
            keys(law, ("viscosity_model", "scope", "darcy_law"))
            keys(law["scope"], closures.FLAGS)
        walls = deepcopy(spec["walls"])
        boiling_laws = []
        if not isinstance(walls, list) or not 1 <= len(walls) <= 16:
            raise ValueError("Wall budget")
        for wall in walls:
            if wall.get("cell") not in names:
                raise ValueError("Wall port")
            law = wall.pop("boiling", None)
            boiling_laws.append(law)
            if law is not None:
                keys(law, ("scope", "chf_model", "maximum_chf_fraction"))
                keys(law["scope"], closures.HEAT_FLAGS)
                if wall["cell"] in names[: len(nodes)]:
                    raise ValueError("Chen requires channel cell")
            wall["cell"] = "e0/0"
        # Reuse only validated scalar wall/control/parser limits from the old family.
        # Dummy physics is never evaluated as actual ports or accepted stocks.
        dummy = baseline("parallel")
        dummy["walls"] = walls
        for field in ["wave_speed_bound_m_s", "max_step_s", "cfl"]:
            dummy[field] = spec[field]
        self._initializing = True
        super().__init__(dummy)
        self._initializing = False
        self.request = deepcopy(request)
        self.identity = digest(request)
        self.n = len(names)
        self.node_count = len(nodes)
        self.names = tuple(names)
        self.states0 = states
        self.s0 = np.array([s.entropy_j_kg_k for s in states])
        self.volumes = np.array(volumes)
        self.eref = states[0].internal_energy_j_kg
        v = np.array(velocities)
        self.m0 = np.array([s.density_kg_m3 * vol for s, vol in zip(states, volumes)])
        self.P0 = self.m0 * v
        self.C0 = (
            np.array([m * (s.internal_energy_j_kg - self.eref) for m, s in zip(self.m0, states)])
            + 0.5 * self.m0 * v * v
        )
        self.faces = faces
        self.face_areas = np.array(areas)
        self.face_count = len(faces)
        self.edge_specs = deepcopy(edges)
        self.edge_indices = indices
        self.friction_laws = deepcopy(request["friction"])
        self.boiling_laws = boiling_laws
        self.ports = np.array([names.index(w["cell"]) for w in spec["walls"]])
        self.minimum_length = min(
            [volumes[i] / incident[i] for i in range(len(nodes))]
            + [e["length_m"] / e["cells"] for e in edges]
        )
        self.wall_slice = slice(3 * self.n, 3 * self.n + self.w)
        cursor = self.wall_slice.stop
        self.face_slice = slice(cursor, cursor + 3 * self.face_count)
        cursor = self.face_slice.stop
        self.cell_heat_slice = slice(cursor, cursor + self.n)
        cursor = self.cell_heat_slice.stop
        self.wall_exchange_slice = slice(cursor, cursor + self.w)
        cursor = self.wall_exchange_slice.stop
        self.wall_external_slice = slice(cursor, cursor + self.w)
        cursor = self.wall_external_slice.stop
        self.entropy_index, self.production_index = cursor, cursor + 1
        cursor += 2
        self.drag_slice = slice(cursor, cursor + self.n)
        cursor = self.drag_slice.stop
        self.support_slice = slice(cursor, cursor + self.n)
        cursor = self.support_slice.stop
        self.drag_entropy_index = cursor
        self.size = cursor + 1
        self._y = np.zeros(self.size)
        self._controls = self._mask(
            dict(
                fluid=dict(zip(names, powers)),
                walls={w["id"]: w["heat_power_w"] for w in spec["walls"]},
            )
        )
        self.source = signature()
        self.evaluate(self._y, self._controls)

    def primitive(self, y):
        if self._initializing:
            return FiniteHEMSystem.primitive(self, y)
        if not isinstance(y, np.ndarray) or y.shape != (self.size,) or not np.isfinite(y).all():
            raise ValueError("Qualified coordinates")
        m = self.m0 + y[: self.n]
        P = self.P0 + y[self.n : 2 * self.n]
        C = self.C0 + y[2 * self.n : 3 * self.n]
        if np.any(m <= 0):
            raise ValueError("Nonpositive mass")
        v = P / m
        u = self.eref + (C - 0.5 * m * v * v) / m
        states = []
        speeds = []
        cache = {}
        for i, seed in enumerate(self.states0):
            key = (float(m[i] / self.volumes[i]), float(u[i]), seed.pressure_pa, seed.temperature_k)
            if key not in cache:
                state = eos.rho_u(*key)
                cache[key] = (state, *eos.characteristic_speeds(state))
            state, lower, upper = cache[key]
            if abs(v[i]) > 0.2 * lower or abs(v[i]) + upper > self.wave:
                raise ValueError("Mach/wave envelope")
            states.append(state)
            speeds.append(upper)
        tw = self.tw0 + y[self.wall_slice]
        if np.any(tw < 300) or np.any(tw > 900):
            raise ValueError("Wall temperature domain")
        return m, P, C, v, states, tw, np.array(speeds)

    def _sources(self):
        if signature() != self.source:
            raise ValueError("Qualified source changed")

    def _thermal_conductance(self, point):
        if self._initializing:
            return self.conductance.copy()
        _, _, _, v, states, tw, _ = point
        K = self.conductance.copy()
        for j, law in enumerate(self.boiling_laws):
            if law is None:
                continue
            i = self.ports[j]
            edge = next(e for e, ix in zip(self.edge_specs, self.edge_indices) if i in ix)
            result = closures.boiling(
                states[i],
                states[i].density_kg_m3 * float(v[i]),
                edge["diameter_m"],
                float(tw[j]),
                edge["length_m"],
                **law,
            )
            area = pi * edge["diameter_m"] * edge["length_m"] / edge["cells"]
            if (
                max(result.coefficient_w_m2_k, result.differential_conductance_w_m2_k) * area
                > self.conductance[j]
            ):
                raise ValueError("Boiling differential conductance exceeds declared step bound")
            K[j] = result.coefficient_w_m2_k * area
        return K

    def _evaluate_point(self, y, controls, point):
        if self._initializing:
            return np.zeros_like(y), {}
        rhs, aux = super()._evaluate_point(y, controls, point)
        m, P, C, v, states, tw, speeds = point
        drag = np.zeros(self.n)
        sigma = 0.0
        for e, ix in zip(self.edge_specs, self.edge_indices):
            law = self.friction_laws.get(e["id"])
            if law is None:
                continue
            for i in ix:
                result = closures.friction(
                    states[i], float(v[i]), e["diameter_m"], e["roughness_m"], **law
                )
                drag[i] = result["force_density_n_m3"] * self.volumes[i]
                sigma += result["entropy_production_w_m3_k"] * self.volumes[i]
        dt = min(self.step_cap, self.cfl * self.minimum_length / self.wave)
        moving = P != 0
        if np.any(dt * abs(drag[moving] / P[moving]) > 0.1):
            raise ValueError("Friction relaxation bound")
        rhs[self.n : 2 * self.n] += drag
        rhs[self.drag_slice] += drag
        rhs[self.drag_entropy_index] += sigma
        aux["drag_force_n"] += drag
        return rhs, aux

    def contract(self):
        v = super().contract()
        v.update(
            model="active-chart-pre-chf-graph-v1",
            eos_domain=dict(
                pressure_pa=[eos.P_MIN, eos.P_MAX], temperature_k=[eos.T_MIN, eos.T_MAX]
            ),
            friction=deepcopy(self.friction_laws),
            boiling=deepcopy(self.boiling_laws),
            limits=dict(nodes=32, edges=32, total_fluid_cells=256, cells_per_edge=64),
            closure_version=closures.VERSION,
            closure_sources=dict(
                boiling=closures.CHEN_SOURCE,
                chf=closures.CHF_SOURCE,
                surface_tension=closures.SURFACE_SOURCE,
            ),
            closure_uncertainty="Not calibrated; no confidence interval assigned; model-choice spread only",
            qualification="Numerical verification only; no experimental or plant promotion",
        )
        v["wall_laws"] = [
            dict(
                id=name,
                law="linear" if law is None else "chen-inl-documented-hfg025",
                conductance_upper_bound_w_k=float(self.conductance[j]),
            )
            for j, (name, law) in enumerate(zip(self.wall_ids, self.boiling_laws))
        ]
        v["friction_scope"] = (
            "Explicit homogeneous no-slip viscosity closure and selected Darcy law; no slip or experimental validation"
        )
        return v

    def _checkpoint(self, *args):
        cp = super()._checkpoint(*args)
        cp["payload"]["model"] = "active-chart-pre-chf-graph-v1"
        cp["sha256"] = digest(cp["payload"])
        return cp

    def restore(self, checkpoint):
        with self._lock:
            self._sources()
            keys(checkpoint, ("payload", "sha256"))
            p = checkpoint["payload"]
            if (
                len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2
                or digest(p) != checkpoint["sha256"]
                or p.get("source") != self.source
                or p.get("input_sha256") != self.identity
            ):
                raise ValueError("Qualified checkpoint identity/size")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("Replay budget")
            other = QualifiedHEMNetwork(self.request)
            for call in calls:
                keys(call, ("duration_s", "command"))
                c = call["command"]
                if c is None:
                    other.advance(call["duration_s"])
                else:
                    keys(c, ("controls", "expected_revision", "controller_state"))
                    other.step(
                        c["controls"],
                        call["duration_s"],
                        expected_revision=c["expected_revision"],
                        controller_state=c["controller_state"],
                    )
            if other.checkpoint() != checkpoint:
                raise ValueError("Full qualified replay differs")
            self._sources()
            self._y, self._time, self._controls, self._controller, self._calls = (
                other._y,
                other._time,
                other._controls,
                other._controller,
                other._calls,
            )
            return self.snapshot()

    def _execute(self, duration, controls, controller, command):
        self._sources()
        duration = eos.number(duration, "duration")
        target = self._time + duration
        if not 0 < duration <= 0.1 or target <= self._time or len(self._calls) >= MAX_CALLS:
            raise ValueError("System duration/call budget")
        y = self._y.copy()
        time = self._time
        steps = 0
        maximum = 0.0
        minimum = float("inf")
        entropy_start = self._observe(y, time, controls, controller, len(self._calls))[
            "global_ledger"
        ]["total_entropy_change_j_k"]
        previous_entropy = entropy_start
        unforced = bool(
            np.all(controls["fluid"] == 0)
            and np.all(controls["walls"] == 0)
            and np.all(self.ua == 0)
        )
        point = self.primitive(y)
        while time < target:
            k1 = self._evaluate_point(y, controls, point)[0]
            m, P, C, v, states, tw, speeds = point
            internal = np.array([s.internal_energy_j_kg - self.eref for s in states])
            du = (
                k1[2 * self.n : 3 * self.n]
                - v * k1[self.n : 2 * self.n]
                + (0.5 * v * v - internal) * k1[: self.n]
            ) / m
            wall_rate = max(abs(k1[self.wall_slice]))
            conductance_by_cell = np.zeros(self.n)
            np.add.at(conductance_by_cell, self.ports, self.conductance)
            fluid_thermal_rate = max(
                (
                    k / (mass * eos.capacity(state))
                    for k, mass, state in zip(conductance_by_cell, m, states)
                    if k > 0
                ),
                default=0.0,
            )
            wall_thermal_rate = float(max((self.conductance + self.ua) / self.capacity))
            thermal_relax = 0.1 / max(fluid_thermal_rate, wall_thermal_rate, 1e-30)
            dt = min(
                self.step_cap,
                self.cfl * self.minimum_length / self.wave,
                target - time,
                thermal_relax,
                500 / max(float(max(abs(du))), 1e-30),
                0.25 / max(float(wall_rate), 1e-30),
            )
            if time + dt <= time or steps >= MAX_STEPS:
                raise ValueError("System step/clock budget")
            a = y + dt * k1
            k2 = self.evaluate(a, controls)[0]
            b = 0.75 * y + 0.25 * (a + dt * k2)
            k3 = self.evaluate(b, controls)[0]
            candidate = y / 3 + 2 * (b + dt * k3) / 3
            next_time = min(target, time + dt)
            point = self.primitive(candidate)
            observed = self._observe(
                candidate, next_time, controls, controller, len(self._calls) + 1, _point=point
            )
            if observed["global_ledger"]["numerical_entropy_residual_j_k"] < -(
                1e-5 + 2e-7 * float(sum(self.m0))
            ):
                raise ValueError("Qualified forced/unforced entropy residual bound exceeded")
            entropy = observed["global_ledger"]["total_entropy_change_j_k"]
            if unforced and entropy < max(entropy_start, previous_entropy) - (
                1e-5 + 2e-7 * float(sum(self.m0))
            ):
                raise ValueError("System unforced entropy decreased")
            y, time = candidate, next_time
            previous_entropy = entropy
            steps += 1
            maximum = max(maximum, dt)
            minimum = min(minimum, dt)
        calls = self._calls + [dict(duration_s=duration, command=deepcopy(command))]
        cp = self._checkpoint(y, time, controls, controller, calls)
        self._sources()
        self._y, self._time, self._controls, self._controller, self._calls = (
            y,
            time,
            deepcopy(controls),
            deepcopy(controller),
            calls,
        )
        return dict(
            **cp["payload"]["observation"],
            steps=steps,
            maximum_step_s=maximum,
            minimum_step_s=minimum,
        )
