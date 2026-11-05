"""Finite mixed stationary plenums connected by oriented Euler channels.

A supported finite-volume junction model, not a zero-volume tee Riemann solver.
Axial port support impulses are bookkeeping in edge coordinates; no global
vector momentum is asserted without spatial geometry.
"""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_hem import eos
from atomic_model_hem_system.model import MAX_CALLS, FiniteHEMSystem
from atomic_model_hem_system.model import signature as foundation_signature
from atomic_model_workspace.common import digest, keys
from physics.materials import iapws


def signature():
    import atomic_model_hem_closures

    return dict(
        foundation=foundation_signature(),
        network={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        closures={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(atomic_model_hem_closures.__file__).parent.glob("*.py"))
        },
    )


def identifier(value):
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.replace("_", "").isalnum()
        or len(value) > 32
    ):
        raise ValueError("Invalid graph identifier")
    return value


class GraphHEMNetwork(FiniteHEMSystem):
    def __init__(self, request):
        keys(
            request,
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
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Graph schema1 required")
        nodes, edges = request["nodes"], request["edges"]
        if (
            not isinstance(nodes, list)
            or not 2 <= len(nodes) <= 6
            or not isinstance(edges, list)
            or not 1 <= len(edges) <= 6
        ):
            raise ValueError("Graph node/edge budget")
        names = []
        points = []
        volumes = []
        powers = []
        ids = []
        for node in nodes:
            keys(node, ("id", "volume_m3", "initial", "heat_power_w"))
            name = identifier(node["id"])
            if name in names:
                raise ValueError("Duplicate node")
            volume = eos.number(node["volume_m3"], "node volume")
            if not 1e-5 <= volume <= 1:
                raise ValueError("Node volume domain")
            if node["initial"].get("velocity_m_s") != 0:
                raise ValueError("Stationary node velocity required")
            names.append(name)
            points.append(deepcopy(node["initial"]))
            volumes.append(volume)
            powers.append(node["heat_power_w"])
        self.node_count = len(nodes)
        self.edge_specs = deepcopy(edges)
        self.edge_indices = []
        self.faces = []
        self.face_areas = []
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
                )
                + (("friction_scope",) if edge["friction"] == "darcy" else ()),
            )
            name = identifier(edge["id"])
            if (
                name in ids
                or name in names
                or edge["from_node"] not in names[: len(nodes)]
                or edge["to_node"] not in names[: len(nodes)]
                or edge["from_node"] == edge["to_node"]
            ):
                raise ValueError("Invalid/duplicate graph edge")
            ids.append(name)
            count = edge["cells"]
            length = eos.number(edge["length_m"], "edge length")
            diameter = eos.number(edge["diameter_m"], "diameter")
            roughness = eos.number(edge["roughness_m"], "roughness")
            if (
                type(count) is not int
                or not 2 <= count <= 6
                or not 0.1 <= length <= 50
                or not 0.005 <= diameter <= 0.3
                or not 0 <= roughness / diameter <= 0.05
                or edge["friction"] not in ("none", "darcy")
            ):
                raise ValueError("Edge geometry/friction domain")
            if (
                not isinstance(edge["initial"], list)
                or len(edge["initial"]) != count
                or not isinstance(edge["heat_power_w"], list)
                or len(edge["heat_power_w"]) != count
            ):
                raise ValueError("Edge profile shape")
            start, end = names.index(edge["from_node"]), names.index(edge["to_node"])
            area = math.pi * diameter**2 / 4
            indices = list(range(len(names), len(names) + count))
            self.edge_indices.append(indices)
            for j, point in enumerate(edge["initial"]):
                names.append(name + "/" + str(j))
                points.append(deepcopy(point))
                volumes.append(area * length / count)
                powers.append(edge["heat_power_w"][j])
            chain = [start, *indices, end]
            self.faces.extend(zip(chain[:-1], chain[1:]))
            self.face_areas.extend([area] * (count + 1))
            incident[start] += area
            incident[end] += area
        if not 6 <= len(names) <= 14 or np.any(incident == 0):
            raise ValueError("Fluid cell budget or isolated node")
        # Require one connected closed graph; disconnected assemblies are separate requests.
        seen = {0}
        while True:
            expanded = (
                seen
                | {b for a, b in self.faces if a in seen}
                | {a for a, b in self.faces if b in seen}
            )
            if expanded == seen:
                break
            seen = expanded
        if len(seen) != len(names):
            raise ValueError("Disconnected graph")
        count = len(names)
        fake_names = ["left"] + ["c" + str(i) for i in range(count - 2)] + ["right"]
        walls = deepcopy(request["walls"])
        if not isinstance(walls, list):
            raise ValueError("Walls must be list")
        self.thermal_laws = []
        for wall in walls:
            law = wall.pop("correlation", None)
            if law is not None:
                keys(law, ("scope", "constant_wall_temperature", "thermally_developed"))
                keys(
                    law["scope"],
                    (
                        "circular_straight",
                        "fully_developed",
                        "constant_properties",
                        "forced_convection",
                        "negligible_compressibility",
                    ),
                )
                if wall.get("cell") not in names[self.node_count :]:
                    raise ValueError("Tube correlation requires a channel cell")
            self.thermal_laws.append(deepcopy(law))
            if wall.get("cell") not in names:
                raise ValueError("Unknown wall graph port")
            wall["cell"] = fake_names[names.index(wall["cell"])]
        fake = dict(
            schema_version=1,
            channel=dict(
                schema_version=1,
                geometry=dict(cells=count - 2, length_m=float(count - 2), area_m2=0.01),
                initial=points[1:-1],
                heat_power_w=powers[1:-1],
                wave_speed_bound_m_s=request["wave_speed_bound_m_s"],
                max_step_s=request["max_step_s"],
                cfl=request["cfl"],
            ),
            terminals=dict(
                left=dict(volume_m3=0.01, initial=points[0], heat_power_w=powers[0]),
                right=dict(volume_m3=0.01, initial=points[-1], heat_power_w=powers[-1]),
            ),
            walls=walls,
        )
        self._constructing = True
        super().__init__(fake)
        self._constructing = False
        velocity = self.P0 / self.m0
        self.request = deepcopy(request)
        self.identity = digest(request)
        self.names = tuple(names)
        self.volumes = np.array(volumes)
        self.m0 = np.array([s.density_kg_m3 * vol for s, vol in zip(self.states0, self.volumes)])
        self.P0 = self.m0 * velocity
        self.C0 = (
            np.array(
                [
                    mass * (s.internal_energy_j_kg - self.eref)
                    for mass, s in zip(self.m0, self.states0)
                ]
            )
            + 0.5 * self.m0 * velocity**2
        )
        self.face_count = len(self.faces)
        self.face_areas = np.array(self.face_areas)
        lengths = [volumes[i] / incident[i] for i in range(self.node_count)]
        lengths += [e["length_m"] / e["cells"] for e in edges]
        self.minimum_length = float(min(lengths))
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
        self.source = signature()
        self.evaluate(self._y, self._controls)

    def _sources(self):
        if signature() != self.source:
            raise ValueError("Graph source changed")

    def contract(self):
        value = super().contract()
        value.update(
            model="finite-plenum-if97-graph-v1",
            outer_boundary="Closed finite graph; stationary mixed nodes, no external mass/energy",
            faces=[
                dict(left=self.names[a], right=self.names[b], area_m2=float(area))
                for (a, b), area in zip(self.faces, self.face_areas)
            ],
            momentum_scope="Per-edge axial balance with signed support bookkeeping; no global vector momentum claim",
        )
        value["wall_laws"] = [
            dict(
                id=name,
                law="linear" if law is None else "single_phase_laminar_or_gnielinski",
                conductance_upper_bound_w_k=float(self.conductance[j]),
            )
            for j, (name, law) in enumerate(zip(self.wall_ids, self.thermal_laws))
        ]
        value["friction_scope"] = (
            "Quasi-steady fully-developed straight circular tube law only with caller attestations; transitions and two-phase drag refused"
        )
        value["ledger_units"].update(
            drag_impulse="kg m/s", node_support_impulse="kg m/s", drag_entropy_production="J/K"
        )
        return value

    def _thermal_conductance(self, point):
        from atomic_model_hem_closures import TubeScope, heat_transfer

        _, _, _, velocity, states, tw, _ = point
        conductance = self.conductance.copy()
        for j, law in enumerate(self.thermal_laws):
            if law is None:
                continue
            i = self.ports[j]
            edge = next(e for e, indices in zip(self.edge_specs, self.edge_indices) if i in indices)
            state = states[i]
            if state.region not in (1, 2):
                raise ValueError("Two-phase wall heat-transfer correlation unsupported")
            result = heat_transfer(
                iapws.properties_pt(state.pressure_pa, state.temperature_k),
                state.density_kg_m3 * float(velocity[i]),
                edge["diameter_m"],
                float(tw[j]),
                scope=TubeScope(**law["scope"]),
                constant_wall_temperature=law["constant_wall_temperature"],
                thermally_developed=law["thermally_developed"],
                roughness_m=edge["roughness_m"],
                length_m=edge["length_m"] / edge["cells"],
            )
            actual = (
                result.coefficient_w_m2_k
                * math.pi
                * edge["diameter_m"]
                * edge["length_m"]
                / edge["cells"]
            )
            if actual > self.conductance[j]:
                raise ValueError("Correlation conductance exceeds declared thermal step bound")
            conductance[j] = actual
        return conductance

    def _evaluate_point(self, y, controls, point):
        if self._constructing:
            # The inherited constructor parses bounded states and wall fields.
            # Its temporary chain's reflecting end faces are not graph ports.
            # Actual graph rates/admission run after topology and stocks exist.
            return np.zeros_like(y), {}
        m, P, C, v, states, tw, speeds = point
        if np.any(P[: self.node_count] != 0):
            raise ValueError("Stationary node acquired momentum")
        W = np.array([m, P, C]) / self.volumes
        pressure = np.array([s.pressure_pa for s in states])
        temperature = np.array([s.temperature_k for s in states])
        F = np.array(
            [P / self.volumes, P * v / self.volumes + pressure, v * (C / self.volumes + pressure)]
        )
        flux = (
            np.array(
                [
                    0.5 * (F[:, a] + F[:, b]) - 0.5 * self.wave * (W[:, b] - W[:, a])
                    for a, b in self.faces
                ]
            ).T
            * self.face_areas
        )
        rhs = np.zeros_like(y)
        net = np.zeros((3, self.n))
        for j, (a, b) in enumerate(self.faces):
            net[:, a] -= flux[:, j]
            net[:, b] += flux[:, j]
        support = np.zeros(self.n)
        support[: self.node_count] = -net[1, : self.node_count]
        drag = np.zeros(self.n)
        from atomic_model_hem_closures import TubeScope, friction_factor

        for edge, indices in zip(self.edge_specs, self.edge_indices):
            if edge["friction"] == "none":
                continue
            for i in indices:
                if states[i].region not in (1, 2):
                    raise ValueError("Two-phase Darcy closure unsupported")
                scope_data = edge["friction_scope"]
                keys(
                    scope_data,
                    (
                        "circular_straight",
                        "fully_developed",
                        "constant_properties",
                        "forced_convection",
                        "negligible_compressibility",
                    ),
                )
                state = iapws.properties_pt(states[i].pressure_pa, states[i].temperature_k)
                # Validate transport/scope even at rest. The zero-force limit is exact;
                # the tiny positive validation flux is never used as a physical rate.
                G = states[i].density_kg_m3 * abs(float(v[i]))
                result = friction_factor(
                    state,
                    max(G, 1e-8),
                    edge["diameter_m"],
                    scope=TubeScope(**scope_data),
                    roughness_m=edge["roughness_m"],
                )
                if G == 0:
                    continue
                factor = result.darcy_factor
                drag[i] = (
                    -factor
                    * self.volumes[i]
                    * states[i].density_kg_m3
                    * v[i]
                    * abs(v[i])
                    / (2 * edge["diameter_m"])
                )
        admission_dt = min(self.step_cap, self.cfl * self.minimum_length / self.wave)
        moving = P != 0
        if np.any(admission_dt * abs(drag[moving] / P[moving]) > 0.1):
            raise ValueError("Explicit Darcy relaxation bound exceeded; reduce max_step_s")
        net[1] += support + drag
        conductance = self._thermal_conductance(point)
        exchange = conductance * (tw - temperature[self.ports])
        external = controls["walls"] + self.ua * (self.ambient - tw)
        net[2] += controls["fluid"]
        np.add.at(net[2], self.ports, exchange)
        rhs[: 3 * self.n] = net.reshape(-1)
        rhs[self.wall_slice] = (external - exchange) / self.capacity
        physical = flux.copy()
        physical[2] += self.eref * physical[0]
        rhs[self.face_slice] = physical.reshape(-1)
        rhs[self.cell_heat_slice] = controls["fluid"]
        rhs[self.wall_exchange_slice] = exchange
        rhs[self.wall_external_slice] = external
        rhs[self.drag_slice] = drag
        rhs[self.support_slice] = support
        rhs[self.entropy_index] = sum(controls["fluid"] / temperature) + sum(
            controls["walls"] / tw + self.ua * (self.ambient - tw) / self.ambient
        )
        rhs[self.production_index] = sum(
            conductance * (tw - temperature[self.ports]) ** 2 / (tw * temperature[self.ports])
            + self.ua * (self.ambient - tw) ** 2 / (self.ambient * tw)
        )
        rhs[self.drag_entropy_index] = sum(-drag * v / temperature)
        if not np.isfinite(rhs).all() or rhs[self.drag_entropy_index] < 0:
            raise ValueError("Invalid graph rates")
        return rhs, dict(
            shared_flux=flux,
            wall_to_fluid_w=exchange,
            drag_force_n=drag,
            node_support_force_n=support,
        )

    def _observe(self, y, time, controls, controller, revision, _point=None):
        if self._constructing:
            return super()._observe(y, time, controls, controller, revision, _point)
        m, P, C, v, states, tw, speeds = self.primitive(y) if _point is None else _point
        if np.any(P[: self.node_count] != 0):
            raise ValueError("Node momentum ledger invalid")
        face = y[self.face_slice].reshape(3, self.face_count)
        net = np.zeros((3, self.n))
        for j, (a, b) in enumerate(self.faces):
            net[:, a] -= face[:, j]
            net[:, b] += face[:, j]
        dm = y[: self.n]
        dP = y[self.n : 2 * self.n]
        dE = y[2 * self.n : 3 * self.n] + self.eref * dm
        exchange = np.zeros(self.n)
        np.add.at(exchange, self.ports, y[self.wall_exchange_slice])
        mass_error = dm - net[0]
        momentum_error = dP - net[1] - y[self.drag_slice] - y[self.support_slice]
        energy_error = dE - net[2] - y[self.cell_heat_slice] - exchange
        wall_energy = self.capacity * y[self.wall_slice]
        wall_error = wall_energy - y[self.wall_external_slice] + y[self.wall_exchange_slice]
        energy = float(
            sum(dE)
            + sum(wall_energy)
            - sum(y[self.cell_heat_slice])
            - sum(y[self.wall_external_slice])
        )
        scale = max(
            1.0,
            float(max(abs(face[2]))),
            float(sum(abs(y[self.cell_heat_slice])) + sum(abs(y[self.wall_external_slice]))),
        )
        if (
            max(abs(mass_error)) > 1e-9
            or max(abs(momentum_error)) > 1e-8
            or max(abs(energy_error)) > 1e-5 + 1e-8 * scale
            or max(abs(wall_error)) > 1e-5 + 1e-8 * scale
            or abs(energy) > 1e-5 + 1e-8 * scale
        ):
            raise ValueError("Graph local/global conservation ledger exceeded")
        ds = float(
            sum(
                (m - self.m0) * self.s0
                + m * (np.array([s.entropy_j_kg_k for s in states]) - self.s0)
            )
            + sum(self.capacity * np.log1p(y[self.wall_slice] / self.tw0))
        )
        return dict(
            time_s=float(time),
            revision=revision,
            controller_state=deepcopy(controller),
            controls=dict(
                fluid=dict(zip(self.names, map(float, controls["fluid"]))),
                walls=dict(zip(self.wall_ids, map(float, controls["walls"]))),
            ),
            cells=[
                dict(
                    id=self.names[i],
                    volume_m3=float(self.volumes[i]),
                    mass_kg=float(m[i]),
                    momentum_kg_m_s=float(P[i]),
                    total_energy_j=float(C[i] + self.eref * m[i]),
                    velocity_m_s=float(v[i]),
                    **states[i].as_dict(),
                )
                for i in range(self.n)
            ],
            walls=[
                dict(
                    id=name,
                    cell=self.names[self.ports[i]],
                    temperature_k=float(tw[i]),
                    energy_change_j=float(wall_energy[i]),
                    heat_to_fluid_j=float(y[self.wall_exchange_slice][i]),
                    external_heat_j=float(y[self.wall_external_slice][i]),
                    energy_residual_j=float(wall_error[i]),
                )
                for i, name in enumerate(self.wall_ids)
            ],
            local_ledgers=dict(
                mass_residual_kg=list(map(float, mass_error)),
                momentum_residual_kg_m_s=list(map(float, momentum_error)),
                energy_residual_j=list(map(float, energy_error)),
                integrated_face_mass_kg=list(map(float, face[0])),
                integrated_face_momentum_kg_m_s=list(map(float, face[1])),
                integrated_face_energy_j=list(map(float, face[2])),
                drag_impulse_kg_m_s=list(map(float, y[self.drag_slice])),
                node_support_impulse_kg_m_s=list(map(float, y[self.support_slice])),
            ),
            global_ledger=dict(
                mass_residual_kg=float(sum(dm)),
                energy_residual_j=energy,
                external_fluid_heat_j=float(sum(y[self.cell_heat_slice])),
                external_wall_heat_j=float(sum(y[self.wall_external_slice])),
                total_entropy_change_j_k=ds,
                external_entropy_j_k=float(y[self.entropy_index]),
                thermal_entropy_production_j_k=float(y[self.production_index]),
                drag_entropy_production_j_k=float(y[self.drag_entropy_index]),
                numerical_entropy_residual_j_k=ds
                - float(
                    y[self.entropy_index] + y[self.production_index] + y[self.drag_entropy_index]
                ),
            ),
            maximum_characteristic_speed_m_s=float(max(abs(v) + speeds)),
            entropy_acceptance="Unforced finite graph entropy nondecrease; forced and junction entropy stability remain diagnostic",
            validated=False,
            plant_referenced=False,
        )

    def _checkpoint(self, *args):
        cp = super()._checkpoint(*args)
        cp["payload"]["model"] = "finite-plenum-if97-graph-v1"
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
                raise ValueError("Graph checkpoint identity/size")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("Graph replay budget")
            other = GraphHEMNetwork(self.request)
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
                raise ValueError("Graph full checkpoint replay differs")
            self._sources()
            self._y, self._time, self._controls, self._controller, self._calls = (
                other._y,
                other._time,
                other._controls,
                other._controller,
                other._calls,
            )
            return self.snapshot()
