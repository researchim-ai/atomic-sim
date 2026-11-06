"""Independent absolute-stock graph IVP, sharing only the strict HEM EOS.

Nodes have mass/internal energy and no momentum. Support impulse is signed
axial bookkeeping summed over oriented edges, not a physical global vector.
Darcy friction uses independently solved laminar/Colebrook equations. Agreement verifies
the same semidiscretization, not junction physics or an independent EOS.
"""

from copy import deepcopy
from math import log10, pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from atomic_model_hem import eos
from physics.materials import iapws


class Assembly:
    """Standalone loop assembly; no production parser, flux or state layout."""

    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.nodes = len(spec["nodes"])
        self.names = [node["id"] for node in spec["nodes"]]
        volumes = [node["volume_m3"] for node in spec["nodes"]]
        points = [node["initial"] for node in spec["nodes"]]
        powers = [node["heat_power_w"] for node in spec["nodes"]]
        self.faces = []
        self.drag_cells = []
        for edge in spec["edges"]:
            if edge["friction"] not in ("none", "darcy"):
                raise ValueError("Unknown reference friction")
            area = pi * edge["diameter_m"] ** 2 / 4
            chain = [self.names.index(edge["from_node"])]
            for j in range(edge["cells"]):
                if edge["friction"] == "darcy":
                    self.drag_cells.append((len(self.names), deepcopy(edge)))
                chain.append(len(self.names))
                self.names.append(edge["id"] + "/" + str(j))
                volumes.append(area * edge["length_m"] / edge["cells"])
                points.append(edge["initial"][j])
                powers.append(edge["heat_power_w"][j])
            chain.append(self.names.index(edge["to_node"]))
            self.faces.extend((a, b, area) for a, b in zip(chain, chain[1:]))
        self.volumes = np.asarray(volumes)
        self.n = len(points)
        self.walls = deepcopy(spec["walls"])
        self.ports = [self.names.index(wall["cell"]) for wall in self.walls]
        self.wave = spec["wave_speed_bound_m_s"]
        self.controls = dict(
            fluid=dict(zip(self.names, powers)),
            walls={w["id"]: w["heat_power_w"] for w in self.walls},
        )
        self.seeds = [self._state(point) for point in points]
        mass = np.array([s.density_kg_m3 * v for s, v in zip(self.seeds, volumes)])
        velocity = np.array([point["velocity_m_s"] for point in points])
        if np.any(velocity[: self.nodes] != 0):
            raise ValueError("Reference nodes must be stationary")
        # Fluid absolute stocks, absolute wall energies, integrated face fluxes,
        # support impulse exerted on fluid, external fluid/wall heat, entropy supply/production.
        self.wall_start = 3 * self.n
        self.face_start = self.wall_start + len(self.walls)
        self.tail = self.face_start + 3 * len(self.faces)
        self.y0 = np.r_[
            mass,
            mass * velocity,
            [
                m * (s.internal_energy_j_kg + v * v / 2)
                for m, s, v in zip(mass, self.seeds, velocity)
            ],
            [w["capacity_j_k"] * w["initial_temperature_k"] for w in self.walls],
            np.zeros(3 * len(self.faces) + 6 + self.n),
        ]

    @staticmethod
    def darcy_factor(reynolds, roughness_ratio):
        if not np.isfinite(reynolds) or reynolds <= 0 or not 0 <= roughness_ratio <= 0.05:
            raise ValueError("Reference Darcy domain")
        if reynolds < 2300:
            return 64 / reynolds
        if not 4000 <= reynolds <= 1e8:
            raise ValueError("Reference Darcy transition/Re domain")
        root = brentq(
            lambda x: x + 2 * log10(roughness_ratio / 3.7 + 2.51 * x / reynolds),
            1.0,
            100.0,
            xtol=1e-13,
        )
        return 1 / root**2

    def drag_force(self, i, edge, y, state):
        fields = (
            "circular_straight",
            "fully_developed",
            "constant_properties",
            "forced_convection",
            "negligible_compressibility",
        )
        scope = edge.get("friction_scope")
        if (
            not isinstance(scope, dict)
            or set(scope) != set(fields)
            or any(scope[k] is not True for k in fields)
        ):
            raise ValueError("Reference Darcy scope requires five true booleans")
        if state.region not in (1, 2) or not 273.15 <= state.temperature_k <= 623.15:
            raise ValueError("Reference Darcy single-phase transport domain")
        ts = iapws.saturation_temperature_k(state.pressure_pa * 1e-6)
        if (state.region == 1 and state.temperature_k >= ts) or (
            state.region == 2 and state.temperature_k <= ts
        ):
            raise ValueError("Reference Darcy strictly off-saturation required")
        mu = iapws.viscosity_pa_s(state.density_kg_m3, state.temperature_k)
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("Reference Darcy viscosity domain")
        velocity = float(y[self.n + i] / y[i])
        if velocity == 0:
            return 0.0
        diameter = edge["diameter_m"]
        re = state.density_kg_m3 * abs(velocity) * diameter / mu
        factor = self.darcy_factor(re, edge["roughness_m"] / diameter)
        return (
            -factor
            * self.volumes[i]
            * state.density_kg_m3
            * velocity
            * abs(velocity)
            / (2 * diameter)
        )

    def wall_conductance(self, wall, y, state):
        metadata = wall.get("correlation")
        if metadata is None:
            return wall["conductance_w_k"]
        target = wall["cell"]
        if "/" not in target:
            raise ValueError("Reference correlated wall requires channel cell")
        edge_id, _ = target.split("/")
        edge = next(e for e in self.spec["edges"] if e["id"] == edge_id)
        scope = metadata["scope"]
        flags = (
            "circular_straight",
            "fully_developed",
            "constant_properties",
            "forced_convection",
            "negligible_compressibility",
        )
        if set(scope) != set(flags) or any(scope[k] is not True for k in flags):
            raise ValueError("Reference heat-transfer scope")
        if (
            metadata["constant_wall_temperature"] is not True
            or metadata["thermally_developed"] is not True
        ):
            raise ValueError("Reference heat-transfer wall scope")
        if state.region not in (1, 2) or edge["roughness_m"] != 0:
            raise ValueError("Reference heat-transfer requires smooth single-phase flow")
        bulk = iapws.properties_pt(state.pressure_pa, state.temperature_k)
        velocity = abs(float(y[self.n + self.names.index(target)] / y[self.names.index(target)]))
        diameter = edge["diameter_m"]
        re = bulk.rho_kg_m3 * velocity * diameter / bulk.viscosity_pa_s
        pr = bulk.cp_j_kg_k * bulk.viscosity_pa_s / bulk.conductivity_w_m_k
        if re < 2300:
            nu = 3.66
        elif 4000 <= re <= 5e6 and 0.5 <= pr <= 2000:
            factor = (1.82 * log10(re) - 1.64) ** -2
            nu = (
                (factor / 8)
                * (re - 1000)
                * pr
                / (1 + 12.7 * sqrt(factor / 8) * (pr ** (2 / 3) - 1))
            )
        else:
            raise ValueError("Reference heat-transfer Reynolds/Prandtl domain")
        conductance = (
            nu
            * bulk.conductivity_w_m_k
            / diameter
            * pi
            * diameter
            * edge["length_m"]
            / edge["cells"]
        )
        if conductance > wall["conductance_w_k"]:
            raise ValueError("Reference wall conductance exceeds declared bound")
        return conductance

    @staticmethod
    def _state(point):
        if point["kind"] == "pt":
            return eos.pt(point["pressure_pa"], point["temperature_k"])
        if point["kind"] == "px":
            return eos.px(point["pressure_pa"], point["quality"])
        if point["kind"] == "rhou":
            return eos.rho_u(
                point["density_kg_m3"],
                point["internal_energy_j_kg"],
                pressure_seed_pa=point["pressure_seed_pa"],
                temperature_seed_k=point["temperature_seed_k"],
            )
        raise ValueError("Unknown reference initial chart")

    def thermos(self, y):
        result = []
        for i, seed in enumerate(self.seeds):
            m, momentum, energy = y[i], y[self.n + i], y[2 * self.n + i]
            if m <= 0:
                raise ValueError("Nonpositive reference mass")
            result.append(
                eos.rho_u(
                    float(m / self.volumes[i]),
                    float(energy / m - (momentum / m) ** 2 / 2),
                    pressure_seed_pa=seed.pressure_pa,
                    temperature_seed_k=seed.temperature_k,
                )
            )
        return result

    def rhs(self, time, y):
        states = self.thermos(y)
        rate = np.zeros_like(y)
        for k, (left, right, area) in enumerate(self.faces):

            def face(i):
                density = y[i] / self.volumes[i]
                velocity = 0.0 if i < self.nodes else y[self.n + i] / y[i]
                energy = y[2 * self.n + i] / self.volumes[i]
                p = states[i].pressure_pa
                return (
                    np.array([density, density * velocity, energy]),
                    np.array(
                        [density * velocity, density * velocity**2 + p, velocity * (energy + p)]
                    ),
                )

            ul, fl = face(left)
            ur, fr = face(right)
            flux = area * ((fl + fr) / 2 - self.wave * (ur - ul) / 2)
            rate[self.face_start + 3 * k : self.face_start + 3 * k + 3] = flux
            for i, sign in ((left, -1), (right, 1)):
                rate[i] += sign * flux[0]
                rate[2 * self.n + i] += sign * flux[2]
                if i < self.nodes:
                    rate[self.tail] -= sign * flux[1]
                else:
                    rate[self.n + i] += sign * flux[1]
        for i, edge in self.drag_cells:
            force = self.drag_force(i, edge, y, states[i])
            rate[self.n + i] += force
            rate[self.tail + 6 + i] = force
            rate[self.tail + 5] -= force * y[self.n + i] / y[i] / states[i].temperature_k
        for i, name in enumerate(self.names):
            power = self.controls["fluid"][name]
            rate[2 * self.n + i] += power
            rate[self.tail + 1] += power
            rate[self.tail + 3] += power / states[i].temperature_k
        for j, wall in enumerate(self.walls):
            i = self.ports[j]
            tw = y[self.wall_start + j] / wall["capacity_j_k"]
            tf = states[i].temperature_k
            exchange = self.wall_conductance(wall, y, states[i]) * (tw - tf)
            ambient = wall["ambient_ua_w_k"] * (wall["ambient_temperature_k"] - tw)
            power = self.controls["walls"][wall["id"]]
            external = power + ambient
            rate[2 * self.n + i] += exchange
            rate[self.wall_start + j] = external - exchange
            rate[self.tail + 2] += external
            rate[self.tail + 3] += power / tw + ambient / wall["ambient_temperature_k"]
            rate[self.tail + 4] += exchange * (1 / tf - 1 / tw) + ambient * (
                1 / tw - 1 / wall["ambient_temperature_k"]
            )
        return rate

    def observation(self, time, y):
        states = self.thermos(y)
        faces = y[self.face_start : self.tail].reshape(-1, 3)
        support = np.zeros(self.n)
        for k, (left, right, _) in enumerate(self.faces):
            if left < self.nodes:
                support[left] += faces[k, 1]
            if right < self.nodes:
                support[right] -= faces[k, 1]
        entropy = sum(
            y[i] * s.entropy_j_kg_k - self.y0[i] * seed.entropy_j_kg_k
            for i, (s, seed) in enumerate(zip(states, self.seeds))
        )
        entropy += sum(
            w["capacity_j_k"] * np.log(y[self.wall_start + j] / self.y0[self.wall_start + j])
            for j, w in enumerate(self.walls)
        )
        observation = dict(
            time_s=float(time),
            cells=[
                dict(
                    id=name,
                    **states[i].as_dict(),
                    mass_kg=float(y[i]),
                    momentum_kg_m_s=float(y[self.n + i]),
                    total_energy_j=float(y[2 * self.n + i]),
                )
                for i, name in enumerate(self.names)
            ],
            walls=[
                dict(
                    id=w["id"],
                    temperature_k=float(y[self.wall_start + j] / w["capacity_j_k"]),
                    energy_change_j=float(y[self.wall_start + j] - self.y0[self.wall_start + j]),
                )
                for j, w in enumerate(self.walls)
            ],
            local_ledgers=dict(
                {
                    name: faces[:, j].tolist()
                    for j, name in enumerate(
                        [
                            "integrated_face_mass_kg",
                            "integrated_face_momentum_kg_m_s",
                            "integrated_face_energy_j",
                        ]
                    )
                },
                node_support_impulse_kg_m_s=support.tolist(),
                drag_impulse_kg_m_s=y[self.tail + 6 : self.tail + 6 + self.n].tolist(),
            ),
            global_ledger=dict(
                support_impulse_kg_m_s=float(y[self.tail]),
                external_fluid_heat_j=float(y[self.tail + 1]),
                external_wall_heat_j=float(y[self.tail + 2]),
                total_entropy_change_j_k=float(entropy),
                external_entropy_j_k=float(y[self.tail + 3]),
                thermal_entropy_production_j_k=float(y[self.tail + 4]),
                drag_entropy_production_j_k=float(y[self.tail + 5]),
            ),
        )
        observation["wall_temperatures_k"] = [w["temperature_k"] for w in observation["walls"]]
        observation["global_integrals"] = dict(observation["global_ledger"])
        return observation


def solve(spec, program, method="DOP853", *, rtol=2e-11, atol=1e-10, max_step=1e-4):
    """Return initial and call-end observations; split exactly at control changes."""
    assembly = Assembly(spec)
    y, time = assembly.y0.copy(), 0.0
    observations = [assembly.observation(time, y)]
    for call in program:
        if call["command"] is not None:
            assembly.controls = deepcopy(call["command"]["controls"])
        duration = call["duration_s"]
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid reference duration")
        target = time + duration
        solved = solve_ivp(
            assembly.rhs, (time, target), y, method=method, rtol=rtol, atol=atol, max_step=max_step
        )
        if not solved.success:
            raise ValueError("Network reference IVP failed: " + solved.message)
        y, time = solved.y[:, -1], target
        observations.append(assembly.observation(time, y))
    return observations
