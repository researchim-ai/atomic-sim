"""Monolithic two-pump, two-cycle thermal assembly with a shared exchanger wall.

Both cycles retain their fluid/shaft inertia, thermal walls and conservative
energy ledgers. Shared-wall heat enters each cycle's explicit thermal port;
it is removed from external heat when computing the assembly energy balance.
Constant density/cp imply no thermal feedback on hydraulic coefficients.
"""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag, csc_matrix

from core.network import NetworkError, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.circulation_loop import CirculationLoop
from physics.hydraulics.pump_loop import PumpCirculationLoop
from physics.thermal.two_stream_exchanger import digest

PASSPORT = ModelPassport(
    model_id="coupled-pump-exchanger-cycles",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Two inertial pump/shaft cycles coupled through a finite-capacity shared wall",
    approximations=(
        "Two closed uniform-area constant-density/cp cycles; signed flow and shaft speeds",
        "Displacement pumps with linear leakage; casing receives leakage heat",
        "Shared mixed-cell wall with reciprocal positive conductances and explicit heat rejection",
        "No acoustics, compressibility, phase change, gravity or manufacturer pump maps",
        "One simultaneous ODE; no sequential thermal/hydraulic splitting",
    ),
    applicability="Reduced single-phase thermal/hydraulic component assemblies",
    data_sources=(
        "I*mdot=H-R*m*abs(m); J*omegadot=tau_motor-D*H-B*omega",
        "Q_s=UA_s*(Tw-Ts); Cw*Twdot=-Q_a-Q_b+P-UAamb*(Tw-Tamb)",
    ),
)
CONTROL_FIELDS = {"a_motor_torque_nm", "b_motor_torque_nm"}
CELL_FIELDS = {
    "a_segment",
    "b_segment",
    "wall_capacity_j_k",
    "a_ua_w_k",
    "b_ua_w_k",
    "initial_wall_temperature_k",
    "ambient_ua_w_k",
    "wall_power_w",
}


def controls(raw):
    if not isinstance(raw, dict) or set(raw) != CONTROL_FIELDS:
        raise NetworkError("Complete two-motor torque vector required")
    return {key: number(value, key) for key, value in raw.items()}


class CoupledPumpExchanger:
    def __init__(self, loop_initials, exchanger, *, energy_tolerance_j=1e-4):
        if not isinstance(loop_initials, dict) or set(loop_initials) != {"a", "b"}:
            raise NetworkError("Two explicit initial pump-cycle snapshots required")
        self._models = [PumpCirculationLoop.from_snapshot(loop_initials[s]) for s in ("a", "b")]
        for model in self._models:
            if (
                model.time_s != 0
                or model.events
                or not np.array_equal(
                    model._state, np.r_[model._loop._state, model._omega0, 0.0, 0.0, 0.0]
                )
            ):
                raise NetworkError("Coupling requires untouched initial pump cycles")
        self._initials = deepcopy(loop_initials)
        if not isinstance(exchanger, dict) or set(exchanger) != {"cells", "ambient_temperature_k"}:
            raise NetworkError("Exact shared-wall exchanger configuration required")
        raw_cells = exchanger["cells"]
        if not isinstance(raw_cells, list) or not 1 <= len(raw_cells) <= 32:
            raise NetworkError("Shared exchanger requires 1..32 cells")
        ambient = number(exchanger["ambient_temperature_k"], "ambient_temperature_k", positive=True)
        cells = []
        for raw in raw_cells:
            if not isinstance(raw, dict) or set(raw) != CELL_FIELDS:
                raise NetworkError("Exact shared exchanger cell fields required")
            cell = {}
            for side, model in zip(("a", "b"), self._models):
                key = f"{side}_segment"
                if not isinstance(raw[key], str) or raw[key] not in model._loop.names:
                    raise NetworkError("Unknown exchanger segment mapping")
                cell[key] = raw[key]
            for key in CELL_FIELDS - {"a_segment", "b_segment"}:
                positive = key in {"wall_capacity_j_k", "initial_wall_temperature_k"}
                cell[key] = number(raw[key], key, positive=positive)
                if cell[key] < 0:
                    raise NetworkError("Negative exchanger conductance/heater power")
            cells.append(cell)
        for side in ("a", "b"):
            if len({c[f"{side}_segment"] for c in cells}) != len(cells):
                raise NetworkError("Each cycle segment may map to at most one exchanger cell")
        self._exchanger = dict(cells=cells, ambient_temperature_k=ambient)
        self.energy_tolerance_j = number(energy_tolerance_j, "energy_tolerance_j", positive=True)
        self.n = len(cells)
        self._sizes = [len(m._state) for m in self._models]
        self._offsets = [0, self._sizes[0]]
        self._wall_start = sum(self._sizes)
        self._ports_start = self._wall_start + 2 * self.n
        self._wall0 = np.array([c["initial_wall_temperature_k"] for c in cells])
        self._capacity = np.array([c["wall_capacity_j_k"] for c in cells])
        self._ua = np.array([[c[f"{s}_ua_w_k"] for c in cells] for s in ("a", "b")])
        self._ambient = np.array([c["ambient_ua_w_k"] for c in cells])
        self._heater = np.array([c["wall_power_w"] for c in cells])
        self._mapping = [
            np.array([m._loop.names.index(c[f"{s}_segment"]) for c in cells])
            for s, m in zip(("a", "b"), self._models)
        ]
        self._state = np.concatenate([m._state for m in self._models] + [np.zeros(2 * self.n + 3)])
        self._initial_state = self._state.copy()
        self._initial_controls = dict(
            zip(("a_motor_torque_nm", "b_motor_torque_nm"), (m._torque for m in self._models))
        )
        self._controls = deepcopy(self._initial_controls)
        self.time_s, self._events = 0.0, []
        self.observation()

    @property
    def configuration(self):
        return dict(
            loop_initials=deepcopy(self._initials),
            exchanger=deepcopy(self._exchanger),
            energy_tolerance_j=self.energy_tolerance_j,
        )

    @property
    def boundary_controls(self):
        return deepcopy(self._controls)

    @property
    def events(self):
        return deepcopy(self._events)

    def _heat(self, state):
        y = np.asarray(state, dtype=float)
        if y.shape != self._state.shape or not np.isfinite(y).all():
            raise NetworkError("Invalid coupled exchanger state")
        wall = self._wall0 + y[self._wall_start : self._wall_start + self.n]
        if np.any(wall <= 0):
            raise NetworkError("Nonpositive shared-wall temperature")
        heat = []
        for model, offset, mapping, ua in zip(self._models, self._offsets, self._mapping, self._ua):
            fluid = model._loop._temperature_reference[mapping] + y[offset + 1 + mapping]
            heat.append(ua * (wall - fluid))
        external = self._heater + self._ambient * (self._exchanger["ambient_temperature_k"] - wall)
        wall_power = external - heat[0] - heat[1]
        if not np.isfinite([heat[0], heat[1], wall_power]).all():
            raise NetworkError("Nonfinite shared-wall power")
        return wall, heat, wall_power, external

    def derivatives(self, stamp, state):
        wall, heat, wall_power, external = self._heat(state)
        rates = np.zeros_like(state, dtype=float)
        for i, (model, offset, size, mapping) in enumerate(
            zip(self._models, self._offsets, self._sizes, self._mapping)
        ):
            rates[offset : offset + size] = model.derivatives(stamp, state[offset : offset + size])
            rates[offset + 1 + mapping] += heat[i] / model._loop.capacity[mapping]
            rates[offset + model._size - 1] += sum(heat[i])
        rates[self._wall_start : self._wall_start + self.n] = wall_power / self._capacity
        rates[self._wall_start + self.n : self._ports_start] = wall_power
        rates[self._ports_start :] = [sum(heat[0]), sum(heat[1]), sum(external)]
        if not np.isfinite(rates).all():
            raise NetworkError("Nonfinite coupled derivatives")
        return rates

    def initialization_report(self):
        if self._controls != self._initial_controls:
            return type(self)(**self.configuration).initialization_report()
        rates = self.derivatives(0.0, self._initial_state)
        reports = {}
        for side, model, offset in zip(("a", "b"), self._models, self._offsets):
            loop = model._loop
            reports[side] = dict(
                momentum_residual_pa=float(loop.inertia_total * rates[offset]),
                shaft_residual_nm=float(model._pump["inertia_kg_m2"] * rates[offset + model._size]),
                fluid_energy_residual_w=(
                    loop.capacity * rates[offset + 1 : offset + 1 + loop.n]
                ).tolist(),
                casing_energy_residual_w=(
                    loop.wall_capacity * rates[offset + 1 + loop.n : offset + 1 + 2 * loop.n]
                ).tolist(),
            )
        return dict(
            loops=reports,
            wall_energy_residual_w=(
                self._capacity * rates[self._wall_start : self._wall_start + self.n]
            ).tolist(),
        )

    def analytic_jacobian(self, stamp, state):
        self._heat(state)
        blocks = [
            model.analytic_jacobian(stamp, state[offset : offset + size])
            for model, offset, size in zip(self._models, self._offsets, self._sizes)
        ]
        jac = np.zeros((len(state), len(state)))
        jac[: self._wall_start, : self._wall_start] = block_diag(blocks).toarray()
        for cell in range(self.n):
            wcol = self._wall_start + cell
            irow = self._wall_start + self.n + cell
            for side, (model, offset, mapping) in enumerate(
                zip(self._models, self._offsets, self._mapping)
            ):
                fcol = offset + 1 + mapping[cell]
                ua = self._ua[side, cell]
                capacity = model._loop.capacity[mapping[cell]]
                for col, sign in ((fcol, -1), (wcol, 1)):
                    jac[fcol, col] += sign * ua / capacity
                    jac[offset + model._size - 1, col] += sign * ua
                    jac[self._ports_start + side, col] += sign * ua
                    jac[wcol, col] -= sign * ua / self._capacity[cell]
                    jac[irow, col] -= sign * ua
            jac[wcol, wcol] -= self._ambient[cell] / self._capacity[cell]
            jac[irow, wcol] -= self._ambient[cell]
            jac[self._ports_start + 2, wcol] -= self._ambient[cell]
        if not np.isfinite(jac).all():
            raise NetworkError("Nonfinite coupled Jacobian")
        return csc_matrix(jac)

    def observation(self, state=None, time_s=None):
        y = self._state if state is None else np.asarray(state, dtype=float)
        stamp = self.time_s if time_s is None else number(time_s, "observation time")
        wall, heat, wall_power, external = self._heat(y)
        observed = [
            model.observation(y[offset : offset + size], stamp)
            for model, offset, size in zip(self._models, self._offsets, self._sizes)
        ]
        wall_energy = self._capacity * y[self._wall_start : self._wall_start + self.n]
        wall_integrals = y[self._wall_start + self.n : self._ports_start]
        ports = y[self._ports_start :]
        shared_residual = float(sum(wall_energy) + sum(ports[:2]) - ports[2])
        total_energy = sum(
            r["loop"]["stored_energy_change_j"] + r["shaft_energy_change_j"] for r in observed
        ) + sum(wall_energy)
        external_heat = (
            sum(r["external_ambient_heater_heat_j"] for r in observed) - sum(ports[:2]) + ports[2]
        )
        total_residual = float(
            total_energy
            - external_heat
            - sum(r["motor_work_j"] - r["bearing_heat_j"] for r in observed)
        )
        local_wall_residual = wall_energy - wall_integrals
        if (
            max(abs(shared_residual), abs(total_residual), max(abs(local_wall_residual)))
            > self.energy_tolerance_j
        ):
            raise NetworkError("Coupled exchanger energy ledger exceeded")
        if not np.isfinite([shared_residual, total_residual]).all():
            raise NetworkError("Nonfinite coupled energy ledger")
        rates = CoupledPumpExchanger.derivatives(self, stamp, y)
        power_residuals = {}
        for side, model, offset, row, coupling in zip(
            ("a", "b"), self._models, self._offsets, observed, heat
        ):
            loop = model._loop
            n = loop.n
            thermal = (
                loop.capacity @ rates[offset + 1 : offset + 1 + n]
                + loop.wall_capacity @ rates[offset + 1 + n : offset + 1 + 2 * n]
            )
            mechanical = (
                loop.inertia_total * y[offset] * rates[offset] / loop.rho
                + model._pump["inertia_kg_m2"]
                * y[offset + model._size]
                * rates[offset + model._size]
            )
            casing = loop._wall_reference + y[offset + 1 + n : offset + 1 + 2 * n]
            ambient_heater = sum(
                loop.heater + loop.ambient_ua * (loop.ambient_temperature_k - casing)
            )
            residual = (
                thermal
                + mechanical
                - row["motor_power_w"]
                + row["bearing_power_w"]
                - ambient_heater
                - sum(coupling)
            )
            if not np.isfinite(residual) or abs(residual) > 1e-6:
                raise NetworkError("Coupled cycle instantaneous energy mismatch")
            power_residuals[side] = float(residual)
        return dict(
            time_s=stamp,
            loops=dict(zip(("a", "b"), observed)),
            controls=self.boundary_controls,
            wall_temperature_k=wall.tolist(),
            heat_into_loops_w={s: heat[i].tolist() for i, s in enumerate(("a", "b"))},
            wall_net_power_w=wall_power.tolist(),
            wall_external_power_w=external.tolist(),
            shared_wall_energy_change_j=float(sum(wall_energy)),
            wall_energy_residual_j=local_wall_residual.tolist(),
            coupling_heat_into_loops_j=dict(zip(("a", "b"), map(float, ports[:2]))),
            wall_external_heat_j=float(ports[2]),
            shared_energy_residual_j=shared_residual,
            stored_energy_change_j=float(total_energy),
            external_heat_j=float(external_heat),
            total_energy_residual_j=total_residual,
            loop_power_residual_w=power_residuals,
            reciprocal_heat_residual_w=float(
                sum(wall_power) + sum(heat[0]) + sum(heat[1]) - sum(external)
            ),
        )

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.1, atol_scale=0.1):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported coupled solver policy")
        end = self.time_s + dt
        if not np.isfinite(end) or end <= self.time_s:
            raise NetworkError("Coupled clock cannot advance")
        # All mutation occurs after solver completion and observation acceptance.
        atol = np.full(len(self._state), 1e-8 * scale)
        for model, offset in zip(self._models, self._offsets):
            atol[offset] = 1e-11 * scale
            atol[offset + 1 : offset + 1 + 2 * model._loop.n] = 1e-12 * scale
            atol[offset + model._size] = 1e-11 * scale
        atol[self._wall_start : self._wall_start + self.n] = 1e-12 * scale
        result = solve_ivp(
            self.derivatives,
            (self.time_s, end),
            self._state,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=step,
            jac=self.analytic_jacobian,
        )
        if not result.success:
            raise NetworkError("Coupled integration failed: " + result.message)
        observed = self.observation(result.y[:, -1], end)
        self._state, self.time_s = result.y[:, -1].copy(), end
        return observed

    def set_controls(self, boundary_controls):
        proposed = controls(boundary_controls)
        if len(self._events) >= 10000:
            raise NetworkError("Coupled control history resource limit exceeded")
        previous = self.boundary_controls
        try:
            self._controls = proposed
            for side, model in zip(("a", "b"), self._models):
                model._torque = proposed[f"{side}_motor_torque_nm"]
            self.observation()
        except Exception:
            self._controls = previous
            for side, model in zip(("a", "b"), self._models):
                model._torque = previous[f"{side}_motor_torque_nm"]
            raise
        self._events.append(dict(t=self.time_s, before=previous, after=deepcopy(proposed)))

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=digest(self.configuration),
            state=self._state.tolist(),
            time_s=self.time_s,
            controls=self.boundary_controls,
            events=self.events,
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "configuration",
            "configuration_sha256",
            "state",
            "time_s",
            "controls",
            "events",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
            or digest(payload["configuration"]) != payload["configuration_sha256"]
            or set(payload["configuration"]) != {"loop_initials", "exchanger", "energy_tolerance_j"}
        ):
            raise NetworkError("Incompatible coupled exchanger snapshot")
        model = cls(**payload["configuration"])
        stamp = number(payload["time_s"], "snapshot time")
        events = payload["events"]
        if stamp < 0 or not isinstance(events, list) or len(events) > 10000:
            raise NetworkError("Invalid coupled snapshot time/history")
        previous_time, boundary = -1.0, model.boundary_controls
        for event in events:
            if not isinstance(event, dict) or set(event) != {"t", "before", "after"}:
                raise NetworkError("Invalid coupled control event")
            t = number(event["t"], "event time")
            if not 0 <= t <= stamp or t < previous_time or controls(event["before"]) != boundary:
                raise NetworkError("Broken coupled control history")
            previous_time, boundary = t, controls(event["after"])
        if boundary != controls(payload["controls"]):
            raise NetworkError("Coupled active controls differ from history")
        state = payload["state"]
        if not isinstance(state, list) or len(state) != len(model._state):
            raise NetworkError("Invalid coupled snapshot state shape")
        state = np.array([number(v, "snapshot state") for v in state])
        model._controls = boundary
        for side, pump in zip(("a", "b"), model._models):
            pump._torque = boundary[f"{side}_motor_torque_nm"]
        model.observation(state, stamp)
        if stamp == 0 and not np.array_equal(state, model._initial_state):
            raise NetworkError("Noninitial coupled state at zero time")
        model._state, model.time_s, model._events = state, stamp, deepcopy(events)
        return model


def steady_coupled_exchanger(loop_initials, exchanger, *, energy_tolerance_j=1e-4):
    """Solve mechanical and coupled thermal equilibrium without a warmup trajectory."""
    given = CoupledPumpExchanger(loop_initials, exchanger, energy_tolerance_j=energy_tolerance_j)

    def rebuild(model, flow, omega, fluid_temperature, wall_temperature):
        loop = model._loop
        initial = CirculationLoop(
            topology=loop.topology,
            fluid=loop.fluid,
            reference=loop.reference,
            ambient_temperature_k=loop.ambient_temperature_k,
            energy_tolerance_j=loop.energy_tolerance_j,
            initial_mass_flow_kg_s=flow,
            initial_temperature_k=dict(zip(loop.names, map(float, fluid_temperature))),
            initial_wall_temperature_k=dict(zip(loop.names, map(float, wall_temperature))),
        )
        return PumpCirculationLoop(initial.snapshot(), model.pump, omega, model._torque0).snapshot()

    initials = {}
    mechanical = []
    for side, model in zip(("a", "b"), given._models):
        loop, p = model._loop, model._pump
        displacement, leakage, drag = (
            p["displacement_m3_rad"],
            p["leakage_m3_s_pa"],
            p["drag_nm_s_rad"],
        )
        alpha = loop.resistance_total * (displacement + drag * leakage / displacement)
        beta = drag / (displacement * loop.rho)
        torque = model._torque0
        magnitude = (
            0.0
            if torque == 0
            else 2 * abs(torque) / (beta + np.sqrt(beta**2 + 4 * alpha * abs(torque)))
        )
        flow = float(np.copysign(magnitude, torque))
        head = loop.resistance_total * flow * abs(flow)
        omega = (flow / loop.rho + leakage * head) / displacement
        mechanical.append((flow, omega))
        initials[side] = rebuild(
            model, flow, omega, loop._temperature_reference, loop._wall_reference
        )
    probe = CoupledPumpExchanger(initials, exchanger, energy_tolerance_j=energy_tolerance_j)
    indices = []
    for model, offset in zip(probe._models, probe._offsets):
        indices.extend(range(offset + 1, offset + 1 + 2 * model._loop.n))
    indices.extend(range(probe._wall_start, probe._wall_start + probe.n))
    matrix = probe.analytic_jacobian(0.0, probe._state).toarray()[np.ix_(indices, indices)]
    if np.linalg.matrix_rank(matrix) != len(indices):
        raise NetworkError("Coupled thermal steady state is unanchored or numerically singular")
    try:
        delta = np.linalg.solve(matrix, -probe.derivatives(0.0, probe._state)[indices])
    except np.linalg.LinAlgError as exc:
        raise NetworkError("Coupled steady thermal solve failed") from exc
    candidate = probe._state.copy()
    candidate[indices] = delta
    # Inventory/pressure domain is checked before rebuilding zero-ledger initial states.
    probe._heat(candidate)
    for side, model, offset, (flow, omega) in zip(
        ("a", "b"), probe._models, probe._offsets, mechanical
    ):
        loop = model._loop
        fluid = loop._temperature_reference + candidate[offset + 1 : offset + 1 + loop.n]
        casing = loop._wall_reference + candidate[offset + 1 + loop.n : offset + 1 + 2 * loop.n]
        initials[side] = rebuild(model, flow, omega, fluid, casing)
    shared = deepcopy(exchanger)
    for cell, temp in zip(
        shared["cells"], probe._wall0 + candidate[probe._wall_start : probe._wall_start + probe.n]
    ):
        cell["initial_wall_temperature_k"] = float(temp)
    result = CoupledPumpExchanger(initials, shared, energy_tolerance_j=energy_tolerance_j)
    residual = result.initialization_report()
    thermal = list(residual["wall_energy_residual_w"])
    for report in residual["loops"].values():
        thermal.extend(report["fluid_energy_residual_w"] + report["casing_energy_residual_w"])
        if abs(report["momentum_residual_pa"]) > 1e-6 or abs(report["shaft_residual_nm"]) > 1e-8:
            raise NetworkError("Coupled mechanical steady residual exceeds budget")
    if max(abs(v) for v in thermal) > 1e-6:
        raise NetworkError("Coupled thermal steady residual exceeds 1e-6 W")
    return result
