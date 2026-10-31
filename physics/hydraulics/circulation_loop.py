"""Reduced uniform-area, constant-property closed circulation loop.

All segments share one signed mass flow by exact incompressible continuity.
I_i=L_i/A, R_i=K_i/(2*rho*A**2), I*mdot=H-R*m*abs(m).
Kinetic energy is I*m**2/(2*rho); friction heats the liquid once. Ideal body
head supplies H*m/rho mechanical power; this is NOT a real pump/shaft model.
Pressure faces are algebraic and referenced to an explicit gauge. No acoustic,
phase, gravity, variable-area or arbitrary branched-network qualification.
"""

import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix

from core.continuation import ImplicitContinuation
from core.network import NetworkError, PortSpec, identifier, number
from core.status import ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="incompressible-thermal-circulation-loop",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Nonlinear closed circulation mode with thermal walls and total-energy ledger",
    approximations=(
        "constant positive density/cp; common area; connected single cycle",
        "one global inertial mass-flow mode, quadratic resistance",
        "inward conservative ports; ideal prescribed body head, no shaft efficiency",
        "constant wall capacities/UA; optional ambient heat sink",
        "no compressibility, acoustic propagation, phase change or gravity",
    ),
    applicability="Reduced closed liquid circulation with positive absolute temperatures and pressures",
    data_sources=(
        "I dm/dt=H-R*m*abs(m); K=I*m^2/(2*rho)",
        "C_i dT_i/dt=cp*abs(m)*(T_donor-T_i)+R_i*abs(m)^3/rho+UA_i*(Tw_i-T_i)",
    ),
)
SEGMENT_ID = "incompressible-loop-segment"
PARAMETERS = {
    "length_m",
    "loss_coefficient",
    "wall_capacity_j_k",
    "fluid_wall_ua_w_k",
    "ambient_ua_w_k",
    "wall_power_w",
    "drive_pressure_pa",
}
PORTS = tuple(
    PortSpec(
        name,
        "hydraulic",
        medium="constant-property-circulation-liquid",
        energy_convention="total_enthalpy",
    )
    for name in ("left", "right")
)


def topology_order(topology, reference_segment):
    if (
        not isinstance(topology, dict)
        or set(topology) != {"schema_version", "nodes", "connections"}
        or type(topology["schema_version"]) is not int
        or topology["schema_version"] != 1
    ):
        raise NetworkError("Circulation requires topology schema 1")
    if (
        not isinstance(topology["nodes"], list)
        or not 2 <= len(topology["nodes"]) <= 32
        or not isinstance(topology["connections"], list)
    ):
        raise NetworkError("Circulation supports 2..32 explicit segments in one closed cycle")
    nodes = {}
    for raw in topology["nodes"]:
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "model_id",
            "model_version",
            "parameters",
        }:
            raise NetworkError("Invalid circulation segment fields")
        name = identifier(raw["id"])
        if name in nodes or raw["model_id"] != SEGMENT_ID or raw["model_version"] != "1.0.0":
            raise NetworkError("Duplicate segment or unsupported circulation passport/version")
        if not isinstance(raw["parameters"], dict) or set(raw["parameters"]) != PARAMETERS:
            raise NetworkError("Exact circulation segment parameters required")
        params = {k: number(v, k) for k, v in raw["parameters"].items()}
        if (
            params["length_m"] <= 0
            or params["wall_capacity_j_k"] <= 0
            or any(
                params[k] < 0
                for k in PARAMETERS - {"length_m", "wall_capacity_j_k", "drive_pressure_pa"}
            )
        ):
            raise NetworkError("Invalid length, loss, capacity, conductance or heater power")
        nodes[name] = dict(id=name, model_id=SEGMENT_ID, model_version="1.0.0", parameters=params)
    if reference_segment not in nodes or len(topology["connections"]) != len(nodes):
        raise NetworkError("Pressure reference and one connection per segment required")
    outgoing = {}
    incoming = {}
    links = {}
    ids = set()
    for raw in topology["connections"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "from", "to"}:
            raise NetworkError("Invalid circulation connection")
        name = identifier(raw["id"])
        if name in ids:
            raise NetworkError("Duplicate circulation connection")
        ids.add(name)
        left, right = raw["from"], raw["to"]
        if not isinstance(left, str) or not isinstance(right, str):
            raise NetworkError("Port names required")
        parts_a, parts_b = left.split("."), right.split(".")
        if len(parts_a) != 2 or len(parts_b) != 2 or parts_a[1] != "right" or parts_b[1] != "left":
            raise NetworkError(
                "Cycle orientation requires right-to-left connections; physical flow may reverse"
            )
        a, b = parts_a[0], parts_b[0]
        if a not in nodes or b not in nodes or a == b or a in outgoing or b in incoming:
            raise NetworkError("Unknown, self-connected or multiply connected circulation port")
        outgoing[a] = b
        incoming[b] = a
        links[a] = dict(id=name, **{"from": left, "to": right})
    order = []
    node = reference_segment
    while node not in order:
        order.append(node)
        if node not in outgoing:
            raise NetworkError("Unconnected circulation port")
        node = outgoing[node]
    if node != reference_segment or len(order) != len(nodes):
        raise NetworkError("Disconnected cycle islands")
    return dict(
        schema_version=1, nodes=[nodes[n] for n in order], connections=[links[n] for n in order]
    )


class CirculationLoop:
    def __init__(
        self,
        topology,
        fluid,
        reference,
        ambient_temperature_k,
        initial_mass_flow_kg_s,
        initial_temperature_k,
        initial_wall_temperature_k,
        *,
        energy_tolerance_j=1e-5,
    ):
        if not isinstance(fluid, dict) or set(fluid) != {
            "density_kg_m3",
            "heat_capacity_j_kg_k",
            "area_m2",
        }:
            raise NetworkError("Explicit constant-property liquid and common area required")
        if not isinstance(reference, dict) or set(reference) != {"segment", "pressure_pa"}:
            raise NetworkError("Explicit pressure gauge required")
        self._fluid = {k: number(v, k, positive=True) for k, v in fluid.items()}
        self._reference = dict(
            segment=identifier(reference["segment"]),
            pressure_pa=number(reference["pressure_pa"], "pressure_pa", positive=True),
        )
        self._topology = topology_order(topology, self.reference["segment"])
        self.names = tuple(n["id"] for n in self.topology["nodes"])
        self.n = len(self.names)
        self.ambient_temperature_k = number(
            ambient_temperature_k, "ambient_temperature_k", positive=True
        )
        self.energy_tolerance_j = number(energy_tolerance_j, "energy_tolerance_j", positive=True)
        p = [v["parameters"] for v in self.topology["nodes"]]

        def array(key):
            return np.array([v[key] for v in p])

        self.rho = self.fluid["density_kg_m3"]
        self.cp = self.fluid["heat_capacity_j_kg_k"]
        self.area = self.fluid["area_m2"]
        self.length = array("length_m")
        self.inertia = self.length / self.area
        self.resistance = array("loss_coefficient") / (2 * self.rho * self.area**2)
        self.capacity = self.rho * self.area * self.length * self.cp
        self.wall_capacity = array("wall_capacity_j_k")
        self.ua = array("fluid_wall_ua_w_k")
        self.ambient_ua = array("ambient_ua_w_k")
        self.heater = array("wall_power_w")
        self._initial_heads = array("drive_pressure_pa")
        self._heads = self._initial_heads.copy()
        self.inertia_total = float(sum(self.inertia))
        self.resistance_total = float(sum(self.resistance))
        if self.resistance_total <= 0:
            raise NetworkError("Positive total resistance required")
        self._temperature_reference = self._temperatures(initial_temperature_k)
        self._wall_reference = self._temperatures(initial_wall_temperature_k)
        self.initial_flow = number(initial_mass_flow_kg_s, "initial_mass_flow_kg_s")
        self._state = np.r_[self.initial_flow, np.zeros(2 * self.n + 2)]
        self.time_s = 0.0
        self.events = []
        self.last_events = ()
        self._continuation = None
        self._policy = None
        self._horizon = None
        self._initial_kinetic = self.inertia_total * self.initial_flow**2 / (2 * self.rho)
        self.initial_energy_j = float(
            self.capacity @ self._temperature_reference
            + self.wall_capacity @ self._wall_reference
            + self._initial_kinetic
        )
        if not np.isfinite(self.initial_energy_j):
            raise NetworkError("Initial inventory overflow")
        for values in (
            self.length,
            self.inertia,
            self.resistance,
            self.capacity,
            self.wall_capacity,
            self.ua,
            self.ambient_ua,
            self.heater,
        ):
            values.setflags(write=False)
        self.observation()

    def configuration_sha256(self):
        config = dict(
            topology=self.topology,
            fluid=self.fluid,
            reference=self.reference,
            ambient_temperature_k=self.ambient_temperature_k,
            initial_mass_flow_kg_s=self.initial_flow,
            initial_temperature_k=self._temperature_reference.tolist(),
            initial_wall_temperature_k=self._wall_reference.tolist(),
            energy_tolerance_j=self.energy_tolerance_j,
        )
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def topology(self):
        return deepcopy(self._topology)

    @property
    def fluid(self):
        return dict(self._fluid)

    @property
    def reference(self):
        return dict(self._reference)

    def _temperatures(self, values):
        if not isinstance(values, dict) or set(values) != set(self.names):
            raise NetworkError("Explicit temperatures required for all and only segments")
        return np.array([number(values[n], n, positive=True) for n in self.names])

    def _check_state(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (2 * self.n + 3,) or not np.isfinite(y).all():
            raise NetworkError("Invalid circulation state vector")
        tf = self._temperature_reference + y[1 : self.n + 1]
        tw = self._wall_reference + y[self.n + 1 : 2 * self.n + 1]
        if np.any(tf <= 0) or np.any(tw <= 0):
            raise NetworkError("Nonpositive absolute temperature")
        return tf, tw

    def _rates(self, y, heads=None):
        y = np.asarray(y, dtype=float)
        tf, tw = self._check_state(y)
        heads = self._heads if heads is None else heads
        m = float(y[0])
        dm = (float(sum(heads)) - self.resistance_total * m * abs(m)) / self.inertia_total
        delta_p = self.inertia * dm + self.resistance * m * abs(m) - heads
        pressure = np.r_[
            self.reference["pressure_pa"], self.reference["pressure_pa"] - np.cumsum(delta_p)[:-1]
        ]
        bad_pressure = np.flatnonzero(~np.isfinite(pressure) | (pressure <= 0))
        if len(bad_pressure):
            i = int(bad_pressure[0])
            raise NetworkError(f"Invalid pressure at {self.names[i]}.left: {pressure[i]} Pa")
        donor = np.roll(np.arange(self.n), 1 if m >= 0 else -1)
        delta_t = (self._temperature_reference[donor] - self._temperature_reference) + (
            y[1 : self.n + 1][donor] - y[1 : self.n + 1]
        )
        qwall = self.ua * (
            (self._wall_reference - self._temperature_reference)
            + (y[self.n + 1 : 2 * self.n + 1] - y[1 : self.n + 1])
        )
        qambient = self.ambient_ua * (
            (self.ambient_temperature_k - self._wall_reference) - y[self.n + 1 : 2 * self.n + 1]
        )
        dissipation = self.resistance * abs(m) ** 3 / self.rho
        fluid_rate = (self.cp * abs(m) * delta_t + dissipation + qwall) / self.capacity
        wall_rate = (self.heater - qwall + qambient) / self.wall_capacity
        rhs = np.r_[
            dm,
            fluid_rate,
            wall_rate,
            float(sum(heads)) * m / self.rho,
            float(sum(self.heater + qambient)),
        ]
        if not np.isfinite(rhs).all():
            raise NetworkError("Nonfinite circulation rates")
        return dict(
            rhs=rhs,
            pressure=pressure,
            fluid_temperature=tf,
            wall_temperature=tw,
            donor=donor,
            delta_t=delta_t,
            wall_heat=qwall,
            ambient_heat=qambient,
            dissipation=dissipation,
        )

    def derivatives(self, _t, y):
        return self._rates(y)["rhs"]

    def analytic_jacobian(self, _t, y):
        r = self._rates(y)
        n = self.n
        m = float(y[0])
        jac = np.zeros((2 * n + 3, 2 * n + 3))
        jac[0, 0] = -2 * self.resistance_total * abs(m) / self.inertia_total
        sign = 1.0 if m >= 0 else -1.0
        for i in range(n):
            jac[1 + i, 0] = (
                self.cp * sign * r["delta_t"][i] + 3 * self.resistance[i] * m * abs(m) / self.rho
            ) / self.capacity[i]
            jac[1 + i, 1 + i] -= (self.cp * abs(m) + self.ua[i]) / self.capacity[i]
            jac[1 + i, 1 + r["donor"][i]] += self.cp * abs(m) / self.capacity[i]
            jac[1 + i, 1 + n + i] = self.ua[i] / self.capacity[i]
            jac[1 + n + i, 1 + i] = self.ua[i] / self.wall_capacity[i]
            jac[1 + n + i, 1 + n + i] = -(self.ua[i] + self.ambient_ua[i]) / self.wall_capacity[i]
            jac[-1, 1 + n + i] = -self.ambient_ua[i]
        jac[-2, 0] = float(sum(self._heads)) / self.rho
        return csc_matrix(jac)

    def observation(self, y=None, time_s=None, heads=None):
        y = self._state if y is None else np.asarray(y, dtype=float)
        time_s = self.time_s if time_s is None else time_s
        heads = self._heads if heads is None else heads
        r = self._rates(y, heads)
        m = float(y[0])
        n = self.n
        kinetic = self.inertia_total * m * m / (2 * self.rho)
        stored_change = float(
            self.capacity @ y[1 : n + 1]
            + self.wall_capacity @ y[n + 1 : 2 * n + 1]
            + kinetic
            - self._initial_kinetic
        )
        defect = stored_change - y[-2] - y[-1]
        if not np.isfinite(defect) or abs(defect) > self.energy_tolerance_j:
            raise NetworkError("Circulation total-energy ledger exceeded")
        # One shared total-enthalpy flux per physical face, then opposite signs.
        face_donor = np.roll(np.arange(n), 1) if m >= 0 else np.arange(n)
        velocity = m / (self.rho * self.area)
        with np.errstate(over="ignore", invalid="ignore"):
            flux = m * (
                self.cp * r["fluid_temperature"][face_donor]
                + r["pressure"] / self.rho
                + 0.5 * velocity**2
            )
        if not np.isfinite(flux).all():
            raise NetworkError("Nonfinite total-enthalpy face flux")
        ports = {}
        segment_residual = {}
        for i, name in enumerate(self.names):
            j = (i + 1) % n
            ports[name + ".left"] = dict(
                pressure_pa=float(r["pressure"][i]), mass_flow_kg_s=m, energy_flow_w=float(flux[i])
            )
            ports[name + ".right"] = dict(
                pressure_pa=float(r["pressure"][j]),
                mass_flow_kg_s=-m,
                energy_flow_w=float(-flux[j]),
            )
            edot = self.capacity[i] * r["rhs"][1 + i] + self.inertia[i] * m * r["rhs"][0] / self.rho
            segment_residual[name] = float(
                edot - (flux[i] - flux[j] + heads[i] * m / self.rho + r["wall_heat"][i])
            )
        if (
            not np.isfinite(list(segment_residual.values())).all()
            or max(abs(v) for v in segment_residual.values()) > 1e-6
        ):
            raise NetworkError("Local total-enthalpy/kinetic energy closure exceeded")
        cycle = float(sum(self.inertia * r["rhs"][0] + self.resistance * m * abs(m) - heads))
        if not np.isfinite(cycle) or abs(cycle) > 1e-6:
            raise NetworkError("Pressure cycle closure exceeded")
        return dict(
            time_s=float(time_s),
            mass_flow_kg_s=m,
            flow_acceleration_kg_s2=float(r["rhs"][0]),
            fluid_temperature_k=dict(zip(self.names, map(float, r["fluid_temperature"]))),
            wall_temperature_k=dict(zip(self.names, map(float, r["wall_temperature"]))),
            ports=ports,
            stored_mass_kg=float(sum(self.rho * self.area * self.length)),
            stored_energy_j=self.initial_energy_j + stored_change,
            stored_energy_change_j=stored_change,
            kinetic_energy_j=kinetic,
            external_head_work_j=float(y[-2]),
            external_heat_j=float(y[-1]),
            energy_residual_j=float(defect),
            local_energy_residual_w=segment_residual,
            dissipation_w=float(sum(r["dissipation"])),
            pressure_cycle_residual_pa=cycle,
        )

    def enable_continuation(self, horizon_s):
        proposed = number(horizon_s, "horizon_s", positive=True)
        if proposed <= self.time_s:
            raise NetworkError("Future horizon required")
        self._horizon = proposed

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.05, atol_scale=1.0):
        dt_s = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        max_step_s = number(max_step_s, "max_step_s", positive=True)
        atol_scale = number(atol_scale, "atol_scale", positive=True)
        if (
            method not in ("Radau", "BDF")
            or not 1e-6 <= atol_scale <= 1
            or not 1e-12 <= rtol <= 1e-2
        ):
            raise NetworkError("Unsupported circulation solver policy")
        target = self.time_s + dt_s
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time cannot advance")
        saved = self._state.copy(), self.time_s, self.last_events, list(self.events)
        policy = (method, rtol, max_step_s, atol_scale)
        n = self.n
        try:
            if (
                self._continuation is None
                or self._policy != policy
                or target > self._continuation.horizon
            ):
                atol = np.r_[1e-11, np.full(2 * n, 1e-12), 1e-8, 1e-8] * atol_scale
                self._continuation = ImplicitContinuation(
                    self.derivatives,
                    self._state,
                    self.time_s,
                    max(target, self._horizon or target),
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_s,
                    initial_target=target,
                    jacobian=self.analytic_jacobian,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            observed = self.observation(result.final, target)
            events = []
            for i in range(len(result.t) - 1):
                a, b = result.y[0, i : i + 2]
                direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                if direction:
                    root = (
                        float(result.t[i + 1])
                        if b == 0
                        else brentq(
                            lambda t: float(result.dense(t)[0]),
                            float(result.t[i]),
                            float(result.t[i + 1]),
                            xtol=1e-12,
                        )
                    )
                    events.append(dict(kind="flow_zero", t=root, direction=direction))
            self._state, self.time_s = result.final.copy(), target
            self.last_events = tuple(events)
            self.events.extend(events)
        except BaseException:
            self._state, self.time_s, self.last_events, self.events = saved
            self._continuation = None
            raise
        return observed

    def set_heads(self, heads_pa):
        if not isinstance(heads_pa, dict) or set(heads_pa) != set(self.names):
            raise NetworkError("Head setting required for every segment")
        proposed = np.array([number(heads_pa[n], n) for n in self.names])
        observed = self.observation(heads=proposed)
        event = dict(
            kind="head_changed",
            t=self.time_s,
            previous_pa=self._heads.tolist(),
            head_pa=proposed.tolist(),
        )
        self._heads = proposed
        self._continuation = None
        self._policy = None
        self.last_events = (event,)
        self.events.append(event)
        return observed

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            topology=deepcopy(self.topology),
            fluid=dict(self.fluid),
            reference=dict(self.reference),
            ambient_temperature_k=self.ambient_temperature_k,
            initial_mass_flow_kg_s=self.initial_flow,
            initial_temperature_k=dict(zip(self.names, map(float, self._temperature_reference))),
            initial_wall_temperature_k=dict(zip(self.names, map(float, self._wall_reference))),
            energy_tolerance_j=self.energy_tolerance_j,
            active_heads_pa=dict(zip(self.names, map(float, self._heads))),
            state=self._state.tolist(),
            time_s=self.time_s,
            events=deepcopy(self.events),
            configuration_sha256=self.configuration_sha256(),
            topology_sha256=hashlib.sha256(
                json.dumps(self.topology, sort_keys=True).encode()
            ).hexdigest(),
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "topology",
            "fluid",
            "reference",
            "ambient_temperature_k",
            "initial_mass_flow_kg_s",
            "initial_temperature_k",
            "initial_wall_temperature_k",
            "energy_tolerance_j",
            "active_heads_pa",
            "state",
            "time_s",
            "events",
            "topology_sha256",
            "configuration_sha256",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
        ):
            raise NetworkError("Incompatible circulation checkpoint")
        model = cls(
            **{
                k: payload[k]
                for k in (
                    "topology",
                    "fluid",
                    "reference",
                    "ambient_temperature_k",
                    "initial_mass_flow_kg_s",
                    "initial_temperature_k",
                    "initial_wall_temperature_k",
                    "energy_tolerance_j",
                )
            }
        )
        if model.configuration_sha256() != payload["configuration_sha256"]:
            raise NetworkError("Circulation configuration hash mismatch")
        if model.snapshot()["topology_sha256"] != payload["topology_sha256"]:
            raise NetworkError("Circulation topology hash mismatch")
        if not isinstance(payload["active_heads_pa"], dict) or set(
            payload["active_heads_pa"]
        ) != set(model.names):
            raise NetworkError("Invalid checkpoint head vector")
        heads = np.array([number(payload["active_heads_pa"][n], n) for n in model.names])
        time_s = number(payload["time_s"], "time_s")
        if time_s < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid checkpoint time/events")
        previous_time = 0.0
        recorded_heads = model._initial_heads.copy()
        for e in payload["events"]:
            if not isinstance(e, dict) or e.get("kind") not in ("flow_zero", "head_changed"):
                raise NetworkError("Invalid event history")
            stamp = number(e.get("t"), "event time")
            if not previous_time <= stamp <= time_s:
                raise NetworkError("Invalid event chronology")
            previous_time = stamp
            if e["kind"] == "flow_zero":
                if (
                    set(e) != {"kind", "t", "direction"}
                    or type(e["direction"]) is not int
                    or e["direction"] not in (-1, 1)
                ):
                    raise NetworkError("Invalid zero-flow event")
            else:
                if (
                    set(e) != {"kind", "t", "previous_pa", "head_pa"}
                    or not isinstance(e["previous_pa"], list)
                    or not isinstance(e["head_pa"], list)
                    or len(e["previous_pa"]) != model.n
                    or len(e["head_pa"]) != model.n
                ):
                    raise NetworkError("Invalid head-change event")
                before = np.array([number(v, "previous head") for v in e["previous_pa"]])
                after = np.array([number(v, "new head") for v in e["head_pa"]])
                if not np.array_equal(before, recorded_heads):
                    raise NetworkError("Broken head history")
                recorded_heads = after
        if not np.array_equal(heads, recorded_heads):
            raise NetworkError("Active heads disagree with event history")
        if not isinstance(payload["state"], list):
            raise NetworkError("Checkpoint state list required")
        state = np.array([number(v, "state") for v in payload["state"]])
        model.observation(state, time_s, heads)
        model._state, model.time_s, model._heads, model.events = (
            state,
            time_s,
            heads,
            deepcopy(payload["events"]),
        )
        return model


def steady_circulation(
    topology, fluid, reference, ambient_temperature_k, *, energy_tolerance_j=1e-5
):
    """Direct nonlinear hydraulic root plus dimensional thermal equilibrium solve.

    A thermal sink is required; an unanchored temperature family is rejected.
    No artificial warmup or stored trajectory is used.
    """
    if (
        not isinstance(fluid, dict)
        or set(fluid) != {"density_kg_m3", "heat_capacity_j_kg_k", "area_m2"}
        or not isinstance(reference, dict)
        or set(reference) != {"segment", "pressure_pa"}
    ):
        raise NetworkError("Explicit fluid and pressure reference required")
    ordered = topology_order(topology, reference["segment"])
    rho = number(fluid["density_kg_m3"], "density", positive=True)
    area = number(fluid["area_m2"], "area", positive=True)
    params = [n["parameters"] for n in ordered["nodes"]]
    head = sum(v["drive_pressure_pa"] for v in params)
    resistance = sum(v["loss_coefficient"] for v in params) / (2 * rho * area**2)
    if resistance <= 0:
        raise NetworkError("Steady flow requires positive total resistance")
    flow = float(np.sign(head) * np.sqrt(abs(head) / resistance))
    temperatures = {n["id"]: ambient_temperature_k for n in ordered["nodes"]}
    probe = CirculationLoop(
        ordered,
        fluid,
        reference,
        ambient_temperature_k,
        flow,
        temperatures,
        temperatures,
        energy_tolerance_j=energy_tolerance_j,
    )
    if sum(probe.ambient_ua) <= 0:
        raise NetworkError("Thermal steady initialization requires an ambient anchor")
    indices = np.arange(1, 2 * probe.n + 1)
    weights = np.r_[probe.capacity, probe.wall_capacity]
    jac = (
        weights[:, None]
        * probe.analytic_jacobian(0, probe._state).toarray()[np.ix_(indices, indices)]
    )
    rhs = -weights * probe.derivatives(0, probe._state)[indices]
    row_scale = np.maximum(np.max(abs(jac), axis=1), 1.0)
    scaled = jac / row_scale[:, None]
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-12:
        raise NetworkError("Thermal equilibrium is singular or unanchored")
    delta = np.linalg.solve(scaled, rhs / row_scale)
    tf = {name: float(ambient_temperature_k + delta[i]) for i, name in enumerate(probe.names)}
    tw = {
        name: float(ambient_temperature_k + delta[probe.n + i])
        for i, name in enumerate(probe.names)
    }
    model = CirculationLoop(
        ordered,
        fluid,
        reference,
        ambient_temperature_k,
        flow,
        tf,
        tw,
        energy_tolerance_j=energy_tolerance_j,
    )
    residual = model.derivatives(0, model._state)
    if (
        abs(probe.inertia_total * residual[0]) > 1e-7
        or max(abs(weights * residual[indices])) > 1e-6
    ):
        raise NetworkError("Dimensional steady residual acceptance failed")
    return model
