"""Two constant-density liquid stores with sealed ideal-gas cushions.

A zero-inertia adiabatic transfer line carries a shared donor enthalpy flux.
Gas/liquid boundary work is equal and opposite. There is no pipe kinetic
inventory, acoustics, boiling, membrane inertia or actuator swept volume.
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
    model_id="gas-cushion-transfer-pair",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Liquid storage with dynamic gas energy, boundary work and resistive transfer",
    approximations=(
        "two rigid vessels, incompressible constant-density liquid; constant cv",
        "sealed calorically perfect ideal gases; massless pressure-equilibrated interfaces",
        "zero-inertia adiabatic line, shared donor enthalpy; downstream hydraulic loss heats liquid",
        "optional memoryless threshold check valve; no contact/actuator energy or hysteresis",
        "no pipe kinetic storage, acoustics, phase change, gravity or plant qualification",
    ),
    applicability="Both liquid and gas inventories remain positive; 1 < gamma <= 2",
    data_sources=("dU_liquid=sum(m*h)-p*dV_liquid+Q; dU_gas=-p*dV_gas-Q",),
)
PORTS = tuple(
    PortSpec(
        name,
        "hydraulic",
        medium="gas-cushion-constant-density-liquid",
        energy_convention="specific_enthalpy_transport",
    )
    for name in ("left", "right")
)
EVENT_RESOLUTION_PA = 1e-5
VESSEL_FIELDS = {
    "id",
    "total_volume_m3",
    "initial_liquid_mass_kg",
    "initial_liquid_temperature_k",
    "initial_gas_pressure_pa",
    "initial_gas_temperature_k",
    "gas_gamma",
    "gas_liquid_ua_w_k",
    "gas_ambient_ua_w_k",
    "liquid_heater_w",
}
CONFIG_FIELDS = {"fluid", "vessels", "connection", "ambient_temperature_k", "energy_tolerance_j"}


class GasStoragePair:
    def __init__(self, fluid, vessels, connection, ambient_temperature_k, energy_tolerance_j=1e-5):
        if not isinstance(fluid, dict) or set(fluid) != {"density_kg_m3", "liquid_cv_j_kg_k"}:
            raise NetworkError("Explicit constant-density liquid properties required")
        fluid = {k: number(v, k, positive=True) for k, v in fluid.items()}
        if not isinstance(vessels, list) or len(vessels) != 2:
            raise NetworkError("Exactly two ordered gas-cushion vessels required")
        normalized = []
        for v in vessels:
            if not isinstance(v, dict) or set(v) != VESSEL_FIELDS:
                raise NetworkError("Exact gas-cushion vessel fields required")
            p = {k: number(value, k) for k, value in v.items() if k != "id"}
            p["id"] = identifier(v["id"])
            if (
                any(
                    p[k] <= 0
                    for k in VESSEL_FIELDS
                    - {"id", "gas_liquid_ua_w_k", "gas_ambient_ua_w_k", "liquid_heater_w"}
                )
                or any(
                    p[k] < 0 for k in ("gas_liquid_ua_w_k", "gas_ambient_ua_w_k", "liquid_heater_w")
                )
                or not 1 < p["gas_gamma"] <= 2
            ):
                raise NetworkError(
                    "Invalid vessel inventory, gas gamma, heat conductance or heater"
                )
            normalized.append(p)
        if normalized[0]["id"] == normalized[1]["id"]:
            raise NetworkError("Distinct vessel identifiers required")
        if (
            not isinstance(connection, dict)
            or set(connection) != {"kind", "conductance_m3_s_pa", "cracking_pressure_pa"}
            or connection["kind"] not in ("bidirectional", "check")
        ):
            raise NetworkError("Explicit zero-inertia resistive connection required")
        connection = dict(
            kind=connection["kind"],
            conductance_m3_s_pa=number(connection["conductance_m3_s_pa"], "conductance"),
            cracking_pressure_pa=number(connection["cracking_pressure_pa"], "cracking pressure"),
        )
        if (
            connection["conductance_m3_s_pa"] < 0
            or connection["cracking_pressure_pa"] < 0
            or (connection["kind"] == "bidirectional" and connection["cracking_pressure_pa"] != 0)
        ):
            raise NetworkError("Invalid connection conductance or threshold")
        self._config = dict(
            fluid=fluid,
            vessels=normalized,
            connection=connection,
            ambient_temperature_k=number(
                ambient_temperature_k, "ambient_temperature_k", positive=True
            ),
            energy_tolerance_j=number(energy_tolerance_j, "energy_tolerance_j", positive=True),
        )
        self.names = tuple(v["id"] for v in normalized)
        self.rho, self.cv = fluid["density_kg_m3"], fluid["liquid_cv_j_kg_k"]

        def array(k):
            return np.array([v[k] for v in normalized])

        self._mass0 = array("initial_liquid_mass_kg")
        self._volume = array("total_volume_m3")
        self._gas_volume0 = self._volume - self._mass0 / self.rho
        self._gamma = array("gas_gamma")
        self._ul0 = self._mass0 * self.cv * array("initial_liquid_temperature_k")
        self._ug0 = array("initial_gas_pressure_pa") * self._gas_volume0 / (self._gamma - 1)
        self._gas_capacity = self._ug0 / array("initial_gas_temperature_k")
        self._ua = array("gas_liquid_ua_w_k")
        self._ambient_ua = array("gas_ambient_ua_w_k")
        self._heater = array("liquid_heater_w")
        for a in (
            self._mass0,
            self._volume,
            self._gas_volume0,
            self._gamma,
            self._ul0,
            self._ug0,
            self._gas_capacity,
            self._ua,
            self._ambient_ua,
            self._heater,
        ):
            if not np.isfinite(a).all():
                raise NetworkError("Nonfinite initial storage coefficients")
            a.setflags(write=False)
        if min(self._gas_volume0) <= 0:
            raise NetworkError("Positive initial gas space required")
        self._conductance = connection["conductance_m3_s_pa"]
        self._state = np.zeros(8)
        self.time_s = 0.0
        self.events = []
        self.last_events = ()
        self._continuation = None
        self._policy = None
        self._horizon = None
        self.observation()

    @property
    def configuration(self):
        return deepcopy(self._config)

    def configuration_sha256(self):
        return hashlib.sha256(
            json.dumps(self.configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _rates(self, y, conductance=None):
        y = np.asarray(y, dtype=float)
        if y.shape != (8,) or not np.isfinite(y).all():
            raise NetworkError("Invalid gas storage state")
        signs = np.array([-1.0, 1.0])
        mass = self._mass0 + signs * y[0]
        vg = self._volume - mass / self.rho
        ul = self._ul0 + y[1:3]
        ug = self._ug0 + y[3:5]
        if min(np.r_[mass, vg, ul, ug]) <= 0:
            raise NetworkError("Liquid/gas inventory or internal energy exhausted")
        tl = ul / (mass * self.cv)
        tg = ug / self._gas_capacity
        pressure = (self._gamma - 1) * ug / vg
        if not np.isfinite(np.r_[tl, tg, pressure]).all():
            raise NetworkError("Nonfinite storage temperature or pressure")
        c = self._conductance if conductance is None else conductance
        dp = float(pressure[0] - pressure[1])
        margin = dp - self._config["connection"]["cracking_pressure_pa"]
        check = self._config["connection"]["kind"] == "check"
        q = c * (max(margin, 0.0) if check else dp)
        flow = self.rho * q
        donor = 0 if flow >= 0 else 1
        enthalpy = self.cv * tl[donor] + pressure[donor] / self.rho
        flux = flow * enthalpy
        heat = self._ua * (tg - tl)
        ambient = self._ambient_ua * (self._config["ambient_temperature_k"] - tg)
        boundary = pressure * signs * q  # positive INTO gas: compression work
        rates = np.r_[
            flow,
            signs * flux - boundary + heat + self._heater,
            boundary - heat + ambient,
            sum(self._heater + ambient),
            boundary,
        ]
        if not np.isfinite(rates).all() or not np.isfinite(flux):
            raise NetworkError("Nonfinite storage rates or flux")
        return dict(
            rhs=rates,
            mass=mass,
            gas_volume=vg,
            liquid_energy=ul,
            gas_energy=ug,
            liquid_temperature=tl,
            gas_temperature=tg,
            pressure=pressure,
            flow=flow,
            flux=flux,
            boundary_work=boundary,
            gas_to_liquid_heat=heat,
            ambient_heat=ambient,
            donor=donor,
            margin=margin,
            active=(not check or margin > 0),
            conductance=c,
        )

    def derivatives(self, _t, y):
        return self._rates(y)["rhs"]

    def analytic_jacobian(self, _t, y):
        r = self._rates(y)
        signs = np.array([-1.0, 1.0])
        tl_grad = np.zeros((2, 8))
        tg_grad = np.zeros((2, 8))
        p_grad = np.zeros((2, 8))
        for i in range(2):
            tl_grad[i, 0] = -signs[i] * r["liquid_temperature"][i] / r["mass"][i]
            tl_grad[i, 1 + i] = 1 / (r["mass"][i] * self.cv)
            tg_grad[i, 3 + i] = 1 / self._gas_capacity[i]
            p_grad[i, 0] = r["pressure"][i] * signs[i] / (self.rho * r["gas_volume"][i])
            p_grad[i, 3 + i] = (self._gamma[i] - 1) / r["gas_volume"][i]
        dq = r["conductance"] * (p_grad[0] - p_grad[1]) if r["active"] else np.zeros(8)
        df = self.rho * dq
        d = r["donor"]
        h = self.cv * r["liquid_temperature"][d] + r["pressure"][d] / self.rho
        dflux = h * df + r["flow"] * (self.cv * tl_grad[d] + p_grad[d] / self.rho)
        heat_grad = self._ua[:, None] * (tg_grad - tl_grad)
        ambient_grad = -self._ambient_ua[:, None] * tg_grad
        boundary_grad = (
            signs[:, None] * (r["flow"] * p_grad + r["pressure"][:, None] * df) / self.rho
        )
        jac = np.zeros((8, 8))
        jac[0] = df
        jac[1:3] = signs[:, None] * dflux - boundary_grad + heat_grad
        jac[3:5] = boundary_grad - heat_grad + ambient_grad
        jac[5] = sum(ambient_grad)
        jac[6:8] = boundary_grad
        if not np.isfinite(jac).all():
            raise NetworkError("Nonfinite storage Jacobian")
        return csc_matrix(jac)

    def observation(self, y=None, time_s=None, conductance=None):
        y = self._state if y is None else np.asarray(y, dtype=float)
        r = self._rates(y, conductance)
        defect = float(sum(y[1:5]) - y[5])
        if not np.isfinite(defect) or abs(defect) > self._config["energy_tolerance_j"]:
            raise NetworkError("Storage total-energy ledger exceeded")
        ports = {}
        for i, name in enumerate(self.names):
            sign = -1 if i == 0 else 1
            ports[name + ".transfer"] = dict(
                pressure_pa=float(r["pressure"][i]),
                mass_flow_kg_s=sign * r["flow"],
                energy_flow_w=sign * r["flux"],
            )
            ports["connection." + ("left" if i == 0 else "right")] = dict(
                pressure_pa=float(r["pressure"][i]),
                mass_flow_kg_s=-sign * r["flow"],
                energy_flow_w=-sign * r["flux"],
            )
        local = (
            r["rhs"][1:3]
            + r["rhs"][3:5]
            - np.array([-1.0, 1.0]) * r["flux"]
            - self._heater
            - r["ambient_heat"]
        )
        if max(abs(local)) > 1e-6:
            raise NetworkError("Local storage enthalpy/work closure exceeded")
        result = dict(
            time_s=float(self.time_s if time_s is None else time_s),
            transfer_mass_kg=float(y[0]),
            mass_flow_kg_s=r["flow"],
            liquid_mass_kg=dict(zip(self.names, map(float, r["mass"]))),
            gas_volume_m3=dict(zip(self.names, map(float, r["gas_volume"]))),
            liquid_temperature_k=dict(zip(self.names, map(float, r["liquid_temperature"]))),
            gas_temperature_k=dict(zip(self.names, map(float, r["gas_temperature"]))),
            pressure_pa=dict(zip(self.names, map(float, r["pressure"]))),
            gas_boundary_power_w=dict(zip(self.names, map(float, r["boundary_work"]))),
            gas_boundary_work_j=dict(zip(self.names, map(float, y[6:8]))),
            gas_to_liquid_heat_w=dict(zip(self.names, map(float, r["gas_to_liquid_heat"]))),
            liquid_energy_change_j=dict(zip(self.names, map(float, y[1:3]))),
            gas_energy_change_j=dict(zip(self.names, map(float, y[3:5]))),
            external_heat_j=float(y[5]),
            energy_residual_j=defect,
            local_energy_residual_w=dict(zip(self.names, map(float, local))),
            dissipation_w=float((r["pressure"][0] - r["pressure"][1]) * r["flow"] / self.rho),
            ports=ports,
            conductance_m3_s_pa=r["conductance"],
            check_margin_pa=r["margin"],
            connection_mode="isolated"
            if r["conductance"] == 0
            else "open"
            if r["active"]
            else "blocked",
        )
        if result["dissipation_w"] < -1e-8:
            raise NetworkError("Nonpassive storage connection")
        return result

    def enable_continuation(self, horizon_s):
        horizon = number(horizon_s, "horizon_s", positive=True)
        if horizon <= self.time_s:
            raise NetworkError("Future storage horizon required")
        self._horizon = horizon

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.05, atol_scale=1.0):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported storage solver policy")
        target = self.time_s + dt
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time cannot advance")
        saved = self._state.copy(), self.time_s, deepcopy(self.events), self.last_events
        policy = method, rtol, step, scale
        try:
            if (
                self._continuation is None
                or policy != self._policy
                or target > self._continuation.horizon
            ):
                self._continuation = ImplicitContinuation(
                    self.derivatives,
                    self._state,
                    self.time_s,
                    max(target, self._horizon or target),
                    rtol=rtol,
                    atol=np.r_[1e-12, np.full(7, 1e-8)] * scale,
                    max_step=step,
                    method=method,
                    initial_target=target,
                    jacobian=self.analytic_jacobian,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            o = self.observation(result.final, target)
            events = []
            if self._config["connection"]["kind"] == "check":
                margins = [self._rates(result.y[:, i])["margin"] for i in range(len(result.t))]
                for i, (a, b) in enumerate(zip(margins, margins[1:])):
                    direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                    if direction:
                        t = (
                            float(result.t[i + 1])
                            if b == 0
                            else brentq(
                                lambda t: self._rates(result.dense(t))["margin"],
                                float(result.t[i]),
                                float(result.t[i + 1]),
                                xtol=1e-12,
                            )
                        )
                        if abs(self._rates(result.dense(t))["margin"]) > EVENT_RESOLUTION_PA:
                            raise NetworkError("Check event pressure root residual exceeded")
                        events.append(dict(kind="check_threshold", t=t, direction=direction))
            self._state, self.time_s = result.final.copy(), target
            self.events.extend(events)
            self.last_events = tuple(events)
        except BaseException:
            self._state, self.time_s, self.events, self.last_events = saved
            self._continuation = None
            raise
        return o

    def set_conductance(self, conductance):
        proposed = number(conductance, "conductance")
        if proposed < 0:
            raise NetworkError("Nonnegative storage conductance required")
        observed = self.observation(conductance=proposed)
        event = dict(
            kind="conductance_changed", t=self.time_s, previous=self._conductance, value=proposed
        )
        self._conductance = proposed
        self.events.append(event)
        self.last_events = (event,)
        self._continuation = None
        self._policy = None
        return observed

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=PASSPORT.model_id,
            model_version=PASSPORT.version,
            configuration=self.configuration,
            configuration_sha256=self.configuration_sha256(),
            state=self._state.tolist(),
            time_s=self.time_s,
            active_conductance_m3_s_pa=self._conductance,
            events=deepcopy(self.events),
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
            "active_conductance_m3_s_pa",
            "events",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
        ):
            raise NetworkError("Incompatible gas storage checkpoint")
        if (
            not isinstance(payload["configuration"], dict)
            or set(payload["configuration"]) != CONFIG_FIELDS
        ):
            raise NetworkError("Invalid storage checkpoint configuration")
        model = cls(**payload["configuration"])
        if model.configuration_sha256() != payload["configuration_sha256"]:
            raise NetworkError("Storage configuration hash mismatch")
        t = number(payload["time_s"], "checkpoint time")
        c = number(payload["active_conductance_m3_s_pa"], "active conductance")
        if t < 0 or c < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid storage checkpoint time/conductance/events")
        previous_t = 0.0
        expected = model._conductance
        for e in payload["events"]:
            if not isinstance(e, dict):
                raise NetworkError("Invalid storage event")
            stamp = number(e.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered storage event")
            previous_t = stamp
            if e.get("kind") == "conductance_changed" and set(e) == {
                "kind",
                "t",
                "previous",
                "value",
            }:
                if number(e["previous"], "previous conductance") != expected:
                    raise NetworkError("Storage command chain mismatch")
                expected = number(e["value"], "conductance")
                if expected < 0:
                    raise NetworkError("Negative checkpoint conductance")
            elif e.get("kind") == "check_threshold" and set(e) == {"kind", "t", "direction"}:
                if (
                    model._config["connection"]["kind"] != "check"
                    or type(e["direction"]) is not int
                    or e["direction"] not in (-1, 1)
                ):
                    raise NetworkError("Invalid check event")
            else:
                raise NetworkError("Unknown storage checkpoint event")
        if expected != c:
            raise NetworkError("Active storage conductance chain mismatch")
        if not isinstance(payload["state"], list) or any(
            type(v) not in (int, float) for v in payload["state"]
        ):
            raise NetworkError("Numeric storage state required")
        state = np.array(payload["state"], dtype=float)
        model.observation(state, t, c)
        if t == 0 and not np.array_equal(state, np.zeros(8)):
            raise NetworkError("Modified initial storage state")
        model._state, model.time_s, model._conductance, model.events = (
            state,
            t,
            c,
            deepcopy(payload["events"]),
        )
        return model
