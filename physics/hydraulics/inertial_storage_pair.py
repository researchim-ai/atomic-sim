"""Gas-cushion stores coupled through an ideal lumped hydraulic inertance.

I*mdot=p_left-p_right-R*m*abs(m), K=I*m**2/(2*rho). Thermal transport
through the link is instantaneous with zero link thermal inventory; this is
an ideal inertance, not a resolved pipe, acoustic model or check-valve impact.
"""

import hashlib
import json
from copy import deepcopy

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csc_matrix

from core.continuation import ImplicitContinuation
from core.network import NetworkError, PortSpec, number
from core.status import ModelPassport, ModelStatus
from physics.hydraulics.gas_storage_pair import CONFIG_FIELDS as STORAGE_FIELDS
from physics.hydraulics.gas_storage_pair import GasStoragePair

PASSPORT = ModelPassport(
    model_id="inertial-gas-storage-pair",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Gas storage boundary work coupled to kinetic hydraulic energy and reversible flow",
    approximations=(
        "two constant-density liquid stores with sealed calorically perfect gas cushions",
        "ideal constant hydraulic inertance with kinetic inventory, quadratic passive loss",
        "instantaneous donor thermal transport; zero link thermal capacity and swept-volume work",
        "friction heats downstream liquid; pressure work charges kinetic inventory",
        "bidirectional finite resistance; no ideal shutoff, contact, acoustics or phase change",
    ),
    applicability="Positive gas/liquid inventories; finite nonnegative resistance and positive inertance",
    data_sources=("I*mdot=dp-R*m*abs(m); K=I*m^2/(2*rho)",),
)
CONFIG_FIELDS = {"storage_configuration", "line", "initial_mass_flow_kg_s"}
PORTS = tuple(
    PortSpec(
        p,
        "hydraulic",
        medium="gas-cushion-liquid-ideal-inertance",
        energy_convention="total_enthalpy",
    )
    for p in ("left", "right")
)


class InertialStoragePair:
    def __init__(self, storage_configuration, line, initial_mass_flow_kg_s):
        if (
            not isinstance(storage_configuration, dict)
            or set(storage_configuration) != STORAGE_FIELDS
        ):
            raise NetworkError("Exact storage configuration required")
        self._storage = GasStoragePair(**deepcopy(storage_configuration))
        if self._storage.configuration["connection"] != dict(
            kind="bidirectional", conductance_m3_s_pa=0.0, cracking_pressure_pa=0.0
        ):
            raise NetworkError("Algebraic transfer must be disabled before adding inertance")
        if not isinstance(line, dict) or set(line) != {"inertance_m_inv", "resistance_pa_s2_kg2"}:
            raise NetworkError("Exact ideal inertance and resistance required")
        inertia = number(line["inertance_m_inv"], "inertance_m_inv", positive=True)
        resistance = number(line["resistance_pa_s2_kg2"], "resistance_pa_s2_kg2")
        if resistance < 0:
            raise NetworkError("Negative line resistance is unsupported")
        flow = number(initial_mass_flow_kg_s, "initial_mass_flow_kg_s")
        self._config = dict(
            storage_configuration=self._storage.configuration,
            line=dict(inertance_m_inv=inertia, resistance_pa_s2_kg2=resistance),
            initial_mass_flow_kg_s=flow,
        )
        vessels = self._storage.configuration["vessels"]
        self._tl0 = np.array([v["initial_liquid_temperature_k"] for v in vessels])
        self._tg0 = np.array([v["initial_gas_temperature_k"] for v in vessels])
        self._p0 = np.array([v["initial_gas_pressure_pa"] for v in vessels])
        for values in (self._tl0, self._tg0, self._p0):
            values.setflags(write=False)
        self.names = self._storage.names
        self.rho = self._storage.rho
        self._inertia = inertia
        self._resistance = resistance
        self._initial_kinetic = inertia * flow**2 / (2 * self.rho)
        self._state = np.r_[np.zeros(8), flow]
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

    def _rates(self, y, resistance=None):
        y = np.asarray(y, dtype=float)
        if y.shape != (9,) or not np.isfinite(y).all():
            raise NetworkError("Invalid inertial storage state")
        base = self._storage._rates(y[:8], conductance=0.0)
        # Preserve the exact initial thermal anchor instead of a U0/C roundtrip.
        signs = np.array([-1.0, 1.0])
        tl = self._tl0 + (y[1:3] - self._storage.cv * self._tl0 * signs * y[0]) / (
            base["mass"] * self._storage.cv
        )
        tg = self._tg0 + y[3:5] / self._storage._gas_capacity
        pressure = (
            self._p0
            * (1 + y[3:5] / self._storage._ug0)
            * (self._storage._gas_volume0 / base["gas_volume"])
        )
        if not np.isfinite(np.r_[tl, tg, pressure]).all() or min(np.r_[tl, tg, pressure]) <= 0:
            raise NetworkError("Invalid anchored storage thermodynamic state")
        base.update(
            liquid_temperature=tl,
            gas_temperature=tg,
            pressure=pressure,
            gas_to_liquid_heat=self._storage._ua * (tg - tl),
            ambient_heat=self._storage._ambient_ua
            * (self._storage._config["ambient_temperature_k"] - tg),
        )
        m = float(y[8])
        r = self._resistance if resistance is None else resistance
        pressure = base["pressure"]
        dm = (pressure[0] - pressure[1] - r * m * abs(m)) / self._inertia
        donor = 0 if m >= 0 else 1
        downstream = 1 - donor
        signs = np.array([-1.0, 1.0])
        thermal_flux = m * self._storage.cv * base["liquid_temperature"][donor]
        dissipation = r * abs(m) ** 3 / self.rho
        heat = np.zeros(2)
        heat[downstream] = dissipation
        boundary = signs * pressure * m / self.rho
        rhs = np.r_[
            m,
            signs * thermal_flux + heat + base["gas_to_liquid_heat"] + self._storage._heater,
            boundary - base["gas_to_liquid_heat"] + base["ambient_heat"],
            sum(self._storage._heater + base["ambient_heat"]),
            boundary,
            dm,
        ]
        # Energy entering the link minus energy leaving it equals dK/dt.
        flux = m * (self._storage.cv * base["liquid_temperature"][donor] + pressure / self.rho)
        flux[1 if m >= 0 else 0] += dissipation if m >= 0 else -dissipation
        if not np.isfinite(rhs).all() or not np.isfinite(flux).all():
            raise NetworkError("Nonfinite inertial storage rates/flux")
        base.update(
            rhs=rhs,
            flow=m,
            donor=donor,
            downstream=downstream,
            dissipation=dissipation,
            boundary_work=boundary,
            face_flux=flux,
            resistance=r,
        )
        return base

    def derivatives(self, _t, y):
        return self._rates(y)["rhs"]

    def analytic_jacobian(self, _t, y):
        r = self._rates(y)
        storage = self._storage
        m = r["flow"]
        signs = np.array([-1.0, 1.0])
        tl_grad = np.zeros((2, 9))
        tg_grad = np.zeros((2, 9))
        p_grad = np.zeros((2, 9))
        flow_grad = np.zeros(9)
        flow_grad[8] = 1
        for i in range(2):
            tl_grad[i, 0] = -signs[i] * r["liquid_temperature"][i] / r["mass"][i]
            tl_grad[i, 1 + i] = 1 / (r["mass"][i] * storage.cv)
            tg_grad[i, 3 + i] = 1 / storage._gas_capacity[i]
            p_grad[i, 0] = r["pressure"][i] * signs[i] / (self.rho * r["gas_volume"][i])
            p_grad[i, 3 + i] = (storage._gamma[i] - 1) / r["gas_volume"][i]
        heat_grad = storage._ua[:, None] * (tg_grad - tl_grad)
        ambient_grad = -storage._ambient_ua[:, None] * tg_grad
        boundary_grad = (
            signs[:, None] * (m * p_grad + r["pressure"][:, None] * flow_grad) / self.rho
        )
        donor = r["donor"]
        thermal_grad = storage.cv * (
            r["liquid_temperature"][donor] * flow_grad + m * tl_grad[donor]
        )
        jac = np.zeros((9, 9))
        jac[0] = flow_grad
        jac[1:3] = signs[:, None] * thermal_grad + heat_grad
        jac[1 + r["downstream"], 8] += 3 * r["resistance"] * m * abs(m) / self.rho
        jac[3:5] = boundary_grad - heat_grad + ambient_grad
        jac[5] = sum(ambient_grad)
        jac[6:8] = boundary_grad
        jac[8] = (p_grad[0] - p_grad[1] - 2 * r["resistance"] * abs(m) * flow_grad) / self._inertia
        if not np.isfinite(jac).all():
            raise NetworkError("Nonfinite inertial storage Jacobian")
        return csc_matrix(jac)

    def observation(self, y=None, time_s=None, resistance=None):
        y = self._state if y is None else np.asarray(y, dtype=float)
        r = self._rates(y, resistance)
        m = r["flow"]
        kinetic = self._inertia * m * m / (2 * self.rho)
        change = (
            self._inertia
            * (m - self._config["initial_mass_flow_kg_s"])
            * (m + self._config["initial_mass_flow_kg_s"])
            / (2 * self.rho)
        )
        defect = float(sum(y[1:5]) + change - y[5])
        kinetic_rate = self._inertia * m * r["rhs"][8] / self.rho
        link_defect = float(r["face_flux"][0] - r["face_flux"][1] - kinetic_rate)
        local = []
        ports = {}
        for i, name in enumerate(self.names):
            sign = -1 if i == 0 else 1
            flux = float(r["face_flux"][i])
            p = float(r["pressure"][i])
            ports[name + ".transfer"] = dict(
                pressure_pa=p, mass_flow_kg_s=sign * m, energy_flow_w=sign * flux
            )
            ports["connection." + ("left" if i == 0 else "right")] = dict(
                pressure_pa=p, mass_flow_kg_s=-sign * m, energy_flow_w=-sign * flux
            )
            local.append(
                float(
                    r["rhs"][1 + i]
                    + r["rhs"][3 + i]
                    - sign * flux
                    - self._storage._heater[i]
                    - r["ambient_heat"][i]
                )
            )
        if (
            not np.isfinite([defect, link_defect, *local]).all()
            or abs(defect) > self._storage._config["energy_tolerance_j"]
            or max(abs(link_defect), max(abs(x) for x in local)) > 1e-6
        ):
            raise NetworkError("Inertial storage total/local energy closure exceeded")
        return dict(
            time_s=float(self.time_s if time_s is None else time_s),
            transfer_mass_kg=float(y[0]),
            mass_flow_kg_s=m,
            flow_acceleration_kg_s2=float(r["rhs"][8]),
            kinetic_energy_j=kinetic,
            kinetic_energy_change_j=change,
            liquid_mass_kg=dict(zip(self.names, map(float, r["mass"]))),
            gas_volume_m3=dict(zip(self.names, map(float, r["gas_volume"]))),
            pressure_pa=dict(zip(self.names, map(float, r["pressure"]))),
            liquid_temperature_k=dict(zip(self.names, map(float, r["liquid_temperature"]))),
            gas_temperature_k=dict(zip(self.names, map(float, r["gas_temperature"]))),
            gas_boundary_power_w=dict(zip(self.names, map(float, r["boundary_work"]))),
            gas_boundary_work_j=dict(zip(self.names, map(float, y[6:8]))),
            liquid_energy_change_j=dict(zip(self.names, map(float, y[1:3]))),
            gas_energy_change_j=dict(zip(self.names, map(float, y[3:5]))),
            external_heat_j=float(y[5]),
            energy_residual_j=defect,
            local_energy_residual_w=dict(zip(self.names, local)),
            link_energy_residual_w=link_defect,
            dissipation_w=r["dissipation"],
            resistance_pa_s2_kg2=r["resistance"],
            ports=ports,
        )

    def enable_continuation(self, horizon_s):
        horizon = number(horizon_s, "horizon_s", positive=True)
        if horizon <= self.time_s:
            raise NetworkError("Future inertial storage horizon required")
        self._horizon = horizon

    def advance(self, dt_s, *, method="Radau", rtol=1e-9, max_step_s=0.05, atol_scale=1.0):
        dt = number(dt_s, "dt_s", positive=True)
        rtol = number(rtol, "rtol", positive=True)
        step = number(max_step_s, "max_step_s", positive=True)
        scale = number(atol_scale, "atol_scale", positive=True)
        if method not in ("Radau", "BDF") or not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= scale <= 1:
            raise NetworkError("Unsupported inertial storage solver")
        target = self.time_s + dt
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time cannot advance")
        saved = self._state.copy(), self.time_s, deepcopy(self.events), self.last_events
        policy = method, rtol, step, scale
        try:
            if (
                self._continuation is None
                or self._policy != policy
                or target > self._continuation.horizon
            ):
                self._continuation = ImplicitContinuation(
                    self.derivatives,
                    self._state,
                    self.time_s,
                    max(target, self._horizon or target),
                    rtol=rtol,
                    atol=np.r_[1e-12, np.full(7, 1e-8), 1e-12] * scale,
                    max_step=step,
                    method=method,
                    initial_target=target,
                    jacobian=self.analytic_jacobian,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            observed = self.observation(result.final, target)
            events = []
            for i in range(len(result.t) - 1):
                a, b = result.y[8, i : i + 2]
                direction = 1 if a < 0 <= b else -1 if a > 0 >= b else 0
                if direction:
                    t = (
                        float(result.t[i + 1])
                        if b == 0
                        else brentq(
                            lambda t: float(result.dense(t)[8]),
                            float(result.t[i]),
                            float(result.t[i + 1]),
                            xtol=1e-12,
                        )
                    )
                    events.append(dict(kind="flow_zero", t=t, direction=direction))
            self._state, self.time_s = result.final.copy(), target
            self.events.extend(events)
            self.last_events = tuple(events)
        except BaseException:
            self._state, self.time_s, self.events, self.last_events = saved
            self._continuation = None
            raise
        return observed

    def set_resistance(self, resistance):
        proposed = number(resistance, "resistance")
        if proposed < 0:
            raise NetworkError("Nonnegative finite resistance required; no ideal shutoff")
        observed = self.observation(resistance=proposed)
        event = dict(
            kind="resistance_changed", t=self.time_s, previous=self._resistance, value=proposed
        )
        self._resistance = proposed
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
            active_resistance_pa_s2_kg2=self._resistance,
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
            "active_resistance_pa_s2_kg2",
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
            raise NetworkError("Incompatible inertial storage checkpoint")
        if (
            not isinstance(payload["configuration"], dict)
            or set(payload["configuration"]) != CONFIG_FIELDS
        ):
            raise NetworkError("Invalid inertial storage configuration")
        model = cls(**payload["configuration"])
        if payload["configuration_sha256"] != model.configuration_sha256():
            raise NetworkError("Inertial storage configuration hash mismatch")
        t = number(payload["time_s"], "checkpoint time")
        r = number(payload["active_resistance_pa_s2_kg2"], "active resistance")
        if t < 0 or r < 0 or not isinstance(payload["events"], list):
            raise NetworkError("Invalid inertial storage time/resistance/events")
        previous_t = 0.0
        expected = model._resistance
        for event in payload["events"]:
            if not isinstance(event, dict):
                raise NetworkError("Invalid inertial storage event")
            stamp = number(event.get("t"), "event time")
            if not previous_t <= stamp <= t:
                raise NetworkError("Unordered inertial storage event")
            previous_t = stamp
            if event.get("kind") == "resistance_changed" and set(event) == {
                "kind",
                "t",
                "previous",
                "value",
            }:
                if number(event["previous"], "previous resistance") != expected:
                    raise NetworkError("Resistance command chain mismatch")
                expected = number(event["value"], "resistance")
                if expected < 0:
                    raise NetworkError("Negative checkpoint resistance")
            elif event.get("kind") == "flow_zero" and set(event) == {"kind", "t", "direction"}:
                if type(event["direction"]) is not int or event["direction"] not in (-1, 1):
                    raise NetworkError("Invalid flow crossing direction")
            else:
                raise NetworkError("Unknown inertial storage event")
        if expected != r:
            raise NetworkError("Active resistance history mismatch")
        if not isinstance(payload["state"], list) or any(
            type(v) not in (int, float) for v in payload["state"]
        ):
            raise NetworkError("Numeric inertial storage state required")
        state = np.array(payload["state"], dtype=float)
        model.observation(state, t, r)
        if t == 0 and not np.array_equal(state, model._state):
            raise NetworkError("Modified initial inertial storage state")
        model._state, model.time_s, model._resistance, model.events = (
            state,
            t,
            r,
            deepcopy(payload["events"]),
        )
        return model
