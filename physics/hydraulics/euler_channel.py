"""Conservative one-dimensional IF97 liquid Euler equations with wall drag.

Cell inventories are M, J=M*u and E=U+J**2/(2*M). Fixed-speed Rusanov flux
uses a checked wave-speed upper bound. Periodic or prescribed exterior-state
Riemann boundaries; these are not exact static-pressure port conditions.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np

from core.continuation import ImplicitContinuation
from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.dynamic_channel import _liquid_state, _sound_speed
from physics.hydraulics.isobaric import _finite
from physics.hydraulics.liquid_derivatives import conserved_derivatives
from physics.materials.iapws import properties_pt

PASSPORT = ModelPassport(
    model_id="conservative-euler-liquid-channel",
    version="0.1.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Консервативная 1D масса, импульс с конвекцией, полная энергия и стенка",
    approximations=(
        "постоянное сечение, горизонтальная жидкость IF97 region 1, Mach<=0.1",
        "фиксированная верхняя граница скорости волн в потоке Rusanov, первый порядок",
        "распределённое квадратичное трение; полная диссипация остаётся в жидкости",
        "периодические или Riemann-границы с предписанными внешними состояниями",
        "без вязкой диффузии, гравитации, кипения, CHF, насоса и нейтроники",
        "позитивность implicit-схемы и общая акустическая точность не квалифицированы",
    ),
    applicability="Однофазный компонент; не модель установки или точных pressure/temperature ports",
    data_sources=(
        "IAPWS IF97 region 1",
        "https://mooseframework.inl.gov/releases/moose/2022-06-10/modules/thermal_hydraulics/source/components/FlowChannel1Phase.html",
    ),
)

CONTACT_PASSPORT = replace(
    PASSPORT,
    model_id="contact-euler-liquid-channel",
    phenomenon="Консервативный IF97 Euler с контактно-разрешающим HLLC",
    approximations=(
        PASSPORT.approximations[0],
        "HLLC с фиксированными внешними волнами +/-a; первый порядок",
        *PASSPORT.approximations[2:],
    ),
    data_sources=PASSPORT.data_sources + ("https://doi.org/10.1007/BF01414629",),
)


@dataclass(frozen=True)
class EulerConfig:
    cells: int = 4
    total_length_m: float = 3.0
    area_m2: float = 0.001
    left_pressure_pa: float = 7e6
    right_pressure_pa: float = 7e6
    left_temperature_k: float = 500.0
    right_temperature_k: float = 500.0
    left_velocity_m_s: float = 0.0
    right_velocity_m_s: float = 0.0
    initial_pressure_pa: float = 7e6
    initial_temperature_k: float = 500.0
    initial_velocity_m_s: float = 0.0
    initial_wall_temperature_k: float = 500.0
    total_loss_coefficient: float = 30.0
    total_heat_capacity_j_k: float = 8000.0
    total_ua_w_k: float = 1000.0
    heat_w: float = 0.0
    wave_speed_bound_m_s: float = 2500.0
    boundary: str = "reservoir"
    numerical_flux: str = "rusanov"
    initial_pressure_profile_pa: tuple | None = None
    initial_temperature_profile_k: tuple | None = None
    initial_velocity_profile_m_s: tuple | None = None
    initial_wall_temperature_profile_k: tuple | None = None

    def __post_init__(self):
        maximum_cells = 512 if self.numerical_flux == "hllc" else 64
        if type(self.cells) is not int or not 1 <= self.cells <= maximum_cells:
            raise ValueError(f"cells must be an integer in [1,{maximum_cells}]")
        if self.boundary not in ("periodic", "reservoir"):
            raise ValueError("Unknown Euler boundary")
        if self.numerical_flux not in ("rusanov", "hllc"):
            raise ValueError("Unknown Euler numerical flux")
        for key in self.__dataclass_fields__:
            if key in ("cells", "boundary", "numerical_flux") or "profile" in key:
                continue
            value = _finite(getattr(self, key), key)
            if "velocity" in key:
                continue
            if key in ("heat_w", "total_loss_coefficient", "total_ua_w_k"):
                if value < 0:
                    raise ValueError("Negative " + key)
            elif value <= 0:
                raise ValueError("Nonpositive " + key)
        for key in self.__dataclass_fields__:
            if "profile" not in key:
                continue
            values = getattr(self, key)
            if values is not None:
                if not isinstance(values, (tuple, list)) or len(values) != self.cells:
                    raise ValueError("Invalid profile size: " + key)
                object.__setattr__(
                    self,
                    key,
                    tuple(_finite(v, key, positive="velocity" not in key) for v in values),
                )
        for p, t in (
            (self.left_pressure_pa, self.left_temperature_k),
            (self.right_pressure_pa, self.right_temperature_k),
            (self.initial_pressure_pa, self.initial_temperature_k),
        ):
            if properties_pt(p, t).region != 1:
                raise DomainError("Euler channel requires liquid IF97 region 1")
        for i in range(self.cells):
            p = (
                self.initial_pressure_profile_pa[i]
                if self.initial_pressure_profile_pa is not None
                else self.initial_pressure_pa
            )
            t = (
                self.initial_temperature_profile_k[i]
                if self.initial_temperature_profile_k is not None
                else self.initial_temperature_k
            )
            if properties_pt(p, t).region != 1:
                raise DomainError("Euler initial profile left liquid region")

    def as_dict(self):
        data = asdict(self)
        if self.numerical_flux == "rusanov":
            data.pop("numerical_flux")
        for k in tuple(data):
            if "profile" in k:
                if data[k] is None:
                    data.pop(k)
                else:
                    data[k] = list(data[k])
        return data


class EulerLiquidChannel:
    passport = PASSPORT

    def __init__(self, config: EulerConfig):
        self.config = config
        self.passport = CONTACT_PASSPORT if config.numerical_flux == "hllc" else PASSPORT
        self.time_s = 0.0
        self.dx = config.total_length_m / config.cells
        self.volume = config.area_m2 * self.dx
        self._continuation = None
        self._horizon = None
        self._policy = None
        self._rhs_calls = self._jac_calls = 0
        self.last_events = ()
        fluid = []
        walls = []
        for i in range(config.cells):

            def initial(name):
                profile = getattr(
                    config,
                    "initial_"
                    + name
                    + "_profile_"
                    + ("m_s" if name == "velocity" else "pa" if name == "pressure" else "k"),
                )
                return (
                    profile[i]
                    if profile is not None
                    else getattr(
                        config,
                        "initial_"
                        + name
                        + "_"
                        + ("m_s" if name == "velocity" else "pa" if name == "pressure" else "k"),
                    )
                )

            fluid.extend(
                self.inventory(initial("pressure"), initial("temperature"), initial("velocity"))
            )
            profile = config.initial_wall_temperature_profile_k
            walls.append(profile[i] if profile is not None else config.initial_wall_temperature_k)
        self._state = np.r_[fluid, walls, 0.0, 0.0, 0.0]
        self._ghosts = [
            self.inventory(
                config.left_pressure_pa, config.left_temperature_k, config.left_velocity_m_s
            ),
            self.inventory(
                config.right_pressure_pa, config.right_temperature_k, config.right_velocity_m_s
            ),
        ]
        self.rates(self._state)
        self.initial_mass_kg, self.initial_momentum_kg_m_s, self.initial_energy_j = (
            self.inventories(self._state)
        )

    def inventory(self, p, t, u):
        thermo = properties_pt(float(p), float(t))
        if thermo.region != 1:
            raise DomainError("Euler inventory left liquid region")
        mass = thermo.rho_kg_m3 * self.volume
        return np.array([mass, mass * u, mass * (thermo.u_j_kg + 0.5 * u * u)])

    def cell(self, q):
        mass, momentum, energy = q
        if mass <= 0:
            raise DomainError("Nonpositive Euler mass")
        velocity = momentum / mass
        internal = energy - 0.5 * momentum * velocity
        state = _liquid_state(
            float(mass / self.volume),
            float(internal / mass),
            self.config.initial_pressure_pa,
            self.config.initial_temperature_k,
        )
        sound = _sound_speed(state.p_pa, state.T_k)
        if abs(velocity) / sound > 0.1:
            raise DomainError("Euler Mach > 0.1")
        if abs(velocity) + sound > self.config.wave_speed_bound_m_s:
            raise DomainError("Euler numerical wave-speed bound exceeded")
        return state, velocity, internal, sound

    def flux(self, q, derivative=False):
        state, u, internal, sound = self.cell(q)
        m, j, e = q
        p = state.p_pa
        flux = np.array(
            [
                j / self.dx,
                j * u / self.dx + p * self.config.area_m2,
                (e + p * self.volume) * u / self.dx,
            ]
        )
        if not derivative:
            return flux
        thermo = conserved_derivatives(state, m, internal, self.volume)
        internal_gradient = np.array([0.5 * u * u, -u, 1.0])
        dp = thermo.pressure[1] * internal_gradient + np.array([thermo.pressure[0], 0.0, 0.0])
        dtemp = thermo.temperature[1] * internal_gradient + np.array(
            [thermo.temperature[0], 0.0, 0.0]
        )
        du = np.array([-u / m, 1 / m, 0.0])
        jac = np.array(
            [
                [0.0, 1 / self.dx, 0.0],
                np.array([-u * u / self.dx, 2 * u / self.dx, 0.0]) + self.config.area_m2 * dp,
                u / self.dx * (np.array([0.0, 0.0, 1.0]) + self.volume * dp)
                + (e + p * self.volume) / self.dx * du,
            ]
        )
        return flux, jac, dtemp

    def rates(self, state, with_jacobian=False):
        from scipy.sparse import lil_matrix

        c = self.config
        n = c.cells
        y = np.asarray(state, dtype=float)
        if y.shape != (4 * n + 3,) or not np.isfinite(y).all():
            raise DomainError("Invalid Euler state vector")
        q = y[: 3 * n].reshape(n, 3)
        tw = y[3 * n : 4 * n]
        if np.any(tw <= 0):
            raise DomainError("Nonpositive Euler wall temperature")
        cells = [self.cell(row) for row in q]
        fluxes = [self.flux(row, with_jacobian) for row in q]
        a = c.wave_speed_bound_m_s / self.dx
        faces = []
        face_gradients = []
        for j in range(n + 1):
            li = j - 1
            ri = j
            if c.boundary == "periodic":
                li %= n
                ri %= n
            left = q[li] if li >= 0 else self._ghosts[0]
            right = q[ri] if ri < n else self._ghosts[1]
            lf = (
                fluxes[li][0]
                if with_jacobian and li >= 0
                else fluxes[li]
                if li >= 0
                else self.flux(left)
            )
            rf = (
                fluxes[ri][0]
                if with_jacobian and ri < n
                else fluxes[ri]
                if ri < n
                else self.flux(right)
            )
            contact = None
            if c.numerical_flux == "hllc":
                from physics.hydraulics.hllc import interface

                contact = interface(self, left, right, with_jacobian)
                faces.append(contact["flux"])
            else:
                faces.append(0.5 * (lf + rf) - 0.5 * a * (right - left))
            if with_jacobian:
                gradients = {}
                for index, sign in ((li, 1), (ri, -1)):
                    if 0 <= index < n:
                        block = (
                            contact["left_gradient" if sign == 1 else "right_gradient"]
                            if contact is not None
                            else 0.5 * fluxes[index][1] + 0.5 * sign * a * np.eye(3)
                        )
                        gradients[index] = gradients.get(index, np.zeros((3, 3))) + block
                face_gradients.append(gradients)
        faces = np.array(faces)
        m = q[:, 0]
        j = q[:, 1]
        u = j / m
        loss = c.total_loss_coefficient / n
        drag = -loss * j * abs(j) / (2 * m * self.dx)
        temperatures = np.array([cell[0].T_k for cell in cells])
        heat = c.total_ua_w_k / n * (tw - temperatures)
        fluid = faces[:-1] - faces[1:]
        fluid[:, 1] += drag
        fluid[:, 2] += heat
        capacity = c.total_heat_capacity_j_k / n
        walls = (c.heat_w / n - heat) / capacity
        boundary = faces[0] - faces[-1]
        ledger = boundary + np.array([0.0, sum(drag), c.heat_w])
        rhs = np.r_[fluid.ravel(), walls, ledger]
        if not np.isfinite(rhs).all():
            raise DomainError("Nonfinite Euler derivative")
        result = dict(
            derivative=rhs,
            faces=faces,
            temperatures=temperatures,
            velocities=u,
            pressures=np.array([cell[0].p_pa for cell in cells]),
            kinetic=0.5 * j * j / m,
            drag=drag,
            wall_heat=heat,
            max_mach=max(abs(cell[1]) / cell[3] for cell in cells),
        )
        if with_jacobian:
            jac = lil_matrix((len(y), len(y)))
            for face, gradients in enumerate(face_gradients):
                for index, block in gradients.items():
                    cols = slice(3 * index, 3 * index + 3)
                    if face > 0:
                        jac[3 * (face - 1) : 3 * face, cols] -= block
                    if face < n:
                        jac[3 * face : 3 * face + 3, cols] += block
                    if face == 0:
                        jac[-3:, cols] += block
                    if face == n:
                        jac[-3:, cols] -= block
            for i in range(n):
                cols = slice(3 * i, 3 * i + 3)
                ds = np.array(
                    [
                        loss * j[i] * abs(j[i]) / (2 * m[i] ** 2 * self.dx),
                        -loss * abs(j[i]) / (m[i] * self.dx),
                        0.0,
                    ]
                )
                jac[3 * i + 1, cols] += ds
                jac[-2, cols] += ds
                dq = -c.total_ua_w_k / n * fluxes[i][2]
                jac[3 * i + 2, cols] += dq
                jac[3 * n + i, cols] -= dq / capacity
                jac[3 * i + 2, 3 * n + i] += c.total_ua_w_k / n
                jac[3 * n + i, 3 * n + i] -= c.total_ua_w_k / c.total_heat_capacity_j_k
            result["jacobian"] = jac.tocsc()
            result["jacobian"].eliminate_zeros()
            if not np.isfinite(result["jacobian"].data).all():
                raise DomainError("Nonfinite Euler analytic Jacobian")
        return result

    def derivatives(self, t, state):
        self._rhs_calls += 1
        return self.rates(state)["derivative"]

    def analytic_jacobian(self, t, state):
        self._jac_calls += 1
        return self.rates(state, with_jacobian=True)["jacobian"]

    def inventories(self, state):
        n = self.config.cells
        q = np.asarray(state[: 3 * n]).reshape(n, 3)
        return (
            float(sum(q[:, 0])),
            float(sum(q[:, 1])),
            float(
                sum(q[:, 2]) + self.config.total_heat_capacity_j_k / n * sum(state[3 * n : 4 * n])
            ),
        )

    def enable_continuation(self, horizon):
        _finite(horizon, "horizon", positive=True)
        if horizon <= self.time_s:
            raise ValueError("Horizon must exceed time")
        self._horizon = horizon
        self._continuation = None

    def advance(
        self,
        dt_s,
        *,
        method="Radau",
        rtol=1e-7,
        max_step_s=0.01,
        jacobian_mode="analytic",
        atol_scale=1.0,
    ):
        for value, name in ((dt_s, "dt"), (rtol, "rtol"), (max_step_s, "max_step")):
            _finite(value, name, positive=True)
        atol_scale = _finite(atol_scale, "atol_scale", positive=True)
        if not 1e-6 <= atol_scale <= 1.0:
            raise ValueError("atol_scale must be in [1e-6,1]")
        if method not in ("BDF", "Radau") or jacobian_mode not in ("analytic", "finite_difference"):
            raise ValueError("Unsupported Euler solver policy")
        target = self.time_s + dt_s
        policy = (method, rtol, max_step_s, jacobian_mode, atol_scale)
        saved = self._state.copy(), self.time_s, self.last_events
        n = self.config.cells
        atol = np.r_[np.tile([1e-10, 1e-8, 1e-4], n), np.full(n, 1e-8), 1e-10, 1e-8, 1e-4]
        atol *= atol_scale
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
                    atol=atol,
                    max_step=max_step_s,
                    method=method,
                    initial_target=target,
                    jacobian=self.analytic_jacobian if jacobian_mode == "analytic" else None,
                )
                self._policy = policy
            result = self._continuation.advance_to(target)
            self.rates(result.final)
            events = []
            if self.config.numerical_flux == "hllc":
                from scipy.optimize import brentq

                momentum = np.sum(result.y[1 : 3 * n : 3], axis=0)
                for i in range(len(result.t) - 1):
                    direction = (
                        1
                        if momentum[i] < 0 <= momentum[i + 1]
                        else -1
                        if momentum[i] > 0 >= momentum[i + 1]
                        else 0
                    )
                    if direction:
                        root = brentq(
                            lambda t: float(sum(result.dense(t)[1 : 3 * n : 3])),
                            float(result.t[i]),
                            float(result.t[i + 1]),
                            xtol=1e-12,
                        )
                        events.append(dict(kind="net_momentum_zero", t=root, direction=direction))
            self._state, self.time_s = result.final.copy(), target
            self.last_events = tuple(events)
            observed = self.observation()
        except BaseException:
            self._state, self.time_s, self.last_events = saved
            self._continuation = None
            raise
        return observed

    def observation(self):
        r = self.rates(self._state)
        c = self.config
        n = c.cells
        mass, momentum, energy = self.inventories(self._state)
        tw = self._state[3 * n : 4 * n]
        result = dict(
            time_s=self.time_s,
            stored_mass_kg=mass,
            stored_momentum_kg_m_s=momentum,
            stored_energy_j=energy,
            mass_residual_kg=mass - self.initial_mass_kg - self._state[-3],
            momentum_residual_kg_m_s=momentum - self.initial_momentum_kg_m_s - self._state[-2],
            energy_residual_j=energy - self.initial_energy_j - self._state[-1],
            kinetic_energy_j=float(sum(r["kinetic"])),
            dissipation_w=float(sum(-r["drag"] * r["velocities"])),
            min_pressure_pa=float(min(r["pressures"])),
            max_pressure_pa=float(max(r["pressures"])),
            min_temperature_k=float(min(r["temperatures"])),
            max_temperature_k=float(max(r["temperatures"])),
            min_velocity_m_s=float(min(r["velocities"])),
            max_velocity_m_s=float(max(r["velocities"])),
            min_wall_temperature_k=float(min(tw)),
            max_wall_temperature_k=float(max(tw)),
            inlet_mass_flow_kg_s=float(r["faces"][0, 0]),
            outlet_mass_flow_kg_s=float(r["faces"][-1, 0]),
            max_mach=float(r["max_mach"]),
            wall_to_fluid_w=float(sum(r["wall_heat"])),
        )
        if c.numerical_flux == "hllc":
            for side, port in self.port_observation(r).items():
                result.update({side + "_port_" + key: value for key, value in port.items()})
            for key, value in self.interval_inventory(0.0, c.total_length_m / 2).items():
                result["left_half_" + key] = value
        return result

    def interval_inventory(self, begin_m, end_m):
        begin_m = _finite(begin_m, "begin_m")
        end_m = _finite(end_m, "end_m")
        if not 0 <= begin_m < end_m <= self.config.total_length_m:
            raise ValueError("Invalid physical interval")
        n = self.config.cells
        edges = np.arange(n + 1) * self.dx
        weights = (
            np.maximum(0, np.minimum(edges[1:], end_m) - np.maximum(edges[:-1], begin_m)) / self.dx
        )
        q = self._state[: 3 * n].reshape(n, 3)
        return dict(
            mass_kg=float(weights @ q[:, 0]),
            momentum_kg_m_s=float(weights @ q[:, 1]),
            total_energy_j=float(
                weights
                @ (q[:, 2] + self.config.total_heat_capacity_j_k / n * self._state[3 * n : 4 * n])
            ),
        )

    def port_observation(self, rates=None):
        r = self.rates(self._state) if rates is None else rates
        ports = {}
        n = self.config.cells
        q = self._state[: 3 * n].reshape(n, 3)
        for side, face, coordinate in (("left", 0, 0.0), ("right", n, self.config.total_length_m)):
            port = dict(
                x_m=coordinate,
                mass_flow_kg_s=float(r["faces"][face, 0]),
                momentum_flux_n=float(r["faces"][face, 1]),
                energy_flow_w=float(r["faces"][face, 2]),
            )
            if self.config.numerical_flux == "hllc":
                from physics.hydraulics.hllc import interface

                left, right = (
                    (q[-1], q[0])
                    if self.config.boundary == "periodic"
                    else (self._ghosts[0], q[0])
                    if side == "left"
                    else (q[-1], self._ghosts[1])
                )
                data = interface(self, left, right)
                port.update(
                    contact_velocity_m_s=data["contact_velocity_m_s"],
                    hllc_pressure_pa=data["pressure_star_pa"],
                )
            ports[side] = port
        return ports

    def solver_diagnostics(self):
        return dict(
            rhs_calls=self._rhs_calls,
            analytic_jacobian_calls=self._jac_calls,
            scope="Process-local attempted callbacks, not serialized physical state",
        )

    def snapshot(self):
        return dict(
            schema_version=4 if self.config.numerical_flux == "hllc" else 3,
            model_id=self.passport.model_id,
            model_version=self.passport.version,
            config=self.config.as_dict(),
            time_s=self.time_s,
            state=self._state.tolist(),
        )

    @classmethod
    def from_snapshot(cls, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "model_id",
            "model_version",
            "config",
            "time_s",
            "state",
        }:
            raise DomainError("Invalid Euler snapshot fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] not in (3, 4):
            raise DomainError("Euler snapshot passport/schema mismatch")
        try:
            model = cls(EulerConfig(**payload["config"]))
            if (
                payload["schema_version"] != (4 if model.config.numerical_flux == "hllc" else 3)
                or payload["model_id"] != model.passport.model_id
                or payload["model_version"] != model.passport.version
            ):
                raise DomainError("Euler flux/passport/schema mismatch")
            time = _finite(payload["time_s"], "time")
            if time < 0:
                raise ValueError("Negative time")
            if not isinstance(payload["state"], list) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in payload["state"]
            ):
                raise ValueError("State must be a numeric JSON vector")
            state = np.asarray(payload["state"], dtype=float)
            model.rates(state)
        except (TypeError, ValueError) as exc:
            raise DomainError("Invalid Euler snapshot values") from exc
        model._state, model.time_s = state.copy(), time
        return model
