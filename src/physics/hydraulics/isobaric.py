"""Консервативный канал с предписанными давлением, входом и объёмным нагревом.

В фиксированной ячейке M=rho(p,h)V, U=M h-pV. При постоянном p:
M dh/dt = m_in(h_in-h)+Q; m_out=m_in-(dM/dh) dh/dt.
Это открытый тепловой компонент, без уравнения импульса и модели CHF.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from core.continuation import ImplicitContinuation
from core.solvers import integrate
from core.status import DomainError, ModelPassport, ModelStatus
from physics.materials.iapws import (
    P_MIN_MPA,
    P_REGION13_MPA,
    _region1_raw,
    _region2_raw,
    properties_ph,
    saturation_properties,
)

PASSPORT = ModelPassport(
    model_id="isobaric-heated-channel",
    version="0.1.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Нестационарное равновесное кипение в открытом канале при заданном давлении",
    approximations=(
        "одинаковое заданное давление во всех ячейках, акустика и импульс исключены",
        "равновесные фазы без скольжения, донорная энтальпия и идеальное смешение ячейки",
        "положительный или нулевой вход, только прямые межъячеечные потоки",
        "нагрев предписан; стенка, кипящий теплообмен и CHF не предсказываются",
        "нет геометрии блока, нейтроники, насоса и естественной циркуляции",
    ),
    applicability="IF97 области 1/2/4 до 16.529 МПа; без сверхкритической области",
    data_sources=(
        "https://iapws.org/technical-guidance/release/IF97-Rev",
        "Нестационарные балансы массы и внутренней энергии фиксированного объёма",
    ),
)


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s должен быть конечным числом" % name)
    if positive and value <= 0:
        raise ValueError("%s должен быть положительным" % name)
    return float(value)


@dataclass(frozen=True)
class ChannelConfig:
    cells: int
    total_volume_m3: float
    pressure_pa: float
    inlet_enthalpy_j_kg: float
    inlet_mass_flow_kg_s: float
    heat_w: float
    flow_decay_s: float | None = None
    heat_decay_s: float | None = None
    initial_enthalpy_j_kg: tuple | None = None

    def __post_init__(self):
        if type(self.cells) is not int or not 1 <= self.cells <= 256:
            raise ValueError("Число ячеек должно быть целым от 1 до 256")
        for key in ("total_volume_m3", "pressure_pa"):
            _finite(getattr(self, key), key, positive=True)
        _finite(self.inlet_enthalpy_j_kg, "inlet_enthalpy_j_kg")
        if not P_MIN_MPA * 1e6 < self.pressure_pa <= P_REGION13_MPA * 1e6:
            raise DomainError("Давление вне паспорта изобарического канала")
        for key in ("inlet_mass_flow_kg_s", "heat_w"):
            if _finite(getattr(self, key), key) < 0:
                raise ValueError("%s не может быть отрицательным" % key)
        for key in ("flow_decay_s", "heat_decay_s"):
            if getattr(self, key) is not None:
                _finite(getattr(self, key), key, positive=True)
        initial = self.initial_enthalpy_j_kg
        if initial is None:
            initial = (self.inlet_enthalpy_j_kg,) * self.cells
        if not isinstance(initial, (list, tuple)) or len(initial) != self.cells:
            raise ValueError("Начальная энтальпия должна содержать значение каждой ячейки")
        object.__setattr__(
            self,
            "initial_enthalpy_j_kg",
            tuple(_finite(value, "initial_enthalpy") for value in initial),
        )


@dataclass(frozen=True)
class IsobaricState:
    density: float
    density_dh: float
    temperature: float
    quality: float
    void: float


@lru_cache(maxsize=2048)
def isobaric_state(pressure_pa: float, enthalpy_j_kg: float) -> IsobaricState:
    """Производная rho по h при p=const; cp смеси для неё не используется."""
    state = properties_ph(pressure_pa, enthalpy_j_kg)
    if state.region == 4:
        liquid, vapor = saturation_properties(pressure_pa)
        dv_dh = (vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg)
        void = state.quality * vapor.v_m3_kg / state.v_m3_kg
    elif state.region in (1, 2):
        raw = (_region1_raw if state.region == 1 else _region2_raw)(state.T_k, pressure_pa * 1e-6)
        dv_dh = raw["dv_dt"] / state.cp_j_kg_k
        void = 0.0 if state.region == 1 else 1.0
    else:
        raise DomainError("Область EOS вне паспорта изобарического канала")
    return IsobaricState(
        state.rho_kg_m3, -(state.rho_kg_m3**2) * dv_dh, state.T_k, state.quality, void
    )


class IsobaricChannel:
    passport = PASSPORT

    def __init__(self, config: ChannelConfig):
        self.config = config
        # Вход может отличаться от начального состояния ячеек: проверяем его
        # отдельно до создания эксперимента и до первого шага.
        isobaric_state(config.pressure_pa, config.inlet_enthalpy_j_kg)
        self.time_s = 0.0
        self._state = np.r_[config.initial_enthalpy_j_kg, 0.0, 0.0].astype(float)
        self._continuation = None
        self._horizon = None
        self._policy = None
        self.initial_mass_kg, self.initial_energy_j = self.inventories(self._state)
        self.rates(0.0, self._state)

    def inventories(self, state):
        c = self.config
        h = np.asarray(state[: c.cells])
        mass = np.array([isobaric_state(c.pressure_pa, float(v)).density for v in h])
        mass *= c.total_volume_m3 / c.cells
        return float(sum(mass)), float(np.dot(mass, h) - c.pressure_pa * c.total_volume_m3)

    def rates(self, t, state, *, cell_heat_w=None):
        c = self.config
        h = np.asarray(state[: c.cells], dtype=float)
        if h.shape != (c.cells,) or not np.all(np.isfinite(h)):
            raise DomainError("Неверный вектор энтальпий канала")
        flow_in = c.inlet_mass_flow_kg_s * (
            1.0 if c.flow_decay_s is None else math.exp(-t / c.flow_decay_s)
        )
        heat = c.heat_w * (1.0 if c.heat_decay_s is None else math.exp(-t / c.heat_decay_s))
        if cell_heat_w is None:
            heating = np.full(c.cells, heat / c.cells)
        else:
            heating = np.asarray(cell_heat_w, dtype=float)
            if heating.shape != (c.cells,) or not np.all(np.isfinite(heating)):
                raise DomainError("Неверный вектор тепловых потоков в ячейки")
            heat = float(sum(heating))
        flow, upstream = flow_in, c.inlet_enthalpy_j_kg
        dh, dm, du, flows = [], [], [], [flow]
        volume = c.total_volume_m3 / c.cells
        for index, value in enumerate(h):
            thermo = isobaric_state(c.pressure_pa, float(value))
            mass = thermo.density * volume
            rate_h = (flow * (upstream - value) + heating[index]) / mass
            rate_m = thermo.density_dh * volume * rate_h
            outgoing = flow - rate_m
            if not math.isfinite(outgoing) or outgoing < 0:
                raise DomainError("Обратный поток вне паспорта изобарического канала")
            dh.append(rate_h)
            dm.append(rate_m)
            du.append(value * rate_m + mass * rate_h)
            flow, upstream = outgoing, value
            flows.append(flow)
        return {
            "dh": np.array(dh),
            "dm": np.array(dm),
            "du": np.array(du),
            "flows": np.array(flows),
            "heat_w": heat,
            "mass_boundary_kg_s": flow_in - flow,
            "energy_boundary_w": flow_in * c.inlet_enthalpy_j_kg - flow * h[-1] + heat,
        }

    def derivatives(self, t, state):
        rates = self.rates(t, state)
        return np.r_[rates["dh"], rates["mass_boundary_kg_s"], rates["energy_boundary_w"]]

    def enable_continuation(self, horizon_s):
        if not math.isfinite(horizon_s) or horizon_s <= self.time_s:
            raise ValueError("Горизонт должен быть позже времени канала")
        self._horizon, self._continuation = horizon_s, None

    def absolute_tolerances(self):
        return np.r_[np.full(self.config.cells, 1e-4), 1e-9, 0.001]

    def advance(self, dt_s, *, rtol=1e-6, method="Radau", max_step_s=1.0):
        _finite(dt_s, "dt_s", positive=True)
        _finite(rtol, "rtol", positive=True)
        _finite(max_step_s, "max_step_s", positive=True)
        initial, t0 = self._state.copy(), self.time_s
        target = t0 + dt_s
        atol = self.absolute_tolerances()
        policy = (rtol, method, max_step_s)
        try:
            if self._horizon is None:
                result = integrate(
                    self.derivatives,
                    initial,
                    (t0, target),
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step_s,
                    method=method,
                )
            else:
                if self._continuation is None or self._policy != policy:
                    self._continuation = ImplicitContinuation(
                        self.derivatives,
                        initial,
                        t0,
                        self._horizon,
                        rtol=rtol,
                        atol=atol,
                        max_step=max_step_s,
                        method=method,
                        initial_target=target,
                    )
                    self._policy = policy
                result = self._continuation.advance_to(target)
            self.rates(target, result.final)
            self._state, self.time_s = result.final.copy(), target
            return self.observation()
        except BaseException:
            self._state, self.time_s, self._continuation = initial, t0, None
            raise

    def observation(self):
        c = self.config
        states = [isobaric_state(c.pressure_pa, float(h)) for h in self._state[: c.cells]]
        mass, energy = self.inventories(self._state)
        rates = self.rates(self.time_s, self._state)
        return {
            "time_s": self.time_s,
            "pressure_pa": c.pressure_pa,
            "outlet_enthalpy_j_kg": float(self._state[c.cells - 1]),
            "outlet_quality": states[-1].quality,
            "max_quality": max(s.quality for s in states),
            "outlet_void": states[-1].void,
            "outlet_temperature_k": states[-1].temperature,
            "inlet_mass_flow_kg_s": float(rates["flows"][0]),
            "outlet_mass_flow_kg_s": float(rates["flows"][-1]),
            "heat_w": rates["heat_w"],
            "stored_mass_kg": mass,
            "stored_energy_j": energy,
            "mass_residual_kg": mass - self.initial_mass_kg - self._state[-2],
            "energy_residual_j": energy - self.initial_energy_j - self._state[-1],
        }

    def snapshot(self):
        config = asdict(self.config)
        config["initial_enthalpy_j_kg"] = list(self.config.initial_enthalpy_j_kg)
        return {
            "schema_version": 1,
            "model_id": self.passport.model_id,
            "model_version": self.passport.version,
            "config": config,
            "time_s": self.time_s,
            "state": self._state.tolist(),
        }

    @classmethod
    def from_snapshot(cls, payload):
        expected = {"schema_version", "model_id", "model_version", "config", "time_s", "state"}
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or type(payload.get("schema_version")) is not int
        ):
            raise DomainError("Неверная схема checkpoint изобарического канала")
        if (
            payload.get("schema_version") != 1
            or payload.get("model_id") != PASSPORT.model_id
            or payload.get("model_version") != PASSPORT.version
        ):
            raise DomainError("Несовместимый checkpoint изобарического канала")
        try:
            config = ChannelConfig(**payload["config"])
            state = np.asarray(payload["state"], dtype=float)
            t = _finite(payload["time_s"], "time_s")
        except (TypeError, ValueError) as exc:
            raise DomainError("Неверные поля checkpoint канала") from exc
        if state.shape != (config.cells + 2,) or not np.all(np.isfinite(state)):
            raise DomainError("Неконечный или неверный вектор checkpoint канала")
        if t < 0:
            raise DomainError("Отрицательное время checkpoint канала")
        candidate = cls(config)
        candidate.rates(t, state)
        candidate._state, candidate.time_s = state.copy(), t
        return candidate
