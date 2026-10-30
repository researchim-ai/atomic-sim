"""Тепловая инерция стенки открытого канала при предписанном постоянном UA.

Внешняя мощность греет стенку; внутренний Q_i=UA_i(Tw_i-T_i) входит в энергию
жидкости с плюсом и в энергию стенки с минусом. Это не корреляция кипения/CHF.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.isobaric import ChannelConfig, IsobaricChannel, _finite, isobaric_state

PASSPORT = ModelPassport(
    model_id="isobaric-wall-heated-channel",
    version="0.1.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Равновесное кипение с запасом энергии в сосредоточенной стенке",
    approximations=(
        "открытый изобарический канал, импульс и динамика давления исключены",
        "одна температура стенки на ячейку, постоянные C и UA заданы пользователем",
        "внешняя мощность поступает в стенку; тепло стенка-жидкость внутреннее",
        "нет корреляции кипящего теплообмена, CHF, напряжений и реальной геометрии блока",
        "изменение расхода и мощности предписано, нейтроника и насос отсутствуют",
    ),
    applicability="IF97 области 1/2/4; прямые потоки, положительные температуры стенки",
    data_sources=(
        "Баланс внутренней энергии и закон Q=UA(Tw-T)",
        "https://iapws.org/technical-guidance/release/IF97-Rev",
    ),
)


@dataclass(frozen=True)
class WallConfig:
    channel: ChannelConfig
    total_heat_capacity_j_k: float
    total_ua_w_k: float
    initial_temperature_k: tuple | None = None
    power_off_time_s: float | None = None

    def __post_init__(self):
        if not isinstance(self.channel, ChannelConfig):
            raise ValueError("Стенка требует ChannelConfig")
        _finite(self.total_heat_capacity_j_k, "wall heat capacity", positive=True)
        _finite(self.total_ua_w_k, "wall UA", positive=True)
        if self.power_off_time_s is not None:
            if _finite(self.power_off_time_s, "power_off_time_s") < 0:
                raise ValueError("Момент отключения нагрева не может быть отрицательным")
        temperatures = self.initial_temperature_k
        if temperatures is None:
            power = self.channel.heat_w if self.power_off_time_s != 0 else 0.0
            temperatures = tuple(
                float(isobaric_state(self.channel.pressure_pa, h).temperature)
                + power / self.total_ua_w_k
                for h in self.channel.initial_enthalpy_j_kg
            )
        if not isinstance(temperatures, (list, tuple)) or len(temperatures) != self.channel.cells:
            raise ValueError("Начальные температуры стенки должны соответствовать ячейкам")
        object.__setattr__(
            self,
            "initial_temperature_k",
            tuple(_finite(value, "wall temperature", positive=True) for value in temperatures),
        )


class WallChannel(IsobaricChannel):
    passport = PASSPORT

    def __init__(self, config: WallConfig):
        self.wall_config, self.config = config, config.channel
        isobaric_state(self.config.pressure_pa, self.config.inlet_enthalpy_j_kg)
        self.time_s = 0.0
        self._state = np.r_[
            self.config.initial_enthalpy_j_kg, config.initial_temperature_k, 0.0, 0.0
        ].astype(float)
        self._continuation = self._horizon = self._policy = None
        self._source_side = self._last_source_side = None
        self.initial_mass_kg, self.initial_energy_j = self.inventories(self._state)
        self.rates(0.0, self._state)

    def wall_energy_j(self, state):
        n = self.config.cells
        return float(sum(state[n : 2 * n]) * self.wall_config.total_heat_capacity_j_k / n)

    def inventories(self, state):
        mass, fluid_energy = IsobaricChannel.inventories(self, state)
        return mass, fluid_energy + self.wall_energy_j(state)

    def external_power_w(self, t):
        off = self.wall_config.power_off_time_s
        active = (off is None or t < off) if self._source_side is None else self._source_side
        if not active:
            return 0.0
        c = self.config
        return c.heat_w * (1.0 if c.heat_decay_s is None else math.exp(-t / c.heat_decay_s))

    def rates(self, t, state):
        n, w = self.config.cells, self.wall_config
        tw = np.asarray(state[n : 2 * n], dtype=float)
        if tw.shape != (n,) or not np.all(np.isfinite(tw)) or np.any(tw <= 0):
            raise DomainError("Неконечная или неположительная температура стенки")
        fluid_t = np.array(
            [isobaric_state(self.config.pressure_pa, float(h)).temperature for h in state[:n]]
        )
        heat = w.total_ua_w_k / n * (tw - fluid_t)
        rates = IsobaricChannel.rates(self, t, state, cell_heat_w=heat)
        power = self.external_power_w(t)
        d_wall = (power / n - heat) / (w.total_heat_capacity_j_k / n)
        rates.update(
            d_wall=d_wall,
            wall_to_fluid_w=float(sum(heat)),
            heat_w=power,
            wall_energy_rate_w=float(sum(d_wall) * w.total_heat_capacity_j_k / n),
        )
        rates["energy_boundary_w"] = (
            rates["flows"][0] * self.config.inlet_enthalpy_j_kg
            - rates["flows"][-1] * state[n - 1]
            + power
        )
        return rates

    def derivatives(self, t, state):
        rates = self.rates(t, state)
        return np.r_[
            rates["dh"], rates["d_wall"], rates["mass_boundary_kg_s"], rates["energy_boundary_w"]
        ]

    def absolute_tolerances(self):
        n = self.config.cells
        return np.r_[np.full(n, 1e-4), np.full(n, 1e-8), 1e-9, 0.001]

    def advance(self, dt_s, **options):
        _finite(dt_s, "dt_s", positive=True)
        saved, t0 = self._state.copy(), self.time_s
        target, off = t0 + dt_s, self.wall_config.power_off_time_s
        ends = [off, target] if off is not None and t0 < off < target else [target]
        try:
            for end in ends:
                active = off is None or self.time_s < off
                if active != self._last_source_side:
                    self._continuation = None
                self._source_side, self._last_source_side = active, active
                super().advance(end - self.time_s, **options)
            self._source_side = None
            return self.observation()
        except BaseException:
            self._state, self.time_s, self._continuation = saved, t0, None
            self._last_source_side = None
            raise
        finally:
            self._source_side = None

    def observation(self):
        result = super().observation()
        n = self.config.cells
        rates = self.rates(self.time_s, self._state)
        result.update(
            wall_energy_j=self.wall_energy_j(self._state),
            fluid_energy_j=IsobaricChannel.inventories(self, self._state)[1],
            min_wall_temperature_k=float(min(self._state[n : 2 * n])),
            max_wall_temperature_k=float(max(self._state[n : 2 * n])),
            wall_to_fluid_w=rates["wall_to_fluid_w"],
        )
        return result

    def snapshot(self):
        payload = super().snapshot()
        config = asdict(self.wall_config)
        config["channel"] = payload["config"]
        config["initial_temperature_k"] = list(self.wall_config.initial_temperature_k)
        payload["config"] = config
        return payload

    @classmethod
    def from_snapshot(cls, payload):
        expected = {"schema_version", "model_id", "model_version", "config", "time_s", "state"}
        if (
            not isinstance(payload, dict)
            or set(payload) != expected
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != PASSPORT.model_id
            or payload["model_version"] != PASSPORT.version
        ):
            raise DomainError("Несовместимый checkpoint стенки канала")
        try:
            raw = dict(payload["config"])
            channel = ChannelConfig(**raw.pop("channel"))
            config = WallConfig(channel=channel, **raw)
            t = _finite(payload["time_s"], "time_s")
            state = np.asarray(payload["state"], dtype=float)
        except (TypeError, ValueError, KeyError) as exc:
            raise DomainError("Неверные поля checkpoint стенки") from exc
        if t < 0 or state.shape != (2 * channel.cells + 2,) or not np.all(np.isfinite(state)):
            raise DomainError("Неверный вектор или время checkpoint стенки")
        candidate = cls(config)
        candidate.rates(t, state)
        candidate.time_s, candidate._state = t, state.copy()
        return candidate
