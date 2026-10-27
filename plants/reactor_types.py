"""Четыре типа с собственной топологией. Коэффициенты ВВЭР сюда не копируются."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.status import DomainError, ModelPassport, ModelStatus
from physics.materials.coolants import helium_density_kg_m3, sodium_cp_j_kg_k, sodium_density_kg_m3
from physics.materials.iapws import properties_pt, saturation_properties

BWR_PASSPORT = ModelPassport(
    model_id="bwr-direct-cycle",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Прямой цикл BWR: кипение в зоне, сепарация, пар на турбину",
    approximations=("один канал", "идеальный сепаратор", "рециркуляция возвращает насыщенную жидкость"),
    applicability="Выход зоны внутри купола при давлении ниже 16.5 МПа",
    data_sources=("IF97", "пустотный коэффициент — параметр этой схемы"),
)

SFR_PASSPORT = ModelPassport(
    model_id="sfr-sodium-loop",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Петля натрия с промежуточным теплообменником",
    approximations=("сосредоточенный натрий", "сток тепла с заданным UA", "вода в этой схеме отсутствует"),
    applicability="Натрий 400–1000 K",
    data_sources=("Линейная плотность натрия из physics.materials.coolants",),
)

HTGR_PASSPORT = ModelPassport(
    model_id="htgr-helium-brayton",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Гелиевый цикл с реакторным подогревом, турбиной и компрессором",
    approximations=("идеальный газ", "постоянные КПД машин", "графит как отдельная ёмкость"),
    applicability="Гелий 250–1500 K и давление до 10 МПа",
    data_sources=("Уравнение состояния идеального гелия",),
)

MSR_PASSPORT = ModelPassport(
    model_id="msr-circulating-fuel",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Циркулирующее топливо: адвекция предшественников между зоной и петлёй",
    approximations=("одна группа предшественников", "мгновенное перемешивание в каждом объёме"),
    applicability="Расход не меняет знак; разрушение контура не моделируется",
    data_sources=("Уравнения переноса предшественников с потоком топлива",),
)


@dataclass
class BWRPlant:
    pressure_pa: float = 7.0e6
    core_flow_kg_s: float = 12000.0
    inlet_temperature_k: float = 550.0
    fission_power_w: float = 3.0e9
    void_coefficient: float = -0.05
    alpha_ref: float = 0.4

    def __post_init__(self) -> None:
        self.passport = BWR_PASSPORT

    def separation(self) -> dict:
        inlet = properties_pt(self.pressure_pa, self.inlet_temperature_k)
        liquid, vapor = saturation_properties(self.pressure_pa)
        h_out = inlet.h_j_kg + self.fission_power_w / self.core_flow_kg_s
        if not liquid.h_j_kg < h_out < vapor.h_j_kg:
            raise DomainError("Выход активной зоны BWR вне двухфазной области")
        quality = (h_out - liquid.h_j_kg) / (vapor.h_j_kg - liquid.h_j_kg)
        steam = quality * self.core_flow_kg_s
        recirculation = self.core_flow_kg_s - steam
        heat_accounted = steam * vapor.h_j_kg + recirculation * liquid.h_j_kg - self.core_flow_kg_s * inlet.h_j_kg
        alpha = (quality / vapor.rho_kg_m3) / (quality / vapor.rho_kg_m3 + (1.0 - quality) / liquid.rho_kg_m3)
        return {
            "quality": quality,
            "steam_kg_s": steam,
            "recirculation_kg_s": recirculation,
            "heat_w": heat_accounted,
            "void": alpha,
            "reactivity": self.void_coefficient * (alpha - self.alpha_ref),
        }


@dataclass
class SFRLoop:
    mass_kg: float = 50000.0
    temperature_k: float = 700.0
    ua_w_k: float = 5.0e6
    sink_temperature_k: float = 600.0
    fission_power_w: float = 5.0e8

    def __post_init__(self) -> None:
        self.passport = SFR_PASSPORT

    def density(self) -> float:
        return sodium_density_kg_m3(self.temperature_k)

    def heat_to_sink_w(self) -> float:
        return self.ua_w_k * (self.temperature_k - self.sink_temperature_k)

    def steady_temperature(self) -> float:
        """Температура, при которой отвод равен мощности делений."""
        return self.sink_temperature_k + self.fission_power_w / self.ua_w_k

    def temperature_derivative_k_s(self) -> float:
        """m cp dT/dt = Q_fission - UA (T - T_sink)."""
        capacity = self.mass_kg * sodium_cp_j_kg_k(self.temperature_k)
        return (self.fission_power_w - self.heat_to_sink_w()) / capacity


@dataclass
class HTGRBrayton:
    reactor_power_w: float = 2.0e8
    compressor_power_w: float = 8.0e7
    turbine_power_w: float = 2.6e8
    helium_pressure_pa: float = 6.0e6
    core_temperature_k: float = 1000.0
    graphite_temperature_k: float = 1100.0
    graphite_capacity_j_k: float = 1.0e8
    ua_fuel_graphite: float = 2.0e6

    def __post_init__(self) -> None:
        self.passport = HTGR_PASSPORT

    def cooler_power_w(self) -> float:
        """Баланс цикла: подогрев + компрессор = турбина + холодильник."""
        return self.reactor_power_w + self.compressor_power_w - self.turbine_power_w

    def helium_density(self) -> float:
        return helium_density_kg_m3(self.helium_pressure_pa, self.core_temperature_k)

    def net_electric_w(self) -> float:
        return self.turbine_power_w - self.compressor_power_w

    def graphite_temperature_derivative_k_s(self) -> float:
        """C dT_g/dt = UA (T_core - T_g). Гелий этим уравнением не подменяется."""
        if self.graphite_capacity_j_k <= 0.0 or self.ua_fuel_graphite <= 0.0:
            raise DomainError("Теплоёмкость графита и UA должны быть положительными")
        return (
            self.ua_fuel_graphite
            * (self.core_temperature_k - self.graphite_temperature_k)
            / self.graphite_capacity_j_k
        )


@dataclass
class MSRPrecursors:
    core_volume_m3: float = 5.0
    loop_volume_m3: float = 5.0
    flow_m3_s: float = 0.5
    decay_constant: float = 0.1
    core_precursors: float = 1.0
    loop_precursors: float = 1.0

    def __post_init__(self) -> None:
        self.passport = MSR_PASSPORT

    def derivatives(self, production_per_s: float = 0.0) -> tuple:
        exchange = self.flow_m3_s * (
            self.core_precursors / self.core_volume_m3 - self.loop_precursors / self.loop_volume_m3
        )
        d_core = production_per_s - self.decay_constant * self.core_precursors - exchange
        d_loop = -self.decay_constant * self.loop_precursors + exchange
        return d_core, d_loop

    def advance(self, dt_s: float, production_per_s: float = 0.0) -> None:
        # Линейная система на шаге с замороженным источником.
        rate = self.flow_m3_s
        lam = self.decay_constant
        vc = self.core_volume_m3
        vl = self.loop_volume_m3
        matrix = np.array(
            [
                [-lam - rate / vc, rate / vl],
                [rate / vc, -lam - rate / vl],
            ]
        )
        from scipy.linalg import expm

        state = expm(matrix * dt_s) @ np.array([self.core_precursors, self.loop_precursors])
        if production_per_s != 0.0:
            # Частное решение для постоянного источника только в зоне.
            particular = np.linalg.solve(matrix, -np.array([production_per_s, 0.0]))
            transient = expm(matrix * dt_s) @ (np.array([self.core_precursors, self.loop_precursors]) - particular)
            state = transient + particular
        self.core_precursors = float(state[0])
        self.loop_precursors = float(state[1])
