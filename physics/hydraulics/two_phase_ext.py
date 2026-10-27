"""Drift-flux, двухжидкостная смесь, критическое истечение и четырёхквадрантный насос."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.status import DomainError, ModelPassport, ModelStatus

DRIFT_PASSPORT = ModelPassport(
    model_id="drift-flux-zuber-findlay",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Объёмное паросодержание по соотношению дрейфа Цубера–Финдли",
    approximations=("постоянные C0 и v_gj на шаге", "равновесные плотности фаз"),
    applicability="Сопутный восходящий поток, G > 0, 0 <= x <= 1",
    data_sources=("Zuber N., Findlay J.A., J. Heat Transfer 87 (1965); C0 и v_gj — параметры замыкания",),
)

TWO_FLUID_PASSPORT = ModelPassport(
    model_id="two-fluid-lumped",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Две фазы с раздельными массами и энергиями и противоположным межфазным обменом",
    approximations=(
        "общее давление",
        "межфазный перенос на энтальпиях фаз",
        "скрытая теплота берётся из энергии фаз",
        "исчезание фазы останавливает обмен",
    ),
    applicability="Закрытый объём без отрицательной массы фазы",
    data_sources=("Балансы массы и энергии двухфазной смеси",),
)

CRITICAL_PASSPORT = ModelPassport(
    model_id="ideal-gas-choke",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Критическое истечение идеального газа",
    approximations=("изоэнтропийное сопло", "однофазный газ"),
    applicability="Давление после горла не поднимает расход выше критического",
    data_sources=("Стандартная газодинамическая формула критического расхода",),
)

PUMP4_PASSPORT = ModelPassport(
    model_id="pump-polynomial-four-quadrant",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Полиномиальный напор во всех сочетаниях знаков расхода и скорости",
    approximations=("не измеренная гомологическая кривая конкретного насоса", "гладкие нечётные степени"),
    applicability="Все четыре квадранта этой полиномиальной модели",
    data_sources=("Полином a1*W|W| + a2*W*F + a3*F|F|, коэффициенты паспорта модели",),
    notes=("Это не заводская четырёхквадрантная характеристика.",),
)


def drift_void_fraction(
    quality: float,
    rho_liquid: float,
    rho_vapor: float,
    mass_flux_kg_m2_s: float,
    distribution: float = 1.2,
    drift_velocity_m_s: float = 0.2,
) -> float:
    if quality < 0.0 or quality > 1.0:
        raise DomainError("Дрейф-флакс определён для x в [0, 1]")
    if mass_flux_kg_m2_s <= 0.0 or distribution <= 0.0:
        raise DomainError("Массовая скорость и C0 должны быть положительными")
    denominator = distribution * (quality + (1.0 - quality) * rho_vapor / rho_liquid) + rho_vapor * drift_velocity_m_s / mass_flux_kg_m2_s
    alpha = quality / denominator
    if alpha < 0.0 or alpha > 1.0:
        raise DomainError("Замыкание дрейфа дало паросодержание вне [0, 1]")
    return alpha


def hem_void_matches_drift_limit(quality: float, rho_liquid: float, rho_vapor: float) -> float:
    """При C0=1 и v_gj=0 формула дрейфа совпадает с HEM."""
    return drift_void_fraction(quality, rho_liquid, rho_vapor, mass_flux_kg_m2_s=1.0e12, distribution=1.0, drift_velocity_m_s=0.0)


@dataclass
class TwoFluidVolume:
    liquid_mass_kg: float
    vapor_mass_kg: float
    liquid_energy_j: float
    vapor_energy_j: float
    exchange_limited: bool = False

    def transfer(
        self,
        gamma_kg_s: float,
        dt_s: float,
        h_liquid_j_kg: float,
        h_vapor_j_kg: float,
        interface_heat_w: float = 0.0,
    ) -> float:
        """Перенос массы через межфазную границу.

        Положительный gamma испаряет жидкость. Скрытая теплота не возникает сама:
        интерфейс без собственной ёмкости требует Q_interface = gamma*(h_v - h_l).
        Недостающее тепло забирается из внутренней энергии жидкости, поэтому при
        нулевом внешнем подогреве сумма энергий фаз не меняется.
        """
        requested = gamma_kg_s * dt_s
        moved = requested
        self.exchange_limited = False
        if moved > 0.0 and moved > self.liquid_mass_kg:
            moved = self.liquid_mass_kg
            self.exchange_limited = True
        if moved < 0.0 and -moved > self.vapor_mass_kg:
            moved = -self.vapor_mass_kg
            self.exchange_limited = True
        heat_from_phases = moved * (h_vapor_j_kg - h_liquid_j_kg) - interface_heat_w * dt_s
        liquid_after = self.liquid_energy_j - moved * h_vapor_j_kg + interface_heat_w * dt_s
        if moved > 0.0 and liquid_after < 0.0:
            if h_vapor_j_kg <= 0.0:
                moved = 0.0
            else:
                moved = max(0.0, (self.liquid_energy_j + interface_heat_w * dt_s) / h_vapor_j_kg)
            self.exchange_limited = True
            heat_from_phases = moved * (h_vapor_j_kg - h_liquid_j_kg) - interface_heat_w * dt_s
        self.liquid_mass_kg -= moved
        self.vapor_mass_kg += moved
        self.vapor_energy_j += moved * h_vapor_j_kg
        self.liquid_energy_j += -moved * h_liquid_j_kg - heat_from_phases
        return moved

    @property
    def total_mass_kg(self) -> float:
        return self.liquid_mass_kg + self.vapor_mass_kg

    @property
    def total_energy_j(self) -> float:
        return self.liquid_energy_j + self.vapor_energy_j


@dataclass
class TwoFluidMomentum:
    """Импульсы фаз. Межфазная сила входит в обе фазы с противоположным знаком.

    Исчезновение фазы останавливает модель: делить импульс на нулевую массу нельзя.
    """

    liquid_mass_kg: float
    vapor_mass_kg: float
    liquid_velocity_m_s: float
    vapor_velocity_m_s: float

    def step(
        self,
        dt_s: float,
        force_liquid_n: float,
        force_vapor_n: float,
        interfacial_on_vapor_n: float,
    ) -> None:
        if self.liquid_mass_kg <= 0.0 or self.vapor_mass_kg <= 0.0:
            raise DomainError("Импульс фазы не определён: одна из фаз исчезла")
        if dt_s < 0.0:
            raise DomainError("Шаг импульса фаз отрицателен")
        self.liquid_velocity_m_s += dt_s * (force_liquid_n - interfacial_on_vapor_n) / self.liquid_mass_kg
        self.vapor_velocity_m_s += dt_s * (force_vapor_n + interfacial_on_vapor_n) / self.vapor_mass_kg

    def momentum_kg_m_s(self) -> float:
        return (
            self.liquid_mass_kg * self.liquid_velocity_m_s
            + self.vapor_mass_kg * self.vapor_velocity_m_s
        )


def interfacial_drag_pair(relative_velocity_m_s: float, coefficient: float) -> tuple:
    """Силы на пар и жидкость равны по модулю и противоположны."""
    force = coefficient * relative_velocity_m_s * abs(relative_velocity_m_s)
    return force, -force


def terminal_relative_velocity(density_difference: float, gravity: float, drag_coefficient: float) -> float:
    if drag_coefficient <= 0.0:
        raise DomainError("Коэффициент межфазного трения должен быть положительным")
    buoyancy = density_difference * gravity
    return math.copysign(math.sqrt(abs(buoyancy) / drag_coefficient), buoyancy)


def frozen_mixture_critical_flux(pressure_pa: float, quality: float, pressure_step_pa: float = 2.0e4) -> float:
    """Замороженное по паросодержанию критическое истечение насыщенной смеси, кг/(м²·с).

    Скорость звука берётся из dv/dp вдоль линии насыщения при фиксированном x.
    Идеальный газ и недогретая жидкость этой функцией не подменяются.
    """
    from physics.materials.iapws import saturation_properties

    if quality <= 0.0 or quality >= 1.0:
        raise DomainError("Замороженная смесь задана при 0 < x < 1; жидкость и пар — другие модели")
    if pressure_pa <= pressure_step_pa:
        raise DomainError("Давление слишком близко к границе шага dv/dp")

    def specific_volume(pressure: float) -> float:
        liquid, vapor = saturation_properties(pressure)
        return (1.0 - quality) / liquid.rho_kg_m3 + quality / vapor.rho_kg_m3

    center = specific_volume(pressure_pa)
    slope = (
        specific_volume(pressure_pa + pressure_step_pa) - specific_volume(pressure_pa - pressure_step_pa)
    ) / (2.0 * pressure_step_pa)
    if slope >= 0.0:
        raise DomainError("Смесь в этой разности не сжимается, критический поток не определён")
    sound_m_s = math.sqrt(-(center * center) / slope)
    return sound_m_s / center


def critical_mass_flux_ideal_gas(pressure_pa: float, temperature_k: float, gamma: float, gas_constant_j_kg_k: float) -> float:
    """Критический массовый поток изоэнтропийного сопла, кг/(м²·с)."""
    if gamma <= 1.0 or pressure_pa <= 0.0 or temperature_k <= 0.0:
        raise DomainError("Критическое истечение идеального газа вне области формулы")
    factor = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return pressure_pa * math.sqrt(gamma / (gas_constant_j_kg_k * temperature_k)) * factor


def choked_mass_flow(upstream_pressure_pa: float, downstream_pressure_pa: float, critical_pressure_ratio: float, critical_flow_kg_s: float, area_coefficient: float) -> float:
    """Ниже критического отношения расход не растёт при дальнейшем снижении противодавления."""
    if upstream_pressure_pa <= 0.0:
        raise DomainError("Давление перед горлом должно быть положительным")
    ratio = downstream_pressure_pa / upstream_pressure_pa
    if ratio <= critical_pressure_ratio:
        return critical_flow_kg_s
    unchoked = area_coefficient * math.sqrt(max(upstream_pressure_pa - downstream_pressure_pa, 0.0))
    return min(unchoked, critical_flow_kg_s)


def four_quadrant_head_pa(speed_ratio: float, flow_ratio: float, a_speed: float, a_cross: float, a_flow: float, head_scale_pa: float) -> float:
    """Напор во всех квадрантах. speed_ratio и flow_ratio нормированы на номинал."""
    group = a_speed * speed_ratio * abs(speed_ratio) + a_cross * speed_ratio * flow_ratio + a_flow * flow_ratio * abs(flow_ratio)
    return head_scale_pa * group


def pump_shaft_power_w(mass_flow_kg_s: float, head_pa: float, density_kg_m3: float) -> float:
    if density_kg_m3 <= 0.0:
        raise DomainError("Плотность насоса должна быть положительной")
    return mass_flow_kg_s * head_pa / density_kg_m3
