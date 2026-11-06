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
    data_sources=(
        "Zuber N., Findlay J.A., J. Heat Transfer 87 (1965); C0 и v_gj — параметры замыкания",
    ),
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
    approximations=(
        "не измеренная гомологическая кривая конкретного насоса",
        "гладкие нечётные степени",
    ),
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
    denominator = (
        distribution * (quality + (1.0 - quality) * rho_vapor / rho_liquid)
        + rho_vapor * drift_velocity_m_s / mass_flux_kg_m2_s
    )
    alpha = quality / denominator
    if alpha < 0.0 or alpha > 1.0:
        raise DomainError("Замыкание дрейфа дало паросодержание вне [0, 1]")
    return alpha


def quality_from_drift_void(
    void: float,
    rho_liquid: float,
    rho_vapor: float,
    mass_flux_kg_m2_s: float,
    distribution: float = 1.2,
    drift_velocity_m_s: float = 0.2,
) -> float:
    """Обращение drift_void_fraction. Неположительный знаменатель не заменяется ближайшим x."""
    if void < 0.0 or void > 1.0:
        raise DomainError("Объёмная доля дрейфа вне [0, 1]")
    if rho_liquid <= 0.0 or rho_vapor <= 0.0 or mass_flux_kg_m2_s <= 0.0 or distribution <= 0.0:
        raise DomainError("Плотности, скорость и C0 обращения дрейфа должны быть положительными")
    if void == 0.0:
        return 0.0
    if void == 1.0:
        return 1.0
    density_ratio = rho_vapor / rho_liquid
    slip = rho_vapor * drift_velocity_m_s / mass_flux_kg_m2_s
    numerator = void * (distribution * density_ratio + slip)
    denominator = 1.0 - void * distribution * (1.0 - density_ratio)
    if denominator <= 0.0:
        raise DomainError("Обращение дрейфа не имеет паросодержания в [0, 1]")
    quality = numerator / denominator
    if quality < 0.0 or quality > 1.0:
        raise DomainError("Обращение дрейфа вышло за [0, 1]")
    return quality


def wall_force_split(force_n: float, liquid_fraction: float) -> tuple:
    """Сила стенки делится между фазами так же, как тепло: сумма частей равна силе."""
    return wall_heat_split(force_n, liquid_fraction)


def condensation_onset_length_m(
    mass_flow_kg_s: float,
    specific_heat_j_kg_k: float,
    inlet_temperature_k: float,
    saturation_temperature_k: float,
    linear_cooling_w_m: float,
) -> float:
    """Длина охлаждения перегретого пара до насыщения. На насыщении длина равна нулю."""
    if mass_flow_kg_s <= 0.0 or specific_heat_j_kg_k <= 0.0 or linear_cooling_w_m <= 0.0:
        raise DomainError(
            "Расход, теплоёмкость и съём тепла конденсации должны быть положительными"
        )
    if inlet_temperature_k < saturation_temperature_k:
        raise DomainError("Вход холоднее насыщения: это уже не снятие перегрева")
    if inlet_temperature_k == saturation_temperature_k:
        return 0.0
    return (
        mass_flow_kg_s
        * specific_heat_j_kg_k
        * (inlet_temperature_k - saturation_temperature_k)
        / linear_cooling_w_m
    )


def boiling_onset_length_m(
    mass_flow_kg_s: float,
    specific_heat_j_kg_k: float,
    inlet_temperature_k: float,
    saturation_temperature_k: float,
    linear_power_w_m: float,
) -> float:
    """Длина подогрева до насыщения. На входе в насыщении длина равна нулю."""
    if mass_flow_kg_s <= 0.0 or specific_heat_j_kg_k <= 0.0 or linear_power_w_m <= 0.0:
        raise DomainError(
            "Расход, теплоёмкость и линейная мощность канала должны быть положительными"
        )
    if inlet_temperature_k > saturation_temperature_k:
        raise DomainError("Вход горячее насыщения: это уже не подогрев до кипения")
    if inlet_temperature_k == saturation_temperature_k:
        return 0.0
    return (
        mass_flow_kg_s
        * specific_heat_j_kg_k
        * (saturation_temperature_k - inlet_temperature_k)
        / linear_power_w_m
    )


def channel_exit_quality(
    distance_m: float,
    mass_flow_kg_s: float,
    specific_heat_j_kg_k: float,
    inlet_temperature_k: float,
    saturation_temperature_k: float,
    linear_power_w_m: float,
    latent_heat_j_kg: float,
    inlet_quality: float = 0.0,
) -> float:
    """До сечения кипения паросодержание на недогретом входе равно нулю, дальше растёт от остатка тепла."""
    if distance_m < 0.0:
        raise DomainError("Длина канала отрицательна")
    onset = boiling_onset_length_m(
        mass_flow_kg_s,
        specific_heat_j_kg_k,
        inlet_temperature_k,
        saturation_temperature_k,
        linear_power_w_m,
    )
    if distance_m <= onset:
        return 0.0 if onset > 0.0 else inlet_quality
    remaining = linear_power_w_m * (distance_m - onset)
    start = inlet_quality if onset == 0.0 else 0.0
    return equilibrium_quality(remaining, mass_flow_kg_s, latent_heat_j_kg, start)


def equilibrium_quality(
    heat_w: float, mass_flow_kg_s: float, latent_heat_j_kg: float, inlet_quality: float = 0.0
) -> float:
    """Равновесное паросодержание: x = x_in + Q / (m h_fg). Выход за купол останавливает расчёт."""
    if mass_flow_kg_s <= 0.0 or latent_heat_j_kg <= 0.0:
        raise DomainError("Расход и теплота парообразования канала должны быть положительными")
    if inlet_quality < 0.0 or inlet_quality > 1.0:
        raise DomainError("Входное паросодержание канала лежит в [0, 1]")
    quality = inlet_quality + heat_w / (mass_flow_kg_s * latent_heat_j_kg)
    if quality < 0.0 or quality > 1.0:
        raise DomainError("Подогрев канала выводит паросодержание за купол")
    return quality


def wall_heat_split(heat_w: float, liquid_fraction: float) -> tuple:
    """Тепло стенки делится между фазами. Сумма частей равна исходному теплу."""
    if liquid_fraction < 0.0 or liquid_fraction > 1.0:
        raise DomainError("Доля тепла на жидкость должна быть в [0, 1]")
    liquid = heat_w * liquid_fraction
    return liquid, heat_w - liquid


def limited_phase_heat_w(energy_j: float, heat_w: float, dt_s: float) -> float:
    """Подогрев, который не делает энергию фазы отрицательной."""
    if dt_s <= 0.0:
        raise DomainError("Шаг ограничения тепла фазы должен быть положительным")
    if energy_j + heat_w * dt_s >= 0.0:
        return heat_w
    return -energy_j / dt_s


def isenthalpic_quality(enthalpy_j_kg: float, pressure_pa: float) -> float:
    """Паросодержание при фиксированной энтальпии. Вне купола однофазное состояние не подставляется."""
    from physics.materials.iapws import saturation_properties

    liquid, vapor = saturation_properties(pressure_pa)
    span = vapor.h_j_kg - liquid.h_j_kg
    if span <= 0.0:
        raise DomainError("Теплота парообразования не положительна")
    quality = (enthalpy_j_kg - liquid.h_j_kg) / span
    if quality <= 0.0 or quality >= 1.0:
        raise DomainError("Энтальпия вне купола: однофазное состояние этой функцией не подменяется")
    return quality


def interface_energy_split_j(
    moved_kg: float, h_liquid_j_kg: float, h_vapor_j_kg: float, interface_heat_j: float
) -> tuple:
    """Пар получает m h_v. Жидкость отдаёт m h_l и скрытую теплоту, которую не принёс интерфейс.

    Сумма изменений энергии фаз равна теплу интерфейса.
    """
    latent = moved_kg * (h_vapor_j_kg - h_liquid_j_kg)
    vapor_delta = moved_kg * h_vapor_j_kg
    liquid_delta = -moved_kg * h_liquid_j_kg - (latent - interface_heat_j)
    return liquid_delta, vapor_delta


def hem_void_matches_drift_limit(quality: float, rho_liquid: float, rho_vapor: float) -> float:
    """При C0=1 и v_gj=0 формула дрейфа совпадает с HEM."""
    return drift_void_fraction(
        quality,
        rho_liquid,
        rho_vapor,
        mass_flux_kg_m2_s=1.0e12,
        distribution=1.0,
        drift_velocity_m_s=0.0,
    )


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
        self.liquid_velocity_m_s += (
            dt_s * (force_liquid_n - interfacial_on_vapor_n) / self.liquid_mass_kg
        )
        self.vapor_velocity_m_s += (
            dt_s * (force_vapor_n + interfacial_on_vapor_n) / self.vapor_mass_kg
        )

    def momentum_kg_m_s(self) -> float:
        return (
            self.liquid_mass_kg * self.liquid_velocity_m_s
            + self.vapor_mass_kg * self.vapor_velocity_m_s
        )


def interfacial_drag_pair(relative_velocity_m_s: float, coefficient: float) -> tuple:
    """Силы на пар и жидкость равны по модулю и противоположны."""
    force = coefficient * relative_velocity_m_s * abs(relative_velocity_m_s)
    return force, -force


def terminal_relative_velocity(
    density_difference: float, gravity: float, drag_coefficient: float
) -> float:
    if drag_coefficient <= 0.0:
        raise DomainError("Коэффициент межфазного трения должен быть положительным")
    buoyancy = density_difference * gravity
    return math.copysign(math.sqrt(abs(buoyancy) / drag_coefficient), buoyancy)


def frozen_mixture_critical_flux(
    pressure_pa: float, quality: float, pressure_step_pa: float = 2.0e4
) -> float:
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
        specific_volume(pressure_pa + pressure_step_pa)
        - specific_volume(pressure_pa - pressure_step_pa)
    ) / (2.0 * pressure_step_pa)
    if slope >= 0.0:
        raise DomainError("Смесь в этой разности не сжимается, критический поток не определён")
    sound_m_s = math.sqrt(-(center * center) / slope)
    return sound_m_s / center


def equilibrium_mixture_critical_flux(
    pressure_pa: float, quality: float, pressure_step_pa: float = 2.0e4
) -> float:
    """Критический поток гомогенного равновесия, кг/(м²·с).

    Удельный объём берётся вдоль изоэнтропы: паросодержание на соседнем давлении
    меняется. Замороженная смесь и идеальный газ этой функцией не подменяются.
    """
    from physics.materials.iapws import saturation_properties

    if quality <= 0.0 or quality >= 1.0:
        raise DomainError("Равновесная смесь задана при 0 < x < 1")
    if pressure_pa <= pressure_step_pa:
        raise DomainError("Давление слишком близко к границе шага dv/dp")
    liquid, vapor = saturation_properties(pressure_pa)
    entropy = (1.0 - quality) * liquid.s_j_kg_k + quality * vapor.s_j_kg_k

    def specific_volume(pressure: float) -> float:
        left, right = saturation_properties(pressure)
        span = right.s_j_kg_k - left.s_j_kg_k
        if span <= 0.0:
            raise DomainError("Энтропия насыщения не разделяет фазы")
        throat = (entropy - left.s_j_kg_k) / span
        if throat <= 0.0 or throat >= 1.0:
            raise DomainError("Изоэнтропа равновесной смеси вышла из двухфазной области")
        return (1.0 - throat) / left.rho_kg_m3 + throat / right.rho_kg_m3

    center = specific_volume(pressure_pa)
    slope = (
        specific_volume(pressure_pa + pressure_step_pa)
        - specific_volume(pressure_pa - pressure_step_pa)
    ) / (2.0 * pressure_step_pa)
    if slope >= 0.0:
        raise DomainError("Равновесная смесь в этой разности не сжимается")
    sound_m_s = math.sqrt(-(center * center) / slope)
    return sound_m_s / center


def ideal_gas_sound_speed_m_s(
    gamma: float, gas_constant_j_kg_k: float, temperature_k: float
) -> float:
    """Скорость звука идеального газа sqrt(γ R T). На горле она равна скорости потока."""
    if gamma <= 1.0 or gas_constant_j_kg_k <= 0.0 or temperature_k <= 0.0:
        raise DomainError("Скорость звука задана вне области идеального газа")
    return math.sqrt(gamma * gas_constant_j_kg_k * temperature_k)


def acoustic_transit_s(length_m: float, sound_speed_m_s: float) -> float:
    if length_m < 0.0 or sound_speed_m_s <= 0.0:
        raise DomainError("Длина пробега звука неотрицательна, скорость положительна")
    return length_m / sound_speed_m_s


def critical_density_ratio(gamma: float) -> float:
    """ρ*/ρ0 = (T*/T0)^{1/(γ−1)}."""
    if gamma <= 1.0:
        raise DomainError("Показатель адиабаты критической плотности должен быть больше единицы")
    return critical_temperature_ratio(gamma) ** (1.0 / (gamma - 1.0))


def ideal_gas_cp_j_kg_k(gamma: float, gas_constant_j_kg_k: float) -> float:
    """cp = γ R / (γ − 1) для идеального газа с постоянными теплоёмкостями."""
    if gamma <= 1.0 or gas_constant_j_kg_k <= 0.0:
        raise DomainError("Теплоёмкость идеального газа задана вне области")
    return gamma * gas_constant_j_kg_k / (gamma - 1.0)


def isentropic_speed_m_s(
    gamma: float,
    gas_constant_j_kg_k: float,
    stagnation_temperature_k: float,
    static_temperature_k: float,
) -> float:
    """Скорость изоэнтропного разгона: sqrt(2 cp (T0 − T))."""
    if static_temperature_k <= 0.0 or static_temperature_k > stagnation_temperature_k:
        raise DomainError(
            "Статическая температура изоэнтропы лежит между нулём и температурой торможения"
        )
    capacity = ideal_gas_cp_j_kg_k(gamma, gas_constant_j_kg_k)
    return math.sqrt(2.0 * capacity * (stagnation_temperature_k - static_temperature_k))


def throat_mass_flux_ideal_gas(
    pressure_pa: float, temperature_k: float, gamma: float, gas_constant_j_kg_k: float
) -> float:
    """Критический поток как ρ* a*. Совпадает с формулой сопла через давление торможения."""
    if pressure_pa <= 0.0 or temperature_k <= 0.0 or gas_constant_j_kg_k <= 0.0 or gamma <= 1.0:
        raise DomainError("Поток на горле задан вне области идеального газа")
    temperature_ratio = critical_temperature_ratio(gamma)
    density = pressure_pa / (gas_constant_j_kg_k * temperature_k) * critical_density_ratio(gamma)
    sound = ideal_gas_sound_speed_m_s(gamma, gas_constant_j_kg_k, temperature_k * temperature_ratio)
    return density * sound


def critical_temperature_ratio(gamma: float) -> float:
    """T*/T0 изоэнтропийного сопла. Давление на пороге равно этой температуре в степени γ/(γ−1)."""
    if gamma <= 1.0:
        raise DomainError("Показатель адиабаты критической температуры должен быть больше единицы")
    return 2.0 / (gamma + 1.0)


def critical_pressure_ratio(gamma: float) -> float:
    """p*/p0 изоэнтропийного сопла. Ниже этого отношения расход идеального газа не растёт."""
    if gamma <= 1.0:
        raise DomainError("Показатель адиабаты критического отношения должен быть больше единицы")
    return (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))


def critical_mass_flux_ideal_gas(
    pressure_pa: float, temperature_k: float, gamma: float, gas_constant_j_kg_k: float
) -> float:
    """Критический массовый поток изоэнтропийного сопла, кг/(м²·с)."""
    if gamma <= 1.0 or pressure_pa <= 0.0 or temperature_k <= 0.0:
        raise DomainError("Критическое истечение идеального газа вне области формулы")
    factor = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return pressure_pa * math.sqrt(gamma / (gas_constant_j_kg_k * temperature_k)) * factor


def choked_mass_flow(
    upstream_pressure_pa: float,
    downstream_pressure_pa: float,
    critical_pressure_ratio: float,
    critical_flow_kg_s: float,
    area_coefficient: float,
) -> float:
    """Ниже критического отношения расход не растёт при дальнейшем снижении противодавления."""
    if upstream_pressure_pa <= 0.0:
        raise DomainError("Давление перед горлом должно быть положительным")
    ratio = downstream_pressure_pa / upstream_pressure_pa
    if ratio <= critical_pressure_ratio:
        return critical_flow_kg_s
    unchoked = area_coefficient * math.sqrt(max(upstream_pressure_pa - downstream_pressure_pa, 0.0))
    return min(unchoked, critical_flow_kg_s)


def four_quadrant_head_pa(
    speed_ratio: float,
    flow_ratio: float,
    a_speed: float,
    a_cross: float,
    a_flow: float,
    head_scale_pa: float,
) -> float:
    """Напор во всех квадрантах. speed_ratio и flow_ratio нормированы на номинал."""
    group = (
        a_speed * speed_ratio * abs(speed_ratio)
        + a_cross * speed_ratio * flow_ratio
        + a_flow * flow_ratio * abs(flow_ratio)
    )
    return head_scale_pa * group


def pump_shaft_power_w(mass_flow_kg_s: float, head_pa: float, density_kg_m3: float) -> float:
    if density_kg_m3 <= 0.0:
        raise DomainError("Плотность насоса должна быть положительной")
    return mass_flow_kg_s * head_pa / density_kg_m3
