"""Замыкания оборудования, которые проверяются отдельно от сборки станции."""

from __future__ import annotations

import math

from core.status import DomainError, ModelPassport, ModelStatus

ORIFICE = ModelPassport(
    model_id="incompressible-orifice",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Расход несжимаемого отверстия",
    approximations=("плотность донора постоянна", "критическое истечение не входит в эту формулу"),
    applicability="Однофазная жидкость, перепад давления много меньше объёмного модуля",
    data_sources=("Стандартное уравнение отверстия m = Cd A sqrt(2 rho |dp|)",),
)

PUMP = ModelPassport(
    model_id="pump-inertia",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Инерция ротора насоса",
    approximations=(
        "один момент сопротивления, пропорциональный скорости",
        "гомологическая четырёхквадрантная характеристика не используется",
    ),
    applicability="Положительная скорость и расход вперёд",
    data_sources=("Уравнение J dw/dt = M_drive - M_hyd - M_loss",),
)


def valve_mass_flow(
    phase: str,
    area_m2: float,
    upstream_pressure_pa: float,
    downstream_pressure_pa: float,
    density_kg_m3: float = 0.0,
    quality: float = -1.0,
    temperature_k: float = 0.0,
    gamma: float = 1.3,
    gas_constant_j_kg_k: float = 461.5,
    discharge_coefficient: float = 0.62,
) -> float:
    """Клапан выбирает одну модель по фазе и не подменяет её соседней.

    Жидкость — несжимаемое отверстие. Смесь — замороженный критический поток.
    Пар — изоэнтропийное сопло идеального газа. Обратный перепад для смеси и пара
    в эту функцию не входит.
    """
    if area_m2 <= 0.0 or upstream_pressure_pa <= 0.0:
        raise DomainError("Площадь клапана и давление перед ним должны быть положительными")
    if phase == "liquid":
        if quality not in (-1.0, 0.0):
            raise DomainError("Жидкостный клапан не принимает паросодержание смеси")
        return orifice_mass_flow(
            discharge_coefficient,
            area_m2,
            density_kg_m3,
            upstream_pressure_pa - downstream_pressure_pa,
        )
    if downstream_pressure_pa < 0.0:
        raise DomainError("Противодавление клапана отрицательно")
    if phase == "mixture":
        from physics.hydraulics.two_phase_ext import frozen_mixture_critical_flux

        return frozen_mixture_critical_flux(upstream_pressure_pa, quality) * area_m2
    if phase == "vapor":
        from physics.hydraulics.two_phase_ext import critical_mass_flux_ideal_gas

        if quality not in (-1.0, 1.0):
            raise DomainError("Паровой клапан не принимает недогретую смесь")
        return (
            critical_mass_flux_ideal_gas(
                upstream_pressure_pa, temperature_k, gamma, gas_constant_j_kg_k
            )
            * area_m2
        )
    raise DomainError("Фаза клапана %s не поддержана" % phase)


def check_valve_mass_flow(
    discharge_coefficient: float,
    area_m2: float,
    density: float,
    upstream_pressure_pa: float,
    downstream_pressure_pa: float,
    cracking_pressure_pa: float = 0.0,
) -> float:
    """Обратный клапан: расход только вперёд. Реверс — ноль, а не отрицательное отверстие."""
    pressure_drop = upstream_pressure_pa - downstream_pressure_pa
    if pressure_drop <= cracking_pressure_pa:
        return 0.0
    return orifice_mass_flow(discharge_coefficient, area_m2, density, pressure_drop)


def orifice_mass_flow(
    discharge_coefficient: float, area_m2: float, density: float, pressure_drop_pa: float
) -> float:
    if discharge_coefficient <= 0.0 or area_m2 <= 0.0 or density <= 0.0:
        raise DomainError("Коэффициент, площадь и плотность отверстия должны быть положительными")
    direction = 1.0 if pressure_drop_pa >= 0.0 else -1.0
    return (
        direction
        * discharge_coefficient
        * area_m2
        * math.sqrt(2.0 * density * abs(pressure_drop_pa))
    )


def affinity_ratios(speed_ratio: float) -> tuple:
    """Подобие несжимаемого насоса: Q~n, H~n², P~n³. Отрицательная скорость — другой квадрант."""
    if speed_ratio < 0.0:
        raise DomainError("Законы подобия заданы для неотрицательного отношения скоростей")
    return speed_ratio, speed_ratio**2, speed_ratio**3


def parallel_orifice_flow(branches, density: float, pressure_drop_pa: float) -> float:
    """Параллельные отверстия при общем перепаде. Расходы складываются."""
    if not branches:
        raise DomainError("Параллельная сборка отверстий пуста")
    total = 0.0
    for coefficient, area in branches:
        total += orifice_mass_flow(coefficient, area, density, pressure_drop_pa)
    return total


def series_orifice_drop_pa(mass_flow_kg_s: float, branches, density: float) -> float:
    """Последовательные отверстия одного расхода. Перепады складываются."""
    pairs = list(branches)
    if not pairs:
        raise DomainError("Последовательная сборка отверстий пуста")
    return float(
        sum(
            orifice_pressure_drop(mass_flow_kg_s, coefficient, area, density)
            for coefficient, area in pairs
        )
    )


def equivalent_series_orifice_area_m2(areas_m2) -> float:
    """При общем Cd: 1/A² = сумма 1/A_i²."""
    areas = list(areas_m2)
    if not areas or any(area <= 0.0 for area in areas):
        raise DomainError("Площади последовательных отверстий должны быть положительными")
    return 1.0 / math.sqrt(sum(1.0 / area**2 for area in areas))


def orifice_pressure_drop(
    mass_flow_kg_s: float, discharge_coefficient: float, area_m2: float, density: float
) -> float:
    """Перепад, который даёт заданный расход отверстия. Нулевой расход даёт нулевой перепад."""
    if min(discharge_coefficient, area_m2, density) <= 0.0:
        raise DomainError("Коэффициент, площадь и плотность отверстия должны быть положительными")
    if mass_flow_kg_s == 0.0:
        return 0.0
    scale = (discharge_coefficient * area_m2) ** 2 * 2.0 * density
    return math.copysign(mass_flow_kg_s**2 / scale, mass_flow_kg_s)


def quadratic_shutoff_head(flow_kg_s: float, flow_max_kg_s: float, head_shutoff_pa: float) -> float:
    """H = H0 (1 - (q/qmax)^2) до номинального расхода. Дальше характеристика не продолжается."""
    if flow_max_kg_s <= 0.0 or head_shutoff_pa < 0.0:
        raise DomainError("Номинальный расход и напор нулевой подачи заданы неверно")
    if abs(flow_kg_s) > flow_max_kg_s:
        raise DomainError("Расход вне квадратичной характеристики насоса")
    ratio = flow_kg_s / flow_max_kg_s
    return head_shutoff_pa * (1.0 - ratio * ratio)


def flow_from_quadratic_head(head_pa: float, flow_max_kg_s: float, head_shutoff_pa: float) -> float:
    if flow_max_kg_s <= 0.0 or head_shutoff_pa <= 0.0:
        raise DomainError("Номинальный расход и напор нулевой подачи должны быть положительными")
    if head_pa < 0.0 or head_pa > head_shutoff_pa:
        raise DomainError("Напор вне квадратичной характеристики насоса")
    return flow_max_kg_s * math.sqrt(1.0 - head_pa / head_shutoff_pa)


def orifice_flow_derivative(
    discharge_coefficient: float, area_m2: float, density: float, pressure_drop_pa: float
) -> float:
    """dm/dp несжимаемого отверстия. На нулевом перепаде производная не определена."""
    if pressure_drop_pa == 0.0:
        raise DomainError("Производная отверстия на нулевом перепаде не определена")
    if discharge_coefficient <= 0.0 or area_m2 <= 0.0 or density <= 0.0:
        raise DomainError("Коэффициент, площадь и плотность отверстия должны быть положительными")
    return 0.5 * discharge_coefficient * area_m2 * math.sqrt(2.0 * density / abs(pressure_drop_pa))


def hydraulic_power_w(mass_flow_kg_s: float, head_pa: float, density_kg_m3: float) -> float:
    """Мощность, передаваемая жидкости. В прямом квадранте она равна моменту на валу, умноженному на скорость."""
    if density_kg_m3 <= 0.0:
        raise DomainError("Плотность насоса должна быть положительной")
    return mass_flow_kg_s * head_pa / density_kg_m3


def torque_matching_hydraulic_power_nm(
    mass_flow_kg_s: float, head_pa: float, density_kg_m3: float, omega_rad_s: float
) -> float:
    if omega_rad_s <= 0.0:
        raise DomainError("Согласование мощности прямого квадранта требует положительную скорость")
    return hydraulic_power_w(mass_flow_kg_s, head_pa, density_kg_m3) / omega_rad_s


def pump_acceleration(
    omega_rad_s: float, drive_nm: float, hydraulic_nm: float, loss_nm: float, inertia_kg_m2: float
) -> float:
    if inertia_kg_m2 <= 0.0:
        raise DomainError("Момент инерции насоса должен быть положительным")
    if omega_rad_s < 0.0:
        raise DomainError("Отрицательная скорость насоса вне проверенного квадранта")
    return (drive_nm - hydraulic_nm - loss_nm) / inertia_kg_m2


def coastdown_dissipated_j(omega0: float, torque_nm: float, inertia: float, time_s: float) -> float:
    """Работа постоянного тормозящего момента. К остановке она равна кинетической энергии ½ J ω0²."""
    if omega0 < 0.0 or time_s < 0.0 or inertia <= 0.0:
        raise DomainError("Работа выбега задана при отрицательной скорости, времени или инерции")
    if torque_nm < 0.0:
        raise DomainError("Тормозящий момент задаётся положительным")
    if torque_nm == 0.0 or omega0 == 0.0:
        return 0.0
    elapsed = min(time_s, time_to_rest_s(omega0, torque_nm, inertia))
    return torque_nm * (omega0 * elapsed - torque_nm * elapsed**2 / (2.0 * inertia))


def time_to_rest_s(omega0: float, torque_nm: float, inertia: float) -> float:
    """Время остановки при постоянном тормозящем моменте: ω0 J / M."""
    if omega0 < 0.0 or torque_nm <= 0.0 or inertia <= 0.0:
        raise DomainError(
            "Время остановки задано при отрицательной скорости или неположительном моменте"
        )
    return omega0 * inertia / torque_nm


def rested_coastdown(omega0: float, torque_nm: float, inertia: float, time_s: float) -> float:
    """Выбег при постоянном тормозящем моменте. После остановки скорость остаётся нулём."""
    if omega0 < 0.0 or time_s < 0.0 or inertia <= 0.0:
        raise DomainError("Выбег задан при отрицательной скорости, времени или инерции")
    if torque_nm < 0.0:
        raise DomainError("Тормозящий момент задаётся положительным")
    if torque_nm == 0.0:
        return float(omega0)
    stop_s = omega0 * inertia / torque_nm
    if time_s >= stop_s:
        return 0.0
    return coastdown_omega(omega0, torque_nm, inertia, time_s)


def coastdown_omega(omega0: float, torque_nm: float, inertia: float, time_s: float) -> float:
    """Точное решение при постоянном тормозящем моменте и нулевом приводе."""
    if torque_nm < 0.0:
        raise DomainError("Тормозящий момент задаётся положительным")
    return omega0 - torque_nm / inertia * time_s
