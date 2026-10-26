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


def pump_acceleration(
    omega_rad_s: float, drive_nm: float, hydraulic_nm: float, loss_nm: float, inertia_kg_m2: float
) -> float:
    if inertia_kg_m2 <= 0.0:
        raise DomainError("Момент инерции насоса должен быть положительным")
    if omega_rad_s < 0.0:
        raise DomainError("Отрицательная скорость насоса вне проверенного квадранта")
    return (drive_nm - hydraulic_nm - loss_nm) / inertia_kg_m2


def coastdown_omega(omega0: float, torque_nm: float, inertia: float, time_s: float) -> float:
    """Точное решение при постоянном тормозящем моменте и нулевом приводе."""
    if torque_nm < 0.0:
        raise DomainError("Тормозящий момент задаётся положительным")
    return omega0 - torque_nm / inertia * time_s
