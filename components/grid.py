"""Механика генератора: электрическая мощность следует из момента, а не из давления пара."""

from __future__ import annotations

from core.status import ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="generator-swing",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Уравнение движения ротора генератора",
    approximations=("одна машина", "постоянная нагрузка на шаге", "нет сети из многих машин"),
    applicability="Скорость остаётся положительной",
    data_sources=(
        "2H/omega0 * domega/dt = P_mech - P_elec в записи через момент инерции J dw/dt = M_mech - M_elec",
    ),
)


def electrical_power_w(torque_nm: float, omega_rad_s: float) -> float:
    return torque_nm * omega_rad_s


def speed_derivative(
    inertia_kg_m2: float, mechanical_torque_nm: float, electrical_torque_nm: float
) -> float:
    return (mechanical_torque_nm - electrical_torque_nm) / inertia_kg_m2


def shaft_work_j(
    omega0: float, inertia: float, mechanical_torque: float, electrical_torque: float, time_s: float
) -> float:
    """Работа моментов на валу при постоянных моментах. Она равна изменению ½ J ω²."""
    if inertia <= 0.0 or time_s < 0.0:
        raise ValueError("Инерция вала положительна, время неотрицательно")
    torque = mechanical_torque - electrical_torque
    return torque * (omega0 * time_s + 0.5 * torque / inertia * time_s**2)


def speed_after_constant_torques(
    omega0: float, inertia: float, mechanical_torque: float, electrical_torque: float, time_s: float
) -> float:
    return omega0 + speed_derivative(inertia, mechanical_torque, electrical_torque) * time_s


def grid_electrical_torque_nm(
    connected: bool, mechanical_power_w: float, omega_rad_s: float
) -> float:
    """При отключённой сети электрический момент равен нулю и не подменяется механическим."""
    if not connected:
        return 0.0
    return synchronous_balance(mechanical_power_w, omega_rad_s)


def synchronous_balance(mechanical_power_w: float, omega_rad_s: float) -> float:
    """Момент, при котором электрическая мощность равна механической."""
    if omega_rad_s == 0.0:
        raise ValueError("Нулевая скорость не задаёт момент")
    return mechanical_power_w / omega_rad_s
