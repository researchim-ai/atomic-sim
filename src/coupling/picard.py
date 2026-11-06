"""Итерация мощности и температуры. Фиксированная точка сверяется с корнем уравнения."""

from __future__ import annotations

from core.status import DomainError


def picard_power_temperature(
    power_scale_w: float,
    sink_k: float,
    conductance_w_k: float,
    feedback_per_k: float,
    temperature_ref_k: float,
    tol_k: float = 1.0e-10,
) -> tuple:
    """P = P0 / (1 + a (T - Tref)), T = T_sink + P / UA."""
    if power_scale_w <= 0.0 or conductance_w_k <= 0.0 or feedback_per_k < 0.0:
        raise DomainError("Масштаб мощности и UA положительны, обратная связь неотрицательна")
    temperature = temperature_ref_k
    power = power_scale_w
    for _ in range(40):
        power = power_scale_w / (1.0 + feedback_per_k * (temperature - temperature_ref_k))
        updated = sink_k + power / conductance_w_k
        if abs(updated - temperature) <= tol_k:
            return updated, power, True
        temperature = updated
    return temperature, power, False
