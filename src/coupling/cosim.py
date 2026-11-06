"""Обмен двух линейных блоков с удержанием значения на шаге.

Это численный образец косимуляции, не экспорт FMI и не модель установки.
"""

from __future__ import annotations

import math

from core.status import DomainError


def hold_linear_step(rate_per_s: float, hold: float, value: float, dt_s: float) -> float:
    """Точное решение y' = -rate y + hold на одном шаге обмена."""
    if rate_per_s <= 0.0 or dt_s < 0.0:
        raise DomainError("Скорость и шаг косимуляции заданы неверно")
    equilibrium = hold / rate_per_s
    return equilibrium + (value - equilibrium) * math.exp(-rate_per_s * dt_s)


def zero_order_hold_pair(y0: float, z0: float, time_s: float, step_s: float) -> tuple:
    """Пара y' = -2y + z, z' = -3z + y. Сосед на шаге заморожен, обмен по Якоби.

    Горизонт должен быть кратен шагу обмена.
    """
    if step_s <= 0.0 or time_s < 0.0:
        raise DomainError("Шаг обмена косимуляции должен быть положительным")
    steps = int(round(time_s / step_s))
    if abs(steps * step_s - time_s) > 1.0e-9:
        raise DomainError("Горизонт косимуляции не кратен шагу обмена")
    ordinate = float(y0)
    partner = float(z0)
    for _ in range(steps):
        next_ordinate = hold_linear_step(2.0, partner, ordinate, step_s)
        next_partner = hold_linear_step(3.0, ordinate, partner, step_s)
        ordinate, partner = next_ordinate, next_partner
    return ordinate, partner


def hold_pair_crossing(y0: float, z0: float, level: float, step_s: float, limit_s: float):
    """Время, когда y' = -2y + z пересекает уровень. Внутри шага обмена корень точный для замороженного соседа."""
    if step_s <= 0.0 or limit_s < 0.0:
        raise DomainError("Шаг и горизонт поиска пересечения заданы неверно")
    clock = 0.0
    ordinate = float(y0)
    partner = float(z0)
    while clock < limit_s - 1.0e-15:
        span = min(step_s, limit_s - clock)
        equilibrium = partner / 2.0
        if ordinate != equilibrium:
            ratio = (level - equilibrium) / (ordinate - equilibrium)
            if 0.0 < ratio <= 1.0:
                tau = -math.log(ratio) / 2.0
                if 0.0 < tau <= span + 1.0e-12:
                    return clock + min(tau, span)
        next_ordinate = hold_linear_step(2.0, partner, ordinate, span)
        next_partner = hold_linear_step(3.0, ordinate, partner, span)
        ordinate, partner = next_ordinate, next_partner
        clock += span
    return None
