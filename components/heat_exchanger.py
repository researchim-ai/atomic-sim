"""Стенка теплообменника с собственной теплоёмкостью.

Первичный контур отдаёт UA_p (T_p - T_w), вторичный получает UA_s (T_w - T_s).
В стационаре эти потоки равны. ВВЭР эту стенку пока не подключает.
"""

from __future__ import annotations

import math

from core.status import DomainError, ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="heat-exchanger-wall",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Теплоёмкость стенки между двумя контурами",
    approximations=("температуры теплоносителей заданы", "UA постоянны", "продольная проводимость стенки не учитывается"),
    applicability="C > 0, UA > 0",
    data_sources=("Баланс энергии стенки C dT/dt = Q_primary - Q_secondary",),
)


class HeatExchangerWall:
    passport = PASSPORT

    def __init__(
        self,
        capacity_j_k: float,
        ua_primary_w_k: float,
        ua_secondary_w_k: float,
        temperature_k: float,
    ):
        if capacity_j_k <= 0.0 or ua_primary_w_k <= 0.0 or ua_secondary_w_k <= 0.0:
            raise DomainError("Теплоёмкость и оба UA стенки должны быть положительными")
        self.capacity_j_k = float(capacity_j_k)
        self.ua_primary_w_k = float(ua_primary_w_k)
        self.ua_secondary_w_k = float(ua_secondary_w_k)
        self.temperature_k = float(temperature_k)

    def heat_flows_w(self, primary_temperature_k: float, secondary_temperature_k: float) -> tuple:
        to_wall = self.ua_primary_w_k * (primary_temperature_k - self.temperature_k)
        to_secondary = self.ua_secondary_w_k * (self.temperature_k - secondary_temperature_k)
        return to_wall, to_secondary

    def derivative_k_s(self, primary_temperature_k: float, secondary_temperature_k: float) -> float:
        to_wall, to_secondary = self.heat_flows_w(primary_temperature_k, secondary_temperature_k)
        return (to_wall - to_secondary) / self.capacity_j_k

    def equilibrium_temperature_k(self, primary_temperature_k: float, secondary_temperature_k: float) -> float:
        weight = self.ua_primary_w_k + self.ua_secondary_w_k
        return (
            self.ua_primary_w_k * primary_temperature_k + self.ua_secondary_w_k * secondary_temperature_k
        ) / weight

    def advance(self, primary_temperature_k: float, secondary_temperature_k: float, dt_s: float) -> float:
        """Точное решение при постоянных температурах теплоносителей."""
        if dt_s < 0.0:
            raise DomainError("Шаг стенки отрицателен")
        equilibrium = self.equilibrium_temperature_k(primary_temperature_k, secondary_temperature_k)
        rate = (self.ua_primary_w_k + self.ua_secondary_w_k) / self.capacity_j_k
        self.temperature_k = equilibrium + (self.temperature_k - equilibrium) * math.exp(-rate * dt_s)
        return self.temperature_k
