"""Стенка теплообменника с собственной теплоёмкостью.

Первичный контур отдаёт UA_p (T_p - T_w), вторичный получает UA_s (T_w - T_s).
В стационаре эти потоки равны. Петля ВВЭР держит такую стенку между сторонами
парогенератора: два коэффициента в ряду, теплоёмкость — параметр модели.
"""

from __future__ import annotations

import math

from core.status import DomainError, ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="heat-exchanger-wall",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Теплоёмкость стенки между двумя контурами",
    approximations=(
        "температуры теплоносителей заданы",
        "UA постоянны",
        "продольная проводимость стенки не учитывается",
    ),
    applicability="C > 0, UA > 0",
    data_sources=("Баланс энергии стенки C dT/dt = Q_primary - Q_secondary",),
)


def log_mean_temperature_difference(delta_a_k: float, delta_b_k: float) -> float:
    """LMTD. Равные положительные сближения дают это сближение; нулевое сближение не логарифмируется."""
    if delta_a_k <= 0.0 or delta_b_k <= 0.0:
        raise DomainError("LMTD определён при положительных сближениях температур")
    scale = max(delta_a_k, delta_b_k)
    if abs(delta_a_k - delta_b_k) <= 1.0e-12 * scale:
        return 0.5 * (delta_a_k + delta_b_k)
    return (delta_a_k - delta_b_k) / math.log(delta_a_k / delta_b_k)


def lmtd_heat_w(ua_w_k: float, delta_a_k: float, delta_b_k: float) -> float:
    if ua_w_k <= 0.0:
        raise DomainError("UA для LMTD должен быть положительным")
    return ua_w_k * log_mean_temperature_difference(delta_a_k, delta_b_k)


def equal_capacity_counterflow(c_w_k: float, inlet_difference_k: float, ntu: float) -> tuple:
    """Cr = 1: тепло C ΔT NTU/(1+NTU) и LMTD = ΔT/(1+NTU)."""
    if c_w_k <= 0.0 or inlet_difference_k < 0.0 or ntu < 0.0:
        raise DomainError("Равные теплоёмкости противотока заданы вне области")
    heat = counterflow_heat_w(c_w_k, inlet_difference_k, 0.0, ntu, 1.0)
    lmtd = inlet_difference_k / (1.0 + ntu)
    return heat, lmtd


def counterflow_asymptote(capacity_ratio: float) -> float:
    """Предел эффективности противотока при NTU → ∞. Для Cr в [0, 1] он равен единице."""
    if capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("Отношение теплоёмкостей предела противотока лежит в [0, 1]")
    return 1.0


def crossflow_cmin_unmixed_effectiveness(ntu: float, capacity_ratio: float) -> float:
    """Поперечный ток: Cmin не перемешан, Cmax перемешан.

    ε = (1/Cr) (1 - exp(-Cr (1 - exp(-NTU)))). При Cr=0 это 1-exp(-NTU).
    """
    if ntu < 0.0 or capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("NTU неотрицателен, отношение теплоёмкостей лежит в [0, 1]")
    if ntu == 0.0:
        return 0.0
    if capacity_ratio == 0.0:
        return 1.0 - math.exp(-ntu)
    return (1.0 / capacity_ratio) * (1.0 - math.exp(-capacity_ratio * (1.0 - math.exp(-ntu))))


def crossflow_cmin_unmixed_asymptote(capacity_ratio: float) -> float:
    """Предел поперечного тока при NTU → ∞: (1 - exp(-Cr)) / Cr, а при Cr=0 единица."""
    if capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("Отношение теплоёмкостей предела поперечного тока лежит в [0, 1]")
    if capacity_ratio == 0.0:
        return 1.0
    return (1.0 - math.exp(-capacity_ratio)) / capacity_ratio


def parallelflow_asymptote(capacity_ratio: float) -> float:
    """Предел прямотока при NTU → ∞: 1/(1+Cr)."""
    if capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("Отношение теплоёмкостей предела прямотока лежит в [0, 1]")
    return 1.0 / (1.0 + capacity_ratio)


def parallelflow_effectiveness(ntu: float, capacity_ratio: float) -> float:
    """Эффективность прямотока (1 - exp(-NTU(1+Cr))) / (1+Cr). При Cr=0 совпадает с противотоком."""
    if ntu < 0.0 or capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("NTU неотрицателен, отношение теплоёмкостей лежит в [0, 1]")
    if ntu == 0.0:
        return 0.0
    return (1.0 - math.exp(-ntu * (1.0 + capacity_ratio))) / (1.0 + capacity_ratio)


def counterflow_effectiveness(ntu: float, capacity_ratio: float) -> float:
    """Эффективность противотока. При равных теплоёмкостях NTU/(1+NTU), при Cr=0 это 1-exp(-NTU)."""
    if ntu < 0.0 or capacity_ratio < 0.0 or capacity_ratio > 1.0:
        raise DomainError("NTU неотрицателен, отношение теплоёмкостей лежит в [0, 1]")
    if ntu == 0.0:
        return 0.0
    if abs(capacity_ratio - 1.0) <= 1.0e-12:
        return ntu / (1.0 + ntu)
    decay = math.exp(-ntu * (1.0 - capacity_ratio))
    return (1.0 - decay) / (1.0 - capacity_ratio * decay)


def counterflow_heat_w(
    c_min_w_k: float, t_hot_in_k: float, t_cold_in_k: float, ntu: float, capacity_ratio: float
) -> float:
    if c_min_w_k <= 0.0 or t_hot_in_k < t_cold_in_k:
        raise DomainError("Противоток задан при неположительной теплоёмкости или обратном перепаде")
    return counterflow_effectiveness(ntu, capacity_ratio) * c_min_w_k * (t_hot_in_k - t_cold_in_k)


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

    def power_split_w(self, primary_temperature_k: float, secondary_temperature_k: float) -> tuple:
        """Мощности первичной стороны, стенки и вторичной стороны. Их сумма равна нулю."""
        to_wall, to_secondary = self.heat_flows_w(primary_temperature_k, secondary_temperature_k)
        return -to_wall, to_wall - to_secondary, to_secondary

    def derivative_k_s(self, primary_temperature_k: float, secondary_temperature_k: float) -> float:
        to_wall, to_secondary = self.heat_flows_w(primary_temperature_k, secondary_temperature_k)
        return (to_wall - to_secondary) / self.capacity_j_k

    def equilibrium_temperature_k(
        self, primary_temperature_k: float, secondary_temperature_k: float
    ) -> float:
        weight = self.ua_primary_w_k + self.ua_secondary_w_k
        return (
            self.ua_primary_w_k * primary_temperature_k
            + self.ua_secondary_w_k * secondary_temperature_k
        ) / weight

    def advance(
        self, primary_temperature_k: float, secondary_temperature_k: float, dt_s: float
    ) -> float:
        """Точное решение при постоянных температурах теплоносителей."""
        if dt_s < 0.0:
            raise DomainError("Шаг стенки отрицателен")
        equilibrium = self.equilibrium_temperature_k(primary_temperature_k, secondary_temperature_k)
        rate = (self.ua_primary_w_k + self.ua_secondary_w_k) / self.capacity_j_k
        self.temperature_k = equilibrium + (self.temperature_k - equilibrium) * math.exp(
            -rate * dt_s
        )
        return self.temperature_k
