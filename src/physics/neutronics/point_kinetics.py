"""Точечная кинетика с шестью группами запаздывающих нейтронов.

dn/dt = ((rho - beta)/Lambda) * n + sum(lambda_i * C_i) + S
dC_i/dt = (beta_i/Lambda) * n - lambda_i * C_i

`n` — амплитуда, нормированная так, что мощность = n * power_nominal.
`C_i` имеет ту же нормировку: в стационаре C_i = beta_i / (Lambda * lambda_i) * n.
Реактивность абсолютная (dk/k), не в долларах. Отрицательные предшественники не обрезаются.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from scipy.linalg import expm

from core.solvers import integrate
from core.status import DomainError, ModelPassport, ModelStatus

# Доли и постоянные распада, типичный тепловой спектр U-235 (Keepin-подобные
# групповые данные, те же числа, что в учебной ветке). Это не beta_eff конкретного ВВЭР.
BETA_U235 = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])
LAMBDA_U235 = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])

PASSPORT = ModelPassport(
    model_id="point-kinetics-6g",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Точечная кинетика с шестью группами предшественников",
    approximations=(
        "пространственно независимая амплитуда",
        "постоянные beta_i, lambda_i и Lambda на шаге",
        "источник S аддитивен в уравнении амплитуды",
    ),
    applicability="|rho| < beta и умеренные скачки, где шестигрупповая модель достаточна; prompt-supercritical считается, но не интерпретируется как авария с разрушением",
    data_sources=("Групповые beta_i и lambda_i теплового деления U-235, учебный набор проекта",),
    notes=(
        "Статус verified — согласие с матричной экспонентой и периодом inhour, не с данными блока.",
    ),
)

Rho = Union[float, Callable[[float], float]]


@dataclass
class PointKinetics:
    beta: np.ndarray
    decay: np.ndarray
    generation_time_s: float
    power_nominal_w: float = 1.0
    source: float = 0.0

    def __post_init__(self) -> None:
        self.beta = np.asarray(self.beta, dtype=float)
        self.decay = np.asarray(self.decay, dtype=float)
        self.beta_total = float(np.sum(self.beta))

    @classmethod
    def thermal_u235(
        cls, generation_time_s: float = 5.0e-5, power_nominal_w: float = 1.0, source: float = 0.0
    ) -> "PointKinetics":
        return cls(BETA_U235.copy(), LAMBDA_U235.copy(), generation_time_s, power_nominal_w, source)

    def equilibrium_with_source(self, reactivity: float, source: float) -> np.ndarray:
        """Стационар n = -S Λ / ρ при ρ < 0. Предшественники согласованы с этой амплитудой."""
        if reactivity >= 0.0:
            raise DomainError("Стационар с источником есть только при отрицательной реактивности")
        if source < 0.0:
            raise DomainError("Внешний источник стационара отрицателен")
        amplitude = -source * self.generation_time_s / reactivity
        return np.concatenate([[amplitude], self.equilibrium_precursors(amplitude)])

    def decayed_precursors(self, precursors: np.ndarray, dt_s: float) -> np.ndarray:
        """Свободный распад при нулевой амплитуде: C exp(-λ t)."""
        if dt_s < 0.0:
            raise DomainError("Время распада предшественников отрицательно")
        concentration = np.asarray(precursors, dtype=float)
        if concentration.shape != self.decay.shape:
            raise DomainError("Число групп предшественников не совпадает с моделью")
        return concentration * np.exp(-self.decay * dt_s)

    def equilibrium_precursors(self, amplitude: float) -> np.ndarray:
        return self.beta / (self.generation_time_s * self.decay) * amplitude

    def initial_state(self, amplitude: float = 1.0) -> np.ndarray:
        return np.concatenate([[amplitude], self.equilibrium_precursors(amplitude)])

    def amplitude_reactivity_slope(self, amplitude: float) -> float:
        """Частная производная dn/dt по реактивности при фиксированных предшественниках: n/Λ."""
        if self.generation_time_s <= 0.0:
            raise DomainError("Время генерации для наклона реактивности должно быть положительным")
        return amplitude / self.generation_time_s

    def inverse_period(
        self, amplitude: float, precursors: np.ndarray, reactivity: float, source: float = 0.0
    ) -> float:
        """(1/n) dn/dt. На равновесных предшественниках сразу после скачка ρ это ρ/Λ."""
        if amplitude == 0.0:
            raise DomainError("Обратный период не определён при нулевой амплитуде")
        return float(self.derivatives(amplitude, precursors, reactivity, source)[0] / amplitude)

    def derivatives(
        self,
        amplitude: float,
        precursors: np.ndarray,
        reactivity: float,
        source: float | None = None,
    ) -> np.ndarray:
        emitted = 0.0 if source is None else source
        dn = (
            ((reactivity - self.beta_total) / self.generation_time_s) * amplitude
            + float(np.dot(self.decay, precursors))
            + emitted
        )
        dC = self.beta / self.generation_time_s * amplitude - self.decay * precursors
        return np.concatenate([[dn], dC])

    def matrix(self, reactivity: float) -> np.ndarray:
        groups = self.beta.size
        matrix = np.zeros((groups + 1, groups + 1))
        matrix[0, 0] = (reactivity - self.beta_total) / self.generation_time_s
        matrix[0, 1:] = self.decay
        matrix[1:, 0] = self.beta / self.generation_time_s
        matrix[1:, 1:] = np.diag(-self.decay)
        return matrix

    def visible_amplitude_modes(self, reactivity: float, tolerance: float = 1.0e-8) -> int:
        """Число собственных мод, которые двигают амплитуду.

        Степени матрицы наблюдаемости в float64 теряют младшие моды этой жёсткой системы.
        Видимость проверяется по компоненте амплитуды нормированной собственной моды.
        """
        if tolerance < 0.0:
            raise DomainError("Допуск видимости моды отрицателен")
        _values, vectors = np.linalg.eig(self.matrix(reactivity))
        visible = 0
        for index in range(vectors.shape[1]):
            vector = vectors[:, index]
            if abs(vector[0]) > tolerance * float(np.linalg.norm(vector)):
                visible += 1
        if visible < vectors.shape[1]:
            raise DomainError("Мода не видна по амплитуде: %s из %s" % (visible, vectors.shape[1]))
        return visible

    def reference_step(
        self, state: np.ndarray, reactivity: float, dt_s: float, source: float = 0.0
    ) -> np.ndarray:
        """Точное решение при постоянных rho и S: y' = A y + b."""
        matrix = self.matrix(reactivity)
        if source == 0.0:
            return expm(matrix * dt_s) @ state
        inhom = np.zeros(state.size)
        inhom[0] = source
        # y(t) = exp(At) y0 + A^{-1} (exp(At) - I) b
        expA = expm(matrix * dt_s)
        particular = np.linalg.solve(matrix, (expA - np.eye(state.size)) @ inhom)
        return expA @ state + particular

    def integrate(
        self,
        state: np.ndarray,
        reactivity: Rho,
        t_end: float,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        source: float | None = None,
    ) -> np.ndarray:
        emitted = self.source if source is None else source

        def rhs(time: float, y: np.ndarray) -> np.ndarray:
            rho = reactivity(time) if callable(reactivity) else float(reactivity)
            return self.derivatives(y[0], y[1:], rho, emitted)

        result = integrate(rhs, state, (0.0, t_end), rtol=rtol, atol=atol, method="BDF")
        final = result.final
        if np.any(final[1:] < -1e-8 * max(1.0, float(np.max(np.abs(state))))):
            # Диагностика, без подмены решения.
            raise ValueError(
                "Интегратор получил отрицательных предшественников; решение не обрезается"
            )
        return final

    def power_w(self, amplitude: float) -> float:
        return amplitude * self.power_nominal_w


def reactivity_in_dollars(reactivity: float, beta: float) -> float:
    """Реактивность в долларах: ρ / β."""
    if beta <= 0.0:
        raise DomainError("Доля запаздывающих нейтронов для доллара должна быть положительной")
    return reactivity / beta


def one_group_inhour_root(
    reactivity: float, generation_time_s: float, beta: float, decay_per_s: float
) -> float:
    """Корень одногруппового inhour из квадратного уравнения.

    Λ ω² + (Λ λ + β − ρ) ω − ρ λ = 0. При ρ > 0 берётся положительный корень,
    при ρ < 0 — ближайший к нулю.
    """
    if generation_time_s <= 0.0 or beta <= 0.0 or decay_per_s <= 0.0:
        raise DomainError("Одногрупповой inhour задан при неположительных Λ, β или λ")
    if reactivity == 0.0:
        return 0.0
    linear = generation_time_s * decay_per_s + beta - reactivity
    discriminant = linear**2 + 4.0 * generation_time_s * reactivity * decay_per_s
    if discriminant < 0.0:
        raise DomainError("Дискриминант одногруппового inhour отрицателен")
    root = math.sqrt(discriminant)
    positive = (-linear + root) / (2.0 * generation_time_s)
    negative = (-linear - root) / (2.0 * generation_time_s)
    if reactivity > 0.0:
        return positive
    return max(positive, negative)


def prompt_jump_ratio(beta: float, reactivity: float) -> float:
    """Амплитуда сразу после скачка, пока предшественники ещё старые: β / (β − ρ).

    Формула задана при ρ < β. На мгновенных нейтронах скачок так не записывается.
    """
    if beta <= 0.0:
        raise DomainError("Доля запаздывающих нейтронов для скачка должна быть положительной")
    if reactivity >= beta:
        raise DomainError("Скачок на мгновенных нейтронах этой формулой не задаётся")
    return beta / (beta - reactivity)


def inhour_root(kinetics: PointKinetics, reactivity: float) -> float:
    """Положительный или доминирующий корень уравнения inhour для ступенчатой реактивности."""
    beta = kinetics.beta
    decay = kinetics.decay
    generation = kinetics.generation_time_s

    def f(omega: float) -> float:
        return reactivity - omega * generation - float(np.sum(omega * beta / (omega + decay)))

    if reactivity == 0.0:
        return 0.0
    if reactivity > 0.0:
        high = max(reactivity / generation, 1.0)
        while f(high) > 0.0:
            high *= 2.0
        # Для rho>0 корень между 0 и prompt pole; f(0)=rho>0, ищем смену знака вверх.
        # f(0+)=rho>0, f(+inf) -> -inf
        a, b = 1e-12, high
    else:
        # Доминирующий корень лежит в (-min(lambda), 0).
        a, b = -float(np.min(decay)) + 1e-10, -1e-12
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        raise ValueError("Корень inhour не локализован")
    for _ in range(80):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < 1e-14 or abs(b - a) < 1e-14:
            return mid
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


def batch_derivatives(
    amplitudes: np.ndarray,
    precursors: np.ndarray,
    reactivity: np.ndarray,
    kinetics: PointKinetics,
    source: np.ndarray | float | None = None,
) -> tuple:
    """Та же правая часть, что и у скалярной модели. precursors имеет форму (batch, groups).

    Источник по умолчанию — `kinetics.source` на каждый элемент пакета. Нулевой источник не подставляется вместо заданного.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    emitted = float(kinetics.source) if source is None else source
    try:
        emitted = np.broadcast_to(np.asarray(emitted, dtype=float), amplitudes.shape).copy()
    except ValueError as exc:
        raise DomainError("Источник пакета не согласован с амплитудами") from exc
    dn = (
        ((reactivity - kinetics.beta_total) / kinetics.generation_time_s) * amplitudes
        + precursors @ kinetics.decay
        + emitted
    )
    dC = (
        amplitudes[:, None] * kinetics.beta / kinetics.generation_time_s
        - precursors * kinetics.decay
    )
    return dn, dC
