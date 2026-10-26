"""Изотопные цепочки, ксенон и остаточное тепло.

Скорость деления R_f = Sigma_f * phi, а не nu*Sigma_f*phi.
Остаточное тепло хранится отдельно от запаздывающих нейтронов.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from core.solvers import integrate
from core.status import ModelPassport, ModelStatus

XENON_PASSPORT = ModelPassport(
    model_id="iodine-xenon",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Накопление I-135 и Xe-135",
    approximations=("одна энергетическая группа микросечения ксенона", "постоянный поток на шаге"),
    applicability="Поток и сечения, для которых заданы выходы; не переносится на другой спектр без замены данных",
    data_sources=(
        "Выходы и постоянные распада теплового деления U-235, учебный набор: gamma_I=0.061, gamma_Xe=0.003, lambda_I=2.87e-5 1/с, lambda_Xe=2.09e-5 1/с",
    ),
)

DECAY_PASSPORT = ModelPassport(
    model_id="decay-heat-3g",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Остаточное тепловыделение тремя экспоненциальными группами",
    approximations=("доли групп постоянны", "всё остаточное тепло выделяется в топливе"),
    applicability="Иллюстративная трёхгрупповая модель для баланса энергии. Это не библиотека ANS-5.1 и не паспорт выдержки блока",
    data_sources=(
        "Доли подобраны так, чтобы равновесная доля остаточного тепла была 6.5% мощности делений",
    ),
)


@dataclass
class XenonIodine:
    gamma_i: float = 0.061
    gamma_xe: float = 0.003
    lambda_i: float = 2.87e-5
    lambda_xe: float = 2.09e-5
    sigma_xe_m2: float = 2.65e-22

    def derivatives(
        self, iodine: float, xenon: float, fission_rate: float, flux: float
    ) -> np.ndarray:
        d_i = self.gamma_i * fission_rate - self.lambda_i * iodine
        d_xe = (
            self.gamma_xe * fission_rate
            + self.lambda_i * iodine
            - (self.lambda_xe + self.sigma_xe_m2 * flux) * xenon
        )
        return np.array([d_i, d_xe])

    def equilibrium(self, fission_rate: float, flux: float) -> np.ndarray:
        iodine = self.gamma_i * fission_rate / self.lambda_i
        xenon = (self.gamma_xe * fission_rate + self.lambda_i * iodine) / (
            self.lambda_xe + self.sigma_xe_m2 * flux
        )
        return np.array([iodine, xenon])

    def reference(
        self, state: np.ndarray, fission_rate: float, flux: float, dt_s: float
    ) -> np.ndarray:
        removal_xe = self.lambda_xe + self.sigma_xe_m2 * flux
        matrix = np.array([[-self.lambda_i, 0.0], [self.lambda_i, -removal_xe]])
        source = np.array([self.gamma_i * fission_rate, self.gamma_xe * fission_rate])
        expA = expm(matrix * dt_s)
        particular = np.linalg.solve(matrix, (expA - np.eye(2)) @ source)
        return expA @ state + particular

    def integrate(
        self, state: np.ndarray, fission_rate: float, flux: float, dt_s: float
    ) -> np.ndarray:
        def rhs(_time: float, y: np.ndarray) -> np.ndarray:
            return self.derivatives(y[0], y[1], fission_rate, flux)

        return integrate(rhs, state, (0.0, dt_s), rtol=1e-8, atol=1e-12).final


def bateman_daughter(
    parent0: float, lambda_parent: float, lambda_daughter: float, time_s: float
) -> float:
    """N_d(t) при N_d(0)=0 и чистом распаде родителя."""
    if abs(lambda_parent - lambda_daughter) < 1e-14:
        return parent0 * lambda_parent * time_s * np.exp(-lambda_parent * time_s)
    return (
        parent0
        * lambda_parent
        / (lambda_daughter - lambda_parent)
        * (np.exp(-lambda_parent * time_s) - np.exp(-lambda_daughter * time_s))
    )


def decay_chain_matrix(lambdas: np.ndarray) -> np.ndarray:
    """Последовательная цепь A->B->C... Матрица сохраняет число ядер плюс стабильный конец."""
    n = lambdas.size
    matrix = np.zeros((n, n))
    for i, lam in enumerate(lambdas):
        matrix[i, i] = -lam
        if i + 1 < n:
            matrix[i + 1, i] = lam
    return matrix


@dataclass
class DecayHeat:
    """dH_i/dt = f_i * P_fission - lambda_i * H_i, q_i = lambda_i * H_i.

    В равновесии sum q_i = sum f_i * P_fission. Мгновенное тепло — остальная доля.
    """

    fractions: np.ndarray
    decay_constants: np.ndarray

    def __post_init__(self) -> None:
        self.fractions = np.asarray(self.fractions, dtype=float)
        self.decay_constants = np.asarray(self.decay_constants, dtype=float)
        self.decay_fraction = float(np.sum(self.fractions))
        if self.decay_fraction <= 0.0 or self.decay_fraction >= 1.0:
            raise ValueError("Сумма долей остаточного тепла должна лежать в (0, 1)")

    @classmethod
    def illustrative(cls) -> "DecayHeat":
        return cls(np.array([0.02, 0.025, 0.02]), np.array([0.1, 0.01, 0.001]))

    def equilibrium(self, fission_power_w: float) -> np.ndarray:
        return self.fractions * fission_power_w / self.decay_constants

    def decay_power_w(self, stored_j: np.ndarray) -> float:
        return float(np.sum(self.decay_constants * stored_j))

    def prompt_power_w(self, fission_power_w: float) -> float:
        return (1.0 - self.decay_fraction) * fission_power_w

    def total_power_w(self, fission_power_w: float, stored_j: np.ndarray) -> float:
        return self.prompt_power_w(fission_power_w) + self.decay_power_w(stored_j)

    def derivatives(self, stored_j: np.ndarray, fission_power_w: float) -> np.ndarray:
        return self.fractions * fission_power_w - self.decay_constants * stored_j

    def integrate(self, stored_j: np.ndarray, fission_power_w: float, dt_s: float) -> np.ndarray:
        def rhs(_time: float, y: np.ndarray) -> np.ndarray:
            return self.derivatives(y, fission_power_w)

        return integrate(rhs, stored_j, (0.0, dt_s), rtol=1e-8, atol=1e-6).final


def advance_nuclides(concentrations: np.ndarray, matrix: np.ndarray, dt_s: float) -> np.ndarray:
    """N(t) = exp(A dt) N. Для постоянных реакционных скоростей."""
    return expm(matrix * dt_s) @ concentrations
