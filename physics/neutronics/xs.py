"""Провайдер сечений. За пределами таблицы — out_of_domain, без тихой экстраполяции."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from core.status import Applicability, DomainError, ModelPassport, ModelStatus
from core.units import barns_to_m2, per_cm_to_per_m

PASSPORT = ModelPassport(
    model_id="xs-provider-open-test",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Макросечения по температуре, относительной плотности и выгоранию",
    approximations=(
        "линейная интерполяция по температуре топлива",
        "однородное масштабирование состава: Sigma ~ плотность, D ~ 1/плотность",
        "ось выгорания не экстраполируется",
    ),
    applicability="Только узлы внутри таблицы температуры и выгорания; плотность > 0",
    data_sources=(
        "Открытая тестовая двухгрупповая библиотека data/open_library.py, не оценённые файлы ENDF",
    ),
)


@dataclass(frozen=True)
class MacroscopicState:
    diffusion_m: np.ndarray
    sigma_a_per_m: np.ndarray
    sigma_f_per_m: np.ndarray
    nu_sigma_f_per_m: np.ndarray
    kappa_sigma_f_j_m: np.ndarray
    scatter_to: np.ndarray
    applicability: Applicability = Applicability.IN_DOMAIN


class CrossSectionProvider:
    def __init__(
        self,
        temperatures_k: np.ndarray,
        tables: Mapping[str, np.ndarray],
        burnup_mwd_per_kg: np.ndarray | None = None,
    ):
        self.temperatures_k = np.asarray(temperatures_k, dtype=float)
        self.tables = {key: np.asarray(value, dtype=float) for key, value in tables.items()}
        order = np.argsort(self.temperatures_k)
        self.temperatures_k = self.temperatures_k[order]
        self.tables = {key: value[order] for key, value in self.tables.items()}
        if burnup_mwd_per_kg is None:
            burnup_mwd_per_kg = np.array([0.0])
        self.burnup_mwd_per_kg = np.asarray(burnup_mwd_per_kg, dtype=float)
        if self.burnup_mwd_per_kg.ndim != 1 or self.burnup_mwd_per_kg.size < 1:
            raise DomainError("Ось выгорания библиотеки пуста")
        if np.any(np.diff(self.burnup_mwd_per_kg) < 0.0):
            raise DomainError("Выгорание в библиотеке должно неубывать")

    def at_temperature(self, temperature_k: float) -> MacroscopicState:
        if temperature_k < self.temperatures_k[0] or temperature_k > self.temperatures_k[-1]:
            raise DomainError("Температура %.2f K вне таблицы сечений" % temperature_k)
        high = int(np.searchsorted(self.temperatures_k, temperature_k, side="left"))
        if high == 0:
            high = 1
        low = high - 1
        span = self.temperatures_k[high] - self.temperatures_k[low]
        weight = 0.0 if span == 0.0 else (temperature_k - self.temperatures_k[low]) / span
        mixed = {
            key: (1.0 - weight) * value[low] + weight * value[high]
            for key, value in self.tables.items()
        }
        if np.any(mixed["sigma_a"] < 0.0) or np.any(mixed["sigma_f"] < 0.0):
            raise DomainError("Интерполяция нарушила неотрицательность сечений")
        return MacroscopicState(
            diffusion_m=mixed["D"],
            sigma_a_per_m=mixed["sigma_a"],
            sigma_f_per_m=mixed["sigma_f"],
            nu_sigma_f_per_m=mixed["nu_sigma_f"],
            kappa_sigma_f_j_m=mixed["kappa_sigma_f"],
            scatter_to=mixed["scatter"],
        )

    def at_state(
        self, temperature_k: float, density_ratio: float = 1.0, burnup_mwd_per_kg: float = 0.0
    ) -> MacroscopicState:
        """Плотность масштабирует макросечения опорного состава. Выгорание вне узлов — отказ."""
        if density_ratio <= 0.0:
            raise DomainError("Относительная плотность сечений должна быть положительной")
        low = float(self.burnup_mwd_per_kg[0])
        high = float(self.burnup_mwd_per_kg[-1])
        if burnup_mwd_per_kg < low - 1.0e-9 or burnup_mwd_per_kg > high + 1.0e-9:
            raise DomainError(
                "Выгорание %.4g МВт·сут/кг вне таблицы сечений; экстраполяция не выполняется"
                % burnup_mwd_per_kg
            )
        if self.burnup_mwd_per_kg.size > 1:
            raise DomainError(
                "Библиотека объявила несколько выгораний, но таблица по этой оси не заполнена"
            )
        base = self.at_temperature(temperature_k)
        ratio = float(density_ratio)
        return MacroscopicState(
            diffusion_m=base.diffusion_m / ratio,
            sigma_a_per_m=base.sigma_a_per_m * ratio,
            sigma_f_per_m=base.sigma_f_per_m * ratio,
            nu_sigma_f_per_m=base.nu_sigma_f_per_m * ratio,
            kappa_sigma_f_j_m=base.kappa_sigma_f_j_m * ratio,
            scatter_to=base.scatter_to * ratio,
        )


def micro_to_macro(
    number_densities: Dict[str, float], micro_barn: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Sigma [1/м] = sum N[1/м^3] * sigma[барн]."""
    reactions = next(iter(micro_barn.values())).keys()
    out = {}
    for reaction in reactions:
        total = None
        for name, density in number_densities.items():
            sigma_m2 = barns_to_m2(1.0) * np.asarray(micro_barn[name][reaction], dtype=float)
            term = density * sigma_m2
            total = term if total is None else total + term
        out[reaction] = total
    return out


def import_per_cm(value_per_cm: float) -> float:
    return per_cm_to_per_m(value_per_cm)
