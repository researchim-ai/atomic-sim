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
    phenomenon="Интерполяция макроскопических сечений по температуре топлива",
    approximations=("линейная интерполяция по T", "одно значение плотности и состава на таблицу"),
    applicability="Только узлы внутри выпуклой оболочки таблицы",
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
    def __init__(self, temperatures_k: np.ndarray, tables: Mapping[str, np.ndarray]):
        self.temperatures_k = np.asarray(temperatures_k, dtype=float)
        self.tables = {key: np.asarray(value, dtype=float) for key, value in tables.items()}
        order = np.argsort(self.temperatures_k)
        self.temperatures_k = self.temperatures_k[order]
        self.tables = {key: value[order] for key, value in self.tables.items()}

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
