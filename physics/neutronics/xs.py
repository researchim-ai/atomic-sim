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
        "прирост поглощения бора интерполируется отдельно и масштабируется плотностью",
        "оси выгорания и бора не экстраполируются",
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
        boron_ppm: np.ndarray | None = None,
        boron_delta_sigma_a: np.ndarray | None = None,
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
        if (boron_ppm is None) != (boron_delta_sigma_a is None):
            raise DomainError("Ось бора и прибавка сечения задаются вместе")
        self.boron_ppm = None if boron_ppm is None else np.asarray(boron_ppm, dtype=float)
        self.boron_delta_sigma_a = (
            None if boron_delta_sigma_a is None else np.asarray(boron_delta_sigma_a, dtype=float)
        )
        if self.boron_ppm is not None:
            if self.boron_ppm.ndim != 1 or self.boron_ppm.size < 2:
                raise DomainError("Ось бора должна содержать хотя бы два узла")
            if np.any(np.diff(self.boron_ppm) <= 0.0):
                raise DomainError("Концентрация бора в библиотеке должна возрастать")
            if self.boron_delta_sigma_a.shape[0] != self.boron_ppm.size:
                raise DomainError("Прибавка сечения бора не совпадает с осью концентраций")

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

    def _boron_absorption(self, boron_ppm: float) -> np.ndarray:
        axis = self.boron_ppm
        table = self.boron_delta_sigma_a
        if boron_ppm < axis[0] or boron_ppm > axis[-1]:
            raise DomainError("Бор %.4g ppm вне таблицы сечений" % boron_ppm)
        high = int(np.searchsorted(axis, boron_ppm, side="left"))
        if high == 0:
            high = 1
        low = high - 1
        span = axis[high] - axis[low]
        weight = 0.0 if span == 0.0 else (boron_ppm - axis[low]) / span
        return (1.0 - weight) * table[low] + weight * table[high]

    def at_state(
        self,
        temperature_k: float,
        density_ratio: float = 1.0,
        burnup_mwd_per_kg: float = 0.0,
        boron_ppm: float | None = None,
    ) -> MacroscopicState:
        """Плотность масштабирует макросечения опорного состава. Выгорание и бор вне узлов — отказ."""
        if density_ratio <= 0.0:
            raise DomainError("Относительная плотность сечений должна быть положительной")
        if (boron_ppm is None) != (self.boron_ppm is None):
            if boron_ppm is None:
                raise DomainError("Библиотека сечений требует концентрацию бора")
            raise DomainError("Бор не входит в эту библиотеку; сечение не подменяется")
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
        absorption = base.sigma_a_per_m * ratio
        if boron_ppm is not None:
            absorption = absorption + self._boron_absorption(float(boron_ppm)) * ratio
        if np.any(absorption < 0.0):
            raise DomainError("Поглощение с бором стало отрицательным")
        return MacroscopicState(
            diffusion_m=base.diffusion_m / ratio,
            sigma_a_per_m=absorption,
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


class SelfShieldingTable:
    """Фактор самоэкранирования по фоновому сечению. За таблицей фактор не экстраполируется."""

    def __init__(self, background_per_m: np.ndarray, factors: np.ndarray):
        self.background_per_m = np.asarray(background_per_m, dtype=float)
        self.factors = np.asarray(factors, dtype=float)
        if self.background_per_m.ndim != 1 or self.background_per_m.size < 2:
            raise DomainError("Таблица самоэкранирования должна содержать хотя бы два фона")
        if self.factors.shape != self.background_per_m.shape or np.any(self.factors < 0.0):
            raise DomainError("Факторы самоэкранирования неотрицательны и лежат на оси фона")
        if np.any(np.diff(self.background_per_m) <= 0.0):
            raise DomainError("Фоновое сечение в таблице должно возрастать")

    def factor(self, background_per_m: float) -> float:
        axis = self.background_per_m
        if background_per_m < axis[0] or background_per_m > axis[-1]:
            raise DomainError("Фон %.4g 1/м вне таблицы самоэкранирования" % background_per_m)
        high = int(np.searchsorted(axis, background_per_m, side="left"))
        if high == 0:
            high = 1
        low = high - 1
        span = axis[high] - axis[low]
        weight = 0.0 if span == 0.0 else (background_per_m - axis[low]) / span
        return float((1.0 - weight) * self.factors[low] + weight * self.factors[high])

    def shielded_macro(self, infinite_dilute_per_m: float, background_per_m: float) -> float:
        if infinite_dilute_per_m < 0.0:
            raise DomainError("Неэкранированное макросечение отрицательно")
        return infinite_dilute_per_m * self.factor(background_per_m)


def require_fission_spectrum(chi: np.ndarray) -> np.ndarray:
    """Спектр деления уже нормирован на единицу. Тихая перенормировка не выполняется."""
    spectrum = np.asarray(chi, dtype=float)
    if spectrum.ndim != 1 or spectrum.size == 0 or np.any(spectrum < 0.0):
        raise DomainError("Спектр деления должен быть неотрицательным вектором групп")
    if abs(float(np.sum(spectrum)) - 1.0) > 1.0e-12:
        raise DomainError("Спектр деления должен суммироваться в 1; нормировка не подставляется")
    return spectrum


def fission_source_per_group(
    chi: np.ndarray, nu_sigma_f: np.ndarray, flux: np.ndarray
) -> np.ndarray:
    """Источник группы chi_g * сумма nuSigma_f phi. Сумма по группам равна полному производству."""
    spectrum = require_fission_spectrum(chi)
    production = np.asarray(nu_sigma_f, dtype=float) * np.asarray(flux, dtype=float)
    if production.shape != spectrum.shape:
        raise DomainError("Сечение деления и поток заданы не на группах спектра")
    if np.any(production < 0.0):
        raise DomainError("Производство нейтронов отрицательно")
    return spectrum * float(np.sum(production))


def normalize_flux_to_power(kappa_sigma_f, flux, volume_m3, target_w: float):
    """Масштаб потока, при котором мощность равна заданной. Форма потока не меняется."""
    if target_w <= 0.0:
        raise DomainError("Целевая мощность нормировки должна быть положительной")
    current = fission_power_w(kappa_sigma_f, flux, volume_m3)
    if current <= 0.0:
        raise DomainError("Нулевая мощность не задаёт масштаб потока")
    scale = target_w / current
    return np.asarray(flux, dtype=float) * scale, scale


def fission_power_w(kappa_sigma_f, flux, volume_m3) -> float:
    """Мощность sum kappa Sigma_f phi V. kappa Sigma_f в Дж/м, поток в 1/(м²·с), объём в м³."""
    kappa = np.asarray(kappa_sigma_f, dtype=float)
    flow = np.asarray(flux, dtype=float)
    volume = np.asarray(volume_m3, dtype=float)
    if kappa.shape != flow.shape:
        raise DomainError("kappa-Sigma_f и поток заданы на разных ячейках")
    if np.any(kappa < 0.0) or np.any(flow < 0.0) or np.any(volume <= 0.0):
        raise DomainError("Мощность деления задана при отрицательном сечении, потоке или объёме")
    try:
        weighted = kappa * flow * volume
    except ValueError as exc:
        raise DomainError("Объём ячейки не согласован с полем мощности") from exc
    return float(np.sum(weighted))
