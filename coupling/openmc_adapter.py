"""Контракт MGXS для OpenMC.

Отсутствующий пакет и отсутствующие ядерные данные не заменяются внутренним спектром.
Результат внешнего расчёта принимается только вместе со статистической ошибкой.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Mapping

import numpy as np

from core.status import Applicability, DomainError
from core.units import per_cm_to_per_m

MGXS_SCORES = ("absorption", "fission", "flux", "nu-fission", "scatter")


def openmc_status() -> dict:
    try:
        import openmc  # type: ignore
    except ImportError as exc:
        return {
            "applicability": Applicability.UNSUPPORTED.value,
            "message": "OpenMC не установлен: %s" % exc,
            "version": None,
        }
    return {
        "applicability": Applicability.IN_DOMAIN.value,
        "message": "OpenMC импортируется; расчёт частиц требует ядерных данных и execute=True",
        "version": getattr(openmc, "__version__", "unknown"),
    }


def slab_specification() -> dict:
    """Открытая спецификация пластины. Это не модель ВВЭР и не результат Monte Carlo."""
    return {
        "name": "open-two-group-slab",
        "energy_bounds_ev": [2.0e7, 0.625, 1.0e-5],
        "geometry": {"type": "slab", "width_m": 1.0, "left": "vacuum", "right": "vacuum"},
        "materials": [
            {
                "name": "test-fuel",
                "density_kg_m3": 10000.0,
                "nuclides": {"U235": 0.03, "U238": 0.97},
            }
        ],
        "settings": {"batches": 50, "inactive": 10, "particles": 1000},
        "scores": list(MGXS_SCORES),
        "openmc_length_unit": "cm",
        "reported_macroscopic_unit": "1/m",
    }


def validate_specification(specification: Mapping) -> None:
    bounds = specification.get("energy_bounds_ev")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 3:
        raise DomainError("Нужны убывающие границы групп в эВ, не меньше двух групп")
    if any(float(left) <= float(right) for left, right in zip(bounds, bounds[1:])):
        raise DomainError("Границы энергии групп должны строго убывать")
    geometry = specification.get("geometry") or {}
    if geometry.get("type") != "slab" or float(geometry.get("width_m", 0.0)) <= 0.0:
        raise DomainError("Сейчас подготовлена только пластина положительной ширины")
    if geometry.get("left") not in ("vacuum", "reflective") or geometry.get("right") not in (
        "vacuum",
        "reflective",
    ):
        raise DomainError("Граница пластины OpenMC не входит в vacuum/reflective")
    materials = specification.get("materials") or []
    if len(materials) != 1:
        raise DomainError("Тестовый pipeline готовит один гомогенный материал")
    fractions = materials[0].get("nuclides") or {}
    if not fractions or abs(sum(float(value) for value in fractions.values()) - 1.0) > 1.0e-8:
        raise DomainError("Атомные доли материала должны суммироваться в 1")
    if float(materials[0].get("density_kg_m3", 0.0)) <= 0.0:
        raise DomainError("Плотность материала должна быть положительной")
    scores = tuple(sorted(specification.get("scores") or []))
    if scores != MGXS_SCORES:
        raise DomainError("Набор MGXS должен включать поток, поглощение, деление, nu-деление и рассеяние")
    settings = specification.get("settings") or {}
    if int(settings.get("particles", 0)) < 1 or int(settings.get("batches", 0)) <= int(
        settings.get("inactive", 0)
    ):
        raise DomainError("Число частиц и активных поколений OpenMC задано неверно")
    if specification.get("openmc_length_unit") != "cm":
        raise DomainError("Внутренняя геометрия OpenMC в этом контракте задаётся в см")
    if specification.get("reported_macroscopic_unit") != "1/m":
        raise DomainError("Наружу макросечения отдаются в 1/м")


def specification_manifest(specification: Mapping) -> dict:
    validate_specification(specification)
    canonical = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "groups": len(specification["energy_bounds_ev"]) - 1,
        "width_m": float(specification["geometry"]["width_m"]),
        "openmc_length_unit": "cm",
        "reported_macroscopic_unit": "1/m",
    }


def _cm_bounds(bounds_ev_descending) -> list:
    """OpenMC ждёт возрастающие границы. Значения энергии остаются в эВ."""
    return [float(value) for value in reversed(bounds_ev_descending)]


class OpenMCAdapter:
    """Не подменяет отсутствующий OpenMC внутренним спектром."""

    def __init__(self) -> None:
        self.status = openmc_status()

    def run_mgxs(self, specification: Mapping, execute: bool = False) -> dict:
        validate_specification(specification)
        manifest = specification_manifest(specification)
        if self.status["applicability"] != Applicability.IN_DOMAIN.value:
            return {
                "applicability": Applicability.UNSUPPORTED.value,
                "message": self.status["message"],
                "version": None,
                "manifest": manifest,
            }
        cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
        if not cross_sections:
            return {
                "applicability": Applicability.OUT_OF_DOMAIN.value,
                "message": "OPENMC_CROSS_SECTIONS не задан: частицы не запускаются и k не подставляется",
                "version": self.status["version"],
                "manifest": manifest,
                "energy_bounds_ev_ascending": _cm_bounds(specification["energy_bounds_ev"]),
                "width_cm": float(specification["geometry"]["width_m"]) * 100.0,
            }
        if not execute:
            return {
                "applicability": Applicability.OUT_OF_DOMAIN.value,
                "message": "Модель подготовлена; запуск поколений требует execute=True",
                "version": self.status["version"],
                "manifest": manifest,
                "width_cm": float(specification["geometry"]["width_m"]) * 100.0,
            }
        raise DomainError(
            "Запуск OpenMC из этого процесса не включён: нет проверенного каталога данных и бюджета частиц"
        )


def interpret_mgxs_result(payload: Mapping) -> dict:
    """Перевод отчёта внешнего кода в СИ. Без k_std результат отвергается."""
    if "k_combined_std" not in payload or payload["k_combined_std"] is None:
        raise DomainError("Статистическая ошибка k не отделена от среднего")
    if float(payload["k_combined_std"]) < 0.0:
        raise DomainError("Стандартное отклонение k отрицательно")
    if payload.get("macroscopic_unit") != "1/cm":
        raise DomainError("Ожидаются макросечения OpenMC в 1/см до перевода в 1/м")
    if "absorption_per_cm" not in payload or "flux_std" not in payload:
        raise DomainError("В отчёте MGXS нет поглощения или ошибки потока")
    absorption = per_cm_to_per_m(np.asarray(payload["absorption_per_cm"], dtype=float))
    return {
        "k_mean": float(payload["k_combined_mean"]),
        "k_std": float(payload["k_combined_std"]),
        "absorption_per_m": absorption,
        "flux_std": np.asarray(payload["flux_std"], dtype=float),
        "openmc_version": payload.get("openmc_version"),
    }


def compare_k(internal_k: float, external: Mapping) -> dict:
    """Разница внутреннего k и внешнего среднего в единицах заявленной ошибки Monte Carlo."""
    interpreted = external if "k_std" in external else interpret_mgxs_result(external)
    std = float(interpreted["k_std"])
    if std <= 0.0:
        raise DomainError("Сравнение с внешним k требует положительной статистической ошибки")
    delta = float(internal_k) - float(interpreted["k_mean"])
    return {"delta_k": delta, "delta_over_std": delta / std}
