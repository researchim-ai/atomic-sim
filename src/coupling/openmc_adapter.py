"""Контракт MGXS для OpenMC.

Отсутствующий пакет и отсутствующие ядерные данные не заменяются внутренним спектром.
Результат внешнего расчёта принимается только вместе со статистической ошибкой.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from core.status import Applicability, DomainError
from core.units import per_cm_to_per_m
from data.catalog import assert_nuclear_data_allowed

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
        raise DomainError(
            "Набор MGXS должен включать поток, поглощение, деление, nu-деление и рассеяние"
        )
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
                "message": "Модель подготовлена. Запуск требует execute=True и license.json источника endf-b-nndc",
                "version": self.status["version"],
                "manifest": manifest,
                "width_cm": float(specification["geometry"]["width_m"]) * 100.0,
            }
        assert_nuclear_data_allowed(cross_sections)
        report = execute_openmc_slab(specification, cross_sections)
        report["applicability"] = Applicability.IN_DOMAIN.value
        report["version"] = self.status["version"]
        report["manifest"] = manifest
        return report


def execute_openmc_slab(specification: Mapping, cross_sections: str) -> dict:
    """Запуск частиц OpenMC во временном каталоге. k без статистической ошибки не возвращается."""
    import openmc

    openmc.config["cross_sections"] = str(cross_sections)
    material_spec = specification["materials"][0]
    material = openmc.Material(name=str(material_spec["name"]))
    material.set_density("g/cm3", float(material_spec["density_kg_m3"]) / 1000.0)
    for name, fraction in material_spec["nuclides"].items():
        material.add_nuclide(str(name), float(fraction))
    materials = openmc.Materials([material])

    width_cm = float(specification["geometry"]["width_m"]) * 100.0
    left = openmc.XPlane(0.0, boundary_type=str(specification["geometry"]["left"]))
    right = openmc.XPlane(width_cm, boundary_type=str(specification["geometry"]["right"]))
    y_min = openmc.YPlane(-1.0, boundary_type="reflective")
    y_max = openmc.YPlane(1.0, boundary_type="reflective")
    z_min = openmc.ZPlane(-1.0, boundary_type="reflective")
    z_max = openmc.ZPlane(1.0, boundary_type="reflective")
    cell = openmc.Cell(fill=material, region=+left & -right & +y_min & -y_max & +z_min & -z_max)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = int(specification["settings"]["batches"])
    settings.inactive = int(specification["settings"]["inactive"])
    settings.particles = int(specification["settings"]["particles"])
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((1.0e-6, -0.99, -0.99), (width_cm - 1.0e-6, 0.99, 0.99))
    )

    groups = openmc.mgxs.EnergyGroups(_cm_bounds(specification["energy_bounds_ev"]))
    library = openmc.mgxs.Library(geometry)
    library.energy_groups = groups
    library.mgxs_types = ["absorption", "fission", "nu-fission", "scatter"]
    library.domain_type = "material"
    library.domains = [material]
    library.by_nuclide = False
    library.legendre_order = 0
    library.build_library()
    tallies = openmc.Tallies()
    if hasattr(library, "add_to_tallies"):
        library.add_to_tallies(tallies)
    else:
        library.add_to_tallies_file(tallies)
    flux = openmc.Tally(name="group-flux")
    flux.filters = [
        openmc.MaterialFilter([material]),
        openmc.EnergyFilter(_cm_bounds(specification["energy_bounds_ev"])),
    ]
    flux.scores = ["flux"]
    tallies.append(flux)

    model = openmc.Model(geometry, materials, settings, tallies)
    with tempfile.TemporaryDirectory(prefix="atomic-openmc-") as folder:
        statepoint_path = model.run(cwd=folder, threads=1)
        with openmc.StatePoint(statepoint_path) as statepoint:
            mean, std = _keff_mean_std(statepoint)
            if std <= 0.0 or not np.isfinite(mean):
                raise DomainError("OpenMC не вернул положительную статистическую ошибку k")
            library.load_from_statepoint(statepoint)
            absorption = _group_values(library.get_mgxs(material, "absorption").get_xs())
            flux_tally = statepoint.get_tally(name="group-flux")
            flux_std = _group_values(np.asarray(flux_tally.std_dev, dtype=float))
        return {
            "k_combined_mean": mean,
            "k_combined_std": std,
            "macroscopic_unit": "1/cm",
            "absorption_per_cm": absorption.tolist(),
            "flux_std": flux_std.tolist(),
            "openmc_version": getattr(openmc, "__version__", None),
            "statepoint": Path(statepoint_path).name,
            "message": "Частицы OpenMC выполнены, statepoint прочитан",
        }


def _keff_mean_std(statepoint) -> tuple:
    keff = getattr(statepoint, "keff", None)
    if keff is not None and hasattr(keff, "n") and hasattr(keff, "s"):
        return float(keff.n), float(keff.s)
    combined = getattr(statepoint, "k_combined", None)
    if combined is not None:
        return float(combined[0]), float(combined[1])
    raise DomainError("Statepoint не содержит k и его статистическую ошибку")


def _group_values(values) -> np.ndarray:
    """OpenMC нумерует группы от низкой энергии. Спецификация проекта — от высокой."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[::-1]


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
