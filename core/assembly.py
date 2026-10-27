"""Сборщик связности. Не задаёт недостающую физику молча."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class EquationSpec:
    name: str
    differential: bool


@dataclass
class AssemblyReport:
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_topology(
    volumes: Sequence[str],
    junctions: Sequence[Tuple[str, str, str]],
    boundary_pressures: Dict[str, float],
) -> AssemblyReport:
    """Проверить объёмы, дубли соединений и незаданные границы открытых концов.

    `junctions` — (имя, объём_откуда, объём_куда). Имя объёма `boundary:*`
    требует заданного давления.
    """
    report = AssemblyReport()
    known = set(volumes)
    seen = set()
    for name, upstream, downstream in junctions:
        if name in seen:
            report.errors.append("Дублированное соединение %s" % name)
        seen.add(name)
        for end in (upstream, downstream):
            if end.startswith("boundary:"):
                if end not in boundary_pressures:
                    report.errors.append("Не задано граничное давление %s" % end)
            elif end not in known:
                report.errors.append(
                    "Соединение %s ссылается на неизвестный объём %s" % (name, end)
                )
    if not volumes:
        report.errors.append("Схема не содержит контрольных объёмов")
    return report


def jacobian_rank(matrix, relative_tolerance: float = 1.0e-8) -> int:
    """Численный ранг. Нулевая матрица имеет ранг 0 и не считается полноранговой."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Ранг определён для непустой матрицы")
    singular = np.linalg.svd(array, compute_uv=False)
    scale = float(singular[0]) if singular.size else 0.0
    if scale == 0.0:
        return 0
    return int(np.sum(singular > relative_tolerance * scale))


def differential_mask(specs: Sequence[EquationSpec]) -> List[bool]:
    if not specs:
        raise ValueError("Пустая система уравнений")
    names = [item.name for item in specs]
    if len(names) != len(set(names)):
        raise ValueError("Повторяющиеся неизвестные в маске состояния")
    return [item.differential for item in specs]
