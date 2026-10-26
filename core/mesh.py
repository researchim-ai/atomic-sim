"""Одномерная конечно-объёмная сетка."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh1D:
    """Ячейки между гранями `x_faces` (м). Объём на единицу площади равен dx."""

    x_faces: np.ndarray

    def __post_init__(self) -> None:
        self.x_faces = np.asarray(self.x_faces, dtype=float)
        if self.x_faces.ndim != 1 or self.x_faces.size < 2:
            raise ValueError("Нужно не менее двух граней")
        if np.any(np.diff(self.x_faces) <= 0.0):
            raise ValueError("Грани должны строго возрастать")
        self.dx = np.diff(self.x_faces)
        self.centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
        self.n_cells = int(self.dx.size)
        self.length = float(self.x_faces[-1] - self.x_faces[0])

    @classmethod
    def uniform(cls, length: float, n_cells: int, origin: float = 0.0) -> "Mesh1D":
        if n_cells < 1 or length <= 0.0:
            raise ValueError("Длина и число ячеек должны быть положительными")
        faces = origin + np.linspace(0.0, length, n_cells + 1)
        return cls(faces)


def harmonic_mean(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        raise ValueError("Гармоническое среднее определено для положительных коэффициентов")
    return 2.0 * left * right / (left + right)


def cylindrical_area_average_radius(r_inner: float, r_outer: float) -> float:
    """Радиус, сохраняющий площадь кольца: sqrt((r_o^2+r_i^2)/2)."""
    return float(np.sqrt(0.5 * (r_inner * r_inner + r_outer * r_outer)))
