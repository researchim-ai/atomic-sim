"""Сборка невязки F(t, y, ydot) = 0.

Принятый старт обязан обнулять невязку. Вырожденный якобиан не считается
согласованной схемой. Это каркас этапа E02, не модель всей установки.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.assembly import jacobian_rank
from core.status import DomainError


class ResidualBlock:
    size: int

    def residual(self, t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float]) -> np.ndarray:
        raise NotImplementedError


class IndexOneDrive(ResidualBlock):
    """x' = -x + z, z = drive. Индекс 1: z алгебраический."""

    size = 2

    def __init__(self, drive: float):
        self.drive = float(drive)

    def residual(self, _t: float, y: np.ndarray, ydot: np.ndarray, _u: Mapping[str, float]) -> np.ndarray:
        return np.array([ydot[0] + y[0] - y[1], y[1] - self.drive])


class DuplicateConstraint(ResidualBlock):
    """Две копии одного алгебраического условия. Ранг неполный."""

    size = 2

    def residual(self, _t: float, y: np.ndarray, _ydot: np.ndarray, _u: Mapping[str, float]) -> np.ndarray:
        gap = y[0] - 1.0
        return np.array([gap, 2.0 * gap])


def scale_residual(residual: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Безразмерная невязка. Масштаб — положительное число той же длины, что и F."""
    residual = np.asarray(residual, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if scales.shape != residual.shape or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise DomainError("Масштаб невязки должен быть положительным и той же длины, что F")
    return residual / scales


class ResidualSystem:
    def __init__(self, blocks: Sequence[ResidualBlock]):
        if not blocks:
            raise DomainError("Сборка невязки не содержит уравнений")
        self.blocks = list(blocks)
        self.size = sum(block.size for block in self.blocks)

    def residual(self, t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float] | None = None) -> np.ndarray:
        u = {} if u is None else u
        y = np.asarray(y, dtype=float)
        ydot = np.asarray(ydot, dtype=float)
        if y.size != self.size or ydot.size != self.size:
            raise DomainError("Длина состояния не совпадает со сборкой невязки")
        parts = []
        cursor = 0
        for block in self.blocks:
            parts.append(
                np.asarray(
                    block.residual(t, y[cursor : cursor + block.size], ydot[cursor : cursor + block.size], u),
                    dtype=float,
                )
            )
            cursor += block.size
        stacked = np.concatenate(parts)
        if stacked.size != self.size:
            raise DomainError("Блок вернул невязку не своего размера")
        return stacked

    def numerical_jacobian(
        self, t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float] | None = None, scale: float = 1.0e-7
    ) -> np.ndarray:
        """dF/dy + dF/dydot. Для index-1 эта сумма должна иметь полный ранг."""
        y = np.asarray(y, dtype=float)
        ydot = np.asarray(ydot, dtype=float)
        base_size = self.size
        jacobian = np.zeros((base_size, base_size))
        for index in range(base_size):
            step_y = scale * max(1.0, abs(y[index]))
            step_d = scale * max(1.0, abs(ydot[index]))
            up_y = y.copy()
            down_y = y.copy()
            up_d = ydot.copy()
            down_d = ydot.copy()
            up_y[index] += step_y
            down_y[index] -= step_y
            up_d[index] += step_d
            down_d[index] -= step_d
            jacobian[:, index] = (
                self.residual(t, up_y, up_d, u) - self.residual(t, down_y, down_d, u)
            ) / (step_y + step_d)
        return jacobian

    def require_consistent(
        self, t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float] | None = None, tolerance: float = 1.0e-10
    ) -> None:
        mismatch = float(np.linalg.norm(self.residual(t, y, ydot, u)))
        if mismatch > tolerance:
            raise DomainError("Начальное состояние не обнуляет невязку: %.3e" % mismatch)

    def require_full_rank(self, jacobian: np.ndarray) -> None:
        rank = jacobian_rank(jacobian)
        if rank < jacobian.shape[0]:
            raise DomainError("Якобиан невязки вырожден: ранг %s из %s" % (rank, jacobian.shape[0]))
