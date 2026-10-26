"""Базовый контракт компонента."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from core.diagnostics import ComponentDiagnostics
from core.status import ModelPassport


class Component:
    passport: ModelPassport

    def declare_variables(self) -> Dict[str, str]:
        """Имя переменной -> 'differential' или 'algebraic'."""
        raise NotImplementedError

    def initialize(self, inputs: Mapping[str, float]) -> np.ndarray:
        raise NotImplementedError

    def residual(
        self, t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float]
    ) -> np.ndarray:
        raise NotImplementedError

    def diagnostics(self) -> ComponentDiagnostics:
        raise NotImplementedError

    def serialize(self) -> dict:
        raise NotImplementedError

    def restore(self, payload: dict) -> None:
        raise NotImplementedError
