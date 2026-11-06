"""Сосредоточенная теплоёмкость как компонент DAE.

Невязка записана явно: C dT/dt + UA (T - T_sink) - Q = 0.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from core.component import Component
from core.diagnostics import ComponentDiagnostics
from core.status import Applicability, DomainError, ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="lumped-energy",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Одна теплоёмкость с линейным теплоотводом",
    approximations=("постоянные C и UA", "температура стока задана"),
    applicability="C > 0, UA > 0",
    data_sources=("Баланс энергии сосредоточенного тела",),
)


class LumpedEnergy(Component):
    passport = PASSPORT

    def __init__(self, capacity_j_k: float, conductance_w_k: float, sink_k: float, heat_w: float):
        if capacity_j_k <= 0.0 or conductance_w_k <= 0.0:
            raise DomainError("Теплоёмкость и проводимость сосредоточенного тела положительны")
        self.capacity_j_k = float(capacity_j_k)
        self.conductance_w_k = float(conductance_w_k)
        self.sink_k = float(sink_k)
        self.heat_w = float(heat_w)

    def declare_variables(self) -> dict:
        return {"temperature_k": "differential"}

    def consistent_temperature(self, heat_w: float | None = None) -> float:
        heat = self.heat_w if heat_w is None else heat_w
        return self.sink_k + heat / self.conductance_w_k

    def initialize(self, inputs: Mapping[str, float]) -> np.ndarray:
        heat = float(inputs.get("heat_w", self.heat_w))
        return np.array([self.consistent_temperature(heat)])

    def residual(
        self, _t: float, y: np.ndarray, ydot: np.ndarray, u: Mapping[str, float]
    ) -> np.ndarray:
        heat = float(u.get("heat_w", self.heat_w))
        balance = self.capacity_j_k * ydot[0] + self.conductance_w_k * (y[0] - self.sink_k) - heat
        return np.array([balance / self.capacity_j_k])

    def diagnostics(self) -> ComponentDiagnostics:
        return ComponentDiagnostics("lumped-energy", Applicability.IN_DOMAIN)

    def serialize(self) -> dict:
        return {
            "capacity_j_k": self.capacity_j_k,
            "conductance_w_k": self.conductance_w_k,
            "sink_k": self.sink_k,
            "heat_w": self.heat_w,
        }

    def restore(self, payload: dict) -> None:
        self.capacity_j_k = float(payload["capacity_j_k"])
        self.conductance_w_k = float(payload["conductance_w_k"])
        self.sink_k = float(payload["sink_k"])
        self.heat_w = float(payload["heat_w"])
