"""Статусы моделей и применимости.

`implemented`, `verified`, `validated` и `plant_referenced` не повышаются
автоматически из-за существования файлов или прохождения произвольных тестов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class Applicability(str, Enum):
    IN_DOMAIN = "in_domain"
    OUT_OF_DOMAIN = "out_of_domain"
    UNSUPPORTED = "unsupported"


class ModelStatus(str, Enum):
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    VALIDATED = "validated"
    PLANT_REFERENCED = "plant_referenced"


class DomainError(RuntimeError):
    """Расчёт вышел за заявленную область или режим не поддержан."""

    def __init__(self, message: str, applicability: Applicability = Applicability.OUT_OF_DOMAIN):
        super().__init__(message)
        self.applicability = applicability


class SolverError(RuntimeError):
    """Интегратор не сошёлся. Результат не подменяется ограничением."""


@dataclass(frozen=True)
class ModelPassport:
    """Паспорт модели: явление, приближения, статус и источники данных."""

    model_id: str
    version: str
    status: ModelStatus
    phenomenon: str
    approximations: Tuple[str, ...]
    applicability: str
    data_sources: Tuple[str, ...]
    plant_referenced: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "phenomenon": self.phenomenon,
            "approximations": list(self.approximations),
            "applicability": self.applicability,
            "data_sources": list(self.data_sources),
            "plant_referenced": self.plant_referenced,
            "notes": list(self.notes),
        }
