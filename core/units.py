"""Преобразования единиц. Внутренние величины ядра — СИ."""

from __future__ import annotations

from dataclasses import dataclass

BARNS_TO_M2 = 1.0e-28
CM_TO_M = 1.0e-2
PER_CM_TO_PER_M = 100.0
PCM_TO_ABSOLUTE = 1.0e-5
PA_TO_MPA = 1.0e-6
MPA_TO_PA = 1.0e6
J_TO_KJ = 1.0e-3
KJ_TO_J = 1.0e3
PPM_TO_FRACTION = 1.0e-6


@dataclass(frozen=True)
class Quantity:
    """Скаляр с явной единицей. Арифметика только для одинаковых единиц."""

    value: float
    unit: str

    def __add__(self, other: "Quantity") -> "Quantity":
        self._check(other)
        return Quantity(self.value + other.value, self.unit)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._check(other)
        return Quantity(self.value - other.value, self.unit)

    def _check(self, other: "Quantity") -> None:
        if not isinstance(other, Quantity) or other.unit != self.unit:
            raise TypeError(
                "Нельзя складывать величины с разной размерностью: %s и %s"
                % (self.unit, getattr(other, "unit", type(other)))
            )


def barns_to_m2(sigma_barn: float) -> float:
    return sigma_barn * BARNS_TO_M2


def per_cm_to_per_m(value: float) -> float:
    return value * PER_CM_TO_PER_M


def cm_to_m(value: float) -> float:
    return value * CM_TO_M


def pcm_to_rho(pcm: float) -> float:
    """Реактивность: 1 pcm = 1e-5 абсолютной dk/k."""
    return pcm * PCM_TO_ABSOLUTE


def pa_to_mpa(pressure_pa: float) -> float:
    return pressure_pa * PA_TO_MPA


def mpa_to_pa(pressure_mpa: float) -> float:
    return pressure_mpa * MPA_TO_PA
