"""Реестр свойств. Выбор теплоносителя явный."""

from __future__ import annotations

from typing import Callable, Dict

from core.status import Applicability, DomainError
from physics.materials.coolants import (
    flibe_density_kg_m3,
    helium_cp_j_kg_k,
    helium_density_kg_m3,
    lead_density_kg_m3,
    sodium_cp_j_kg_k,
    sodium_density_kg_m3,
)
from physics.materials.iapws import ThermoState, properties_pt

WATER = "water"
SODIUM = "sodium"
LEAD = "lead"
HELIUM = "helium"
FLIBE = "flibe"


def water_state(pressure_pa: float, temperature_k: float) -> ThermoState:
    return properties_pt(pressure_pa, temperature_k)


def density(coolant: str, pressure_pa: float, temperature_k: float) -> float:
    if coolant == WATER:
        return water_state(pressure_pa, temperature_k).rho_kg_m3
    if coolant == SODIUM:
        return sodium_density_kg_m3(temperature_k)
    if coolant == LEAD:
        return lead_density_kg_m3(temperature_k)
    if coolant == HELIUM:
        return helium_density_kg_m3(pressure_pa, temperature_k)
    if coolant == FLIBE:
        return flibe_density_kg_m3(temperature_k)
    raise DomainError("Теплоноситель %s не поддержан" % coolant, Applicability.UNSUPPORTED)


def specific_heat(coolant: str, pressure_pa: float, temperature_k: float) -> float:
    if coolant == WATER:
        return water_state(pressure_pa, temperature_k).cp_j_kg_k
    if coolant == SODIUM:
        return sodium_cp_j_kg_k(temperature_k)
    if coolant == HELIUM:
        return helium_cp_j_kg_k(temperature_k)
    raise DomainError(
        "Теплоёмкость %s не задана отдельной корреляцией" % coolant, Applicability.UNSUPPORTED
    )


SUPPORTED: Dict[str, Callable] = {
    WATER: water_state,
    SODIUM: sodium_density_kg_m3,
    LEAD: lead_density_kg_m3,
    HELIUM: helium_density_kg_m3,
    FLIBE: flibe_density_kg_m3,
}
