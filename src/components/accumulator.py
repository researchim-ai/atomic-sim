"""Гидроаккумулятор: несжимаемая жидкость и газовая подушка.

p V^gamma = const. Изотермический газ — gamma = 1. Исчезновение подушки
или жидкости останавливает модель. В петлю ВВЭР не подключён.
"""

from __future__ import annotations

from core.status import DomainError, ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="gas-accumulator",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Давление газовой подушки при изменении объёма жидкости",
    approximations=("жидкость несжимаема", "газ подчиняется p V^gamma = const", "растворение газа не моделируется"),
    applicability="Остаются и жидкость, и газ; gamma >= 1",
    data_sources=("Политропное сжатие идеального газа",),
)


class GasAccumulator:
    passport = PASSPORT

    def __init__(
        self,
        total_volume_m3: float,
        gas_volume_m3: float,
        pressure_pa: float,
        gamma: float,
        liquid_density_kg_m3: float,
    ):
        if gamma < 1.0 or pressure_pa <= 0.0 or liquid_density_kg_m3 <= 0.0:
            raise DomainError("Аккумулятор: gamma >= 1, давление и плотность жидкости положительны")
        if not 0.0 < gas_volume_m3 < total_volume_m3:
            raise DomainError("Газовая подушка должна занимать часть объёма бака")
        self.total_volume_m3 = float(total_volume_m3)
        self.gas_volume_m3 = float(gas_volume_m3)
        self.pressure_pa = float(pressure_pa)
        self.gamma = float(gamma)
        self.liquid_density_kg_m3 = float(liquid_density_kg_m3)
        self._invariant = self.pressure_pa * self.gas_volume_m3**self.gamma

    def liquid_volume_m3(self) -> float:
        return self.total_volume_m3 - self.gas_volume_m3

    def liquid_mass_kg(self) -> float:
        return self.liquid_volume_m3() * self.liquid_density_kg_m3

    def set_liquid_mass_kg(self, mass_kg: float) -> float:
        if mass_kg <= 0.0:
            raise DomainError("Аккумулятор осушен")
        gas_volume = self.total_volume_m3 - mass_kg / self.liquid_density_kg_m3
        if gas_volume <= 0.0:
            raise DomainError("Газовая подушка исчезла")
        self.gas_volume_m3 = gas_volume
        self.pressure_pa = self._invariant / gas_volume**self.gamma
        return self.pressure_pa
