"""Барабан-сепаратор: пар уходит насыщенным паром, вода — насыщенной жидкостью."""

from __future__ import annotations

from dataclasses import dataclass

from core.status import DomainError, ModelPassport, ModelStatus
from physics.materials.iapws import saturation_properties

PASSPORT = ModelPassport(
    model_id="drum-separator",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Разделение фаз в барабане с уровнем",
    approximations=("идеальное разделение", "равновесие при давлении барабана", "унос капель не моделируется"),
    applicability="Есть обе фазы; осушение или сплошное заполнение паром — out_of_domain",
    data_sources=("Балансы массы и энергии контрольного объёма",),
)


@dataclass
class SteamDrum:
    area_m2: float
    volume_m3: float
    pressure_pa: float
    liquid_mass_kg: float
    vapor_mass_kg: float

    def level_m(self) -> float:
        _liquid, _vapor = saturation_properties(self.pressure_pa)
        if self.liquid_mass_kg < 0.0 or self.vapor_mass_kg < 0.0:
            raise DomainError("Масса фазы барабана стала отрицательной")
        return self.liquid_mass_kg / (_liquid.rho_kg_m3 * self.area_m2)

    def step(
        self,
        feed_kg_s: float,
        feed_enthalpy_j_kg: float,
        steam_kg_s: float,
        liquid_drain_kg_s: float,
        heat_w: float,
        dt_s: float,
    ) -> None:
        liquid, vapor = saturation_properties(self.pressure_pa)
        if steam_kg_s < 0.0 or liquid_drain_kg_s < 0.0 or feed_kg_s < 0.0:
            raise DomainError("Расходы барабана задаются неотрицательными; реверс — отдельный режим")
        dm_liquid = (feed_kg_s - liquid_drain_kg_s) * dt_s
        dm_vapor = -steam_kg_s * dt_s
        if self.liquid_mass_kg + dm_liquid <= 0.0 or self.vapor_mass_kg + dm_vapor <= 0.0:
            raise DomainError("Барабан теряет фазу: модель идеального разделения здесь останавливается")
        self.liquid_mass_kg += dm_liquid
        self.vapor_mass_kg += dm_vapor
        energy = self.liquid_mass_kg * liquid.u_j_kg + self.vapor_mass_kg * vapor.u_j_kg
        # Энергия до шага была у старых масс; пересчитываем приращение явно.
        energy = (
            (self.liquid_mass_kg - dm_liquid) * liquid.u_j_kg
            + (self.vapor_mass_kg - dm_vapor) * vapor.u_j_kg
            + feed_kg_s * feed_enthalpy_j_kg * dt_s
            - steam_kg_s * vapor.h_j_kg * dt_s
            - liquid_drain_kg_s * liquid.h_j_kg * dt_s
            + heat_w * dt_s
        )
        # Давление в этой сосредоточенной модели удерживается внешней системой.
        # Проверка вызывающего кода — сравнение energy с суммой u*m после шага плюс работа расширения.
        self._energy_j = energy

    @property
    def accounted_energy_j(self) -> float:
        return self._energy_j
