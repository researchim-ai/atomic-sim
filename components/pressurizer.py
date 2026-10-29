"""Компенсатор давления: нагреватель, впрыск и приток по линии связи.

Давление берётся из запасов объёма. Нагреватель только подводит тепло.
Впрыск — жидкость. Петля ВВЭР применяет те же знаки на своём объёме компенсатора.
"""

from __future__ import annotations

from core.status import DomainError, ModelPassport, ModelStatus
from physics.materials.iapws import saturation_properties

from components.condenser import DynamicCondenser

PASSPORT = ModelPassport(
    model_id="pressurizer-heater-spray",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Двухфазный компенсатор с нагревателем и впрыском",
    approximations=("равновесное состояние IF97", "впрыск не выше энтальпии насыщенной жидкости", "один геометрический объём"),
    applicability="0 < x < 1 после восстановления состояния; осушение останавливает расчёт",
    data_sources=("Балансы массы и энергии контрольного объёма",),
)


class Pressurizer:
    passport = PASSPORT

    def __init__(self, volume_m3: float, pressure_pa: float, quality: float):
        self.vessel = DynamicCondenser.from_quality(volume_m3, pressure_pa, quality)

    def derivatives(
        self,
        heater_w: float,
        spray_kg_s: float,
        spray_enthalpy_j_kg: float,
        surge_kg_s: float,
        surge_enthalpy_j_kg: float,
    ) -> tuple:
        if heater_w < 0.0:
            raise DomainError("Нагреватель компенсатора не отводит тепло")
        if spray_kg_s < 0.0:
            raise DomainError("Впрыск задаётся неотрицательным расходом")
        liquid, _vapor = saturation_properties(self.vessel.state().p_pa)
        if spray_kg_s > 0.0 and spray_enthalpy_j_kg > liquid.h_j_kg + 1.0:
            raise DomainError("Впрыск горячее насыщенной жидкости этого давления")
        mass_rate = spray_kg_s + surge_kg_s
        energy_rate = (
            heater_w
            + spray_kg_s * spray_enthalpy_j_kg
            + surge_kg_s * surge_enthalpy_j_kg
        )
        return mass_rate, energy_rate
