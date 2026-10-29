"""Конденсатор фиксированного геометрического объёма.

Давление восстанавливается из плотности и удельной энергии, а не удерживается
заданной константой. Петля ВВЭР держит такой объём: в номинале это двухфазная
смесь, и давление оболочки следует из (rho, u).
"""

from __future__ import annotations

from core.status import DomainError, ModelPassport, ModelStatus
from physics.materials.iapws import properties_ph, properties_pt, properties_rho_u, saturation_properties

PASSPORT = ModelPassport(
    model_id="dynamic-condenser",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Контрольный объём конденсатора с давлением из уравнения состояния",
    approximations=("геометрический объём постоянен", "равновесное состояние IF97", "теплоотвод задаёт вызывающий код"),
    applicability="Масса положительна и (rho, u) лежит в реализованной области IF97",
    data_sources=("IAPWS-IF97, обращение (rho, u)",),
)


class DynamicCondenser:
    passport = PASSPORT

    def __init__(self, volume_m3: float, pressure_pa: float, temperature_k: float):
        if volume_m3 <= 0.0:
            raise DomainError("Объём конденсатора должен быть положительным")
        state = properties_pt(pressure_pa, temperature_k)
        self.volume_m3 = float(volume_m3)
        self.mass_kg = state.rho_kg_m3 * self.volume_m3
        self.energy_j = self.mass_kg * state.u_j_kg
        self._pressure_guess_pa = state.p_pa
        self._temperature_guess_k = state.T_k

    @classmethod
    def from_quality(cls, volume_m3: float, pressure_pa: float, quality: float) -> "DynamicCondenser":
        if not 0.0 < quality < 1.0:
            raise DomainError("Двухфазный конденсатор задаётся при 0 < x < 1")
        liquid, vapor = saturation_properties(pressure_pa)
        enthalpy = (1.0 - quality) * liquid.h_j_kg + quality * vapor.h_j_kg
        state = properties_ph(pressure_pa, enthalpy)
        condenser = cls.__new__(cls)
        condenser.volume_m3 = float(volume_m3)
        condenser.mass_kg = state.rho_kg_m3 * condenser.volume_m3
        condenser.energy_j = condenser.mass_kg * state.u_j_kg
        condenser._pressure_guess_pa = state.p_pa
        condenser._temperature_guess_k = state.T_k
        return condenser

    def state(self):
        if self.mass_kg <= 0.0:
            raise DomainError("Конденсатор осушен")
        recovered = properties_rho_u(
            self.mass_kg / self.volume_m3,
            self.energy_j / self.mass_kg,
            self._pressure_guess_pa,
            self._temperature_guess_k,
        )
        self._pressure_guess_pa = recovered.p_pa
        self._temperature_guess_k = recovered.T_k
        return recovered

    def derivatives(
        self,
        steam_kg_s: float,
        steam_enthalpy_j_kg: float,
        feed_kg_s: float,
        feed_enthalpy_j_kg: float,
        heat_w: float,
    ) -> tuple:
        return (
            steam_kg_s - feed_kg_s,
            steam_kg_s * steam_enthalpy_j_kg - feed_kg_s * feed_enthalpy_j_kg + heat_w,
        )
