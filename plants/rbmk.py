"""Канальная схема РБМК. Не копия коэффициентов ВВЭР и не модель разрушения."""

from __future__ import annotations

from dataclasses import dataclass

from core.status import Applicability, DomainError, ModelPassport, ModelStatus
from physics.materials.iapws import properties_ph, saturation_properties, void_fraction

PASSPORT = ModelPassport(
    model_id="rbmk-channel-lumped",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Один канал РБМК: топливо, графит, пароводяной поток и барабан",
    approximations=(
        "один эквивалентный канал",
        "графит — отдельная теплоёмкость, не среда ВВЭР",
        "пустотный коэффициент задаётся параметром этой схемы и не переносится на ВВЭР",
        "повреждение канала не моделируется",
    ),
    applicability="Равновесный двухфазный канал при давлении ниже границы области 3 IF97",
    data_sources=(
        "IF97",
        "знак и величина пустотного коэффициента — параметр паспорта, не измерение блока",
    ),
    notes=("verified — баланс энергии и отдельность графита. Не plant_referenced.",),
)


@dataclass
class RBMKChannel:
    pressure_pa: float = 6.8e6
    flow_kg_s: float = 4.5
    inlet_enthalpy_j_kg: float = 1.2e6
    fission_power_w: float = 2.0e6
    graphite_fraction: float = 0.05
    void_coefficient: float = 0.02
    alpha_ref: float = 0.2
    graphite_temperature_k: float = 700.0
    graphite_capacity_j_k: float = 5.0e6
    fuel_temperature_k: float = 900.0
    fuel_capacity_j_k: float = 2.0e5
    ua_fuel_graphite: float = 5000.0
    ua_graphite_channel: float = 8000.0

    def __post_init__(self) -> None:
        self.passport = PASSPORT
        self.applicability = Applicability.IN_DOMAIN

    def channel_state(self):
        liquid, vapor = saturation_properties(self.pressure_pa)
        # Энергия, пришедшая с питанием и делением, задаёт энтальпию смеси на выходе
        # только как диагностика стационарного качества; динамика канала считается ниже.
        h_out = (
            self.inlet_enthalpy_j_kg
            + (1.0 - self.graphite_fraction) * self.fission_power_w / self.flow_kg_s
        )
        if h_out < liquid.h_j_kg or h_out > vapor.h_j_kg:
            raise DomainError("Выход канала вне двухфазной области паспорта РБМК")
        state = properties_ph(self.pressure_pa, h_out)
        alpha = void_fraction(state.quality, liquid.rho_kg_m3, vapor.rho_kg_m3)
        return state, alpha

    def void_reactivity(self, alpha: float) -> float:
        return self.void_coefficient * (alpha - self.alpha_ref)

    def derivatives(self) -> tuple:
        _state, alpha = self.channel_state()
        q_graphite = self.graphite_fraction * self.fission_power_w
        q_fuel = self.fission_power_w - q_graphite
        q_to_graphite = self.ua_fuel_graphite * (
            self.fuel_temperature_k - self.graphite_temperature_k
        )
        q_to_channel = self.ua_graphite_channel * (self.graphite_temperature_k - _state.T_k)
        d_fuel = (q_fuel - q_to_graphite) / self.fuel_capacity_j_k
        d_graphite = (q_graphite + q_to_graphite - q_to_channel) / self.graphite_capacity_j_k
        return d_fuel, d_graphite, alpha, self.void_reactivity(alpha)

    def steady_temperatures(self) -> None:
        """Стационар тепловой структуры при заданной мощности. Канал остаётся двухфазным."""
        _state, _alpha = self.channel_state()
        q_graphite = self.graphite_fraction * self.fission_power_w
        q_fuel = self.fission_power_w - q_graphite
        # q_fuel = UA_fg (Tf - Tg), q_graphite + q_fuel = UA_gc (Tg - Tch)
        t_channel = _state.T_k
        self.graphite_temperature_k = t_channel + self.fission_power_w / self.ua_graphite_channel
        self.fuel_temperature_k = self.graphite_temperature_k + q_fuel / self.ua_fuel_graphite
