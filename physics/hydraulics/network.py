"""Сеть контрольных объёмов и соединений.

Опорные уравнения объёма:
dm/dt = sum m_in - sum m_out
dU/dt = sum m_donor * h_donor + Q + W

Соединение:
(L/A) dm/dt = p_up - p_down - rho g (z_down - z_up) - dp_friction + dp_pump

Расход может менять знак. Ограничение max(m, 0) не используется.
Насосный напор при обратном расходе — out_of_domain, а не четырёхквадрантная догадка.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.status import Applicability, DomainError
from physics.materials.iapws import (
    ThermoState,
    properties_ph,
    properties_pt,
    properties_rho_u,
    saturation_properties,
)

G = 9.80665


@dataclass
class BoundaryState:
    pressure_pa: float
    enthalpy_j_kg: float
    density_kg_m3: float
    elevation_m: float
    boron_fraction: float = 0.0


@dataclass
class Volume:
    name: str
    volume_m3: float
    elevation_m: float
    mass_kg: float = 1.0
    internal_energy_j: float = 1.0
    boron_kg: float = 0.0
    pressure_pa: float = 1.0e5
    temperature_k: float = 300.0
    heat_w: float = 0.0
    thermo: Optional[ThermoState] = None
    energy_mismatch_j_kg: float = 0.0
    volume_mismatch: float = 0.0

    def set_ph(self, pressure_pa: float, enthalpy_j_kg: float, boron_ppm: float = 0.0) -> None:
        state = properties_ph(pressure_pa, enthalpy_j_kg)
        self.pressure_pa = state.p_pa
        self.temperature_k = state.T_k
        self.thermo = state
        self.mass_kg = state.rho_kg_m3 * self.volume_m3
        self.internal_energy_j = self.mass_kg * state.u_j_kg
        self.boron_kg = boron_ppm * 1.0e-6 * self.mass_kg

    def set_liquid(self, pressure_pa: float, temperature_k: float, boron_ppm: float = 0.0) -> None:
        state = properties_pt(pressure_pa, temperature_k)
        self.pressure_pa = pressure_pa
        self.temperature_k = temperature_k
        self.thermo = state
        self.mass_kg = state.rho_kg_m3 * self.volume_m3
        self.internal_energy_j = self.mass_kg * state.u_j_kg
        self.boron_kg = boron_ppm * 1.0e-6 * self.mass_kg

    def set_quality(self, pressure_pa: float, quality: float) -> None:
        liquid, vapor = saturation_properties(pressure_pa)
        state = properties_ph(pressure_pa, (1.0 - quality) * liquid.h_j_kg + quality * vapor.h_j_kg)
        self.pressure_pa = state.p_pa
        self.temperature_k = state.T_k
        self.thermo = state
        self.mass_kg = state.rho_kg_m3 * self.volume_m3
        self.internal_energy_j = self.mass_kg * state.u_j_kg

    def sync(self) -> ThermoState:
        if self.mass_kg <= 0.0:
            raise DomainError(
                "Объём %s осушен: исчезновение фазы без отдельной модели не продолжается"
                % self.name
            )
        density = self.mass_kg / self.volume_m3
        energy = self.internal_energy_j / self.mass_kg
        try:
            state = properties_rho_u(density, energy, self.pressure_pa, self.temperature_k)
        except DomainError as exc:
            raise DomainError(
                "Объём %s: %s (rho=%.6g, u=%.6g, p_guess=%.6g, T_guess=%.3f)"
                % (self.name, exc, density, energy, self.pressure_pa, self.temperature_k)
            ) from exc
        self.thermo = state
        self.pressure_pa = state.p_pa
        self.temperature_k = state.T_k
        self.energy_mismatch_j_kg = state.energy_mismatch_j_kg
        self.volume_mismatch = state.volume_mismatch
        return state

    @property
    def boron_fraction(self) -> float:
        return 0.0 if self.mass_kg <= 0.0 else self.boron_kg / self.mass_kg


@dataclass
class Junction:
    name: str
    upstream: str
    downstream: str
    area_m2: float
    length_m: float
    loss_coefficient: float
    mass_flow_kg_s: float = 0.0
    pump_pressure_pa: float = 0.0
    enthalpy_mode: str = "donor"
    delivery_enthalpy_j_kg: Optional[float] = None
    boron_carry: float = 1.0
    flow_law: str = "momentum"
    flow_target_kg_s: Optional[float] = None
    flow_time_constant_s: float = 0.5

    def friction_dp(self, density: float) -> float:
        if density <= 0.0:
            raise DomainError("Плотность в соединении %s не положительна" % self.name)
        dynamic = self.mass_flow_kg_s * abs(self.mass_flow_kg_s) / (2.0 * density * self.area_m2**2)
        return self.loss_coefficient * dynamic


@dataclass
class FlowNetwork:
    volumes: Dict[str, Volume]
    junctions: Dict[str, Junction]
    boundaries: Dict[str, BoundaryState] = field(default_factory=dict)
    applicability: Applicability = Applicability.IN_DOMAIN
    message: str = ""

    def volume_names(self) -> List[str]:
        return list(self.volumes.keys())

    def junction_names(self) -> List[str]:
        return list(self.junctions.keys())

    def pack(self) -> np.ndarray:
        values = []
        for name in self.volume_names():
            volume = self.volumes[name]
            values.extend([volume.mass_kg, volume.internal_energy_j, volume.boron_kg])
        for name in self.junction_names():
            values.append(self.junctions[name].mass_flow_kg_s)
        return np.array(values, dtype=float)

    def unpack(self, state: np.ndarray) -> None:
        cursor = 0
        for name in self.volume_names():
            volume = self.volumes[name]
            volume.mass_kg = float(state[cursor])
            volume.internal_energy_j = float(state[cursor + 1])
            volume.boron_kg = float(state[cursor + 2])
            cursor += 3
        for name in self.junction_names():
            self.junctions[name].mass_flow_kg_s = float(state[cursor])
            cursor += 1

    def _node(self, name: str):
        if name in self.volumes:
            volume = self.volumes[name]
            if volume.thermo is None:
                volume.sync()
            return volume
        if name in self.boundaries:
            return self.boundaries[name]
        raise DomainError("Узел %s не объявлен" % name)

    def refresh_properties(self) -> None:
        for volume in self.volumes.values():
            volume.sync()

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        self.unpack(state)
        self.applicability = Applicability.IN_DOMAIN
        self.message = ""
        self.refresh_properties()
        dm = {name: 0.0 for name in self.volumes}
        dU = {name: volume.heat_w for name, volume in self.volumes.items()}
        dB = {name: 0.0 for name in self.volumes}
        dmdot = []
        for name in self.junction_names():
            junction = self.junctions[name]
            up = self._node(junction.upstream)
            down = self._node(junction.downstream)
            flow = junction.mass_flow_kg_s
            if flow >= 0.0:
                donor, receiver_name = up, junction.downstream
            else:
                donor, receiver_name = down, junction.upstream
            density = (
                donor.density_kg_m3 if isinstance(donor, BoundaryState) else donor.thermo.rho_kg_m3
            )
            enthalpy = (
                donor.enthalpy_j_kg if isinstance(donor, BoundaryState) else donor.thermo.h_j_kg
            )
            donor_pressure = (
                donor.pressure_pa if isinstance(donor, BoundaryState) else donor.thermo.p_pa
            )
            if junction.enthalpy_mode != "donor":
                liquid, vapor = saturation_properties(donor_pressure)
                if junction.enthalpy_mode == "saturated_vapor":
                    if isinstance(donor, Volume) and donor.thermo.quality <= 0.0:
                        raise DomainError(
                            "Отбор насыщенного пара из однофазной жидкости в %s" % name
                        )
                    enthalpy = vapor.h_j_kg
                elif junction.enthalpy_mode == "saturated_liquid":
                    enthalpy = liquid.h_j_kg
                else:
                    raise DomainError("Режим энтальпии %s не поддержан" % junction.enthalpy_mode)
            arrive = (
                enthalpy
                if junction.delivery_enthalpy_j_kg is None
                else junction.delivery_enthalpy_j_kg
            )
            boron = (
                donor.boron_fraction if isinstance(donor, BoundaryState) else donor.boron_fraction
            ) * junction.boron_carry
            signed = flow
            if junction.upstream in self.volumes:
                dm[junction.upstream] -= signed
                leaving = self.volumes[junction.upstream].boron_fraction if signed >= 0.0 else boron
                dB[junction.upstream] -= signed * leaving
            if junction.downstream in self.volumes:
                dm[junction.downstream] += signed
                arriving = (
                    boron if signed >= 0.0 else self.volumes[junction.downstream].boron_fraction
                )
                dB[junction.downstream] += signed * arriving
            transported = abs(flow)
            if flow >= 0.0:
                if junction.upstream in self.volumes:
                    dU[junction.upstream] -= transported * enthalpy
                if junction.downstream in self.volumes:
                    dU[junction.downstream] += transported * arrive
            else:
                if junction.downstream in self.volumes:
                    dU[junction.downstream] -= transported * enthalpy
                if junction.upstream in self.volumes:
                    dU[junction.upstream] += transported * arrive
            if junction.pump_pressure_pa != 0.0 and flow < 0.0:
                raise DomainError(
                    "Обратный расход через насос %s вне проверенного квадранта" % name
                )
            if (
                flow >= 0.0
                and junction.pump_pressure_pa != 0.0
                and junction.downstream in self.volumes
            ):
                dU[junction.downstream] += flow * junction.pump_pressure_pa / density
            if junction.flow_law == "momentum":
                dp_fric = junction.friction_dp(density)
                z_up = up.elevation_m
                z_down = down.elevation_m
                p_up = up.pressure_pa if isinstance(up, BoundaryState) else up.thermo.p_pa
                p_down = down.pressure_pa if isinstance(down, BoundaryState) else down.thermo.p_pa
                gravity = density * G * (z_down - z_up)
                accel = (junction.area_m2 / junction.length_m) * (
                    p_up - p_down - gravity - dp_fric + junction.pump_pressure_pa
                )
            elif junction.flow_law == "target":
                if junction.flow_target_kg_s is None:
                    raise DomainError("Для соединения %s не задан целевой расход" % name)
                accel = (junction.flow_target_kg_s - flow) / junction.flow_time_constant_s
            else:
                raise DomainError("Закон расхода %s не поддержан" % junction.flow_law)
            dmdot.append(accel)
            del receiver_name
        derivatives = []
        for name in self.volume_names():
            derivatives.extend([dm[name], dU[name], dB[name]])
        derivatives.extend(dmdot)
        return np.array(derivatives, dtype=float)

    def total_mass(self) -> float:
        return float(sum(volume.mass_kg for volume in self.volumes.values()))

    def total_energy(self) -> float:
        return float(sum(volume.internal_energy_j for volume in self.volumes.values()))

    def total_boron(self) -> float:
        return float(sum(volume.boron_kg for volume in self.volumes.values()))
