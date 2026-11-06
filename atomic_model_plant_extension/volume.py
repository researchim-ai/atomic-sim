"""Strict stock inversion for network inventories, unchanged IF97 coefficients."""

from dataclasses import replace

from core.status import DomainError
from physics.hydraulics.network import Volume
from physics.materials import iapws

from . import water


class StrictVolume(Volume):
    def sync(self):
        if self.mass_kg <= 0:
            raise DomainError("Nonpositive strict water inventory")
        density = float(self.mass_kg / self.volume_m3)
        energy = float(self.internal_energy_j / self.mass_kg)
        key = (density, energy)
        if getattr(self, "_thermo_key", None) == key and self.thermo is not None:
            return self.thermo
        try:
            if not hasattr(self, "_inverse_seed"):
                self._inverse_seed = (float(self.pressure_pa), float(self.temperature_k))
            point = water.rho_u(density, energy, *self._inverse_seed)
            forward = iapws.properties_ph(point.pressure_pa, point.enthalpy_j_kg)
        except ValueError as exc:
            raise DomainError(f"Strict inventory{self.name}: {exc}") from exc
        self.energy_mismatch_j_kg = energy - point.internal_energy_j_kg
        self.volume_mismatch = 1 / density - 1 / point.density_kg_m3
        self.thermo = replace(
            forward,
            T_k=point.temperature_k,
            p_pa=point.pressure_pa,
            rho_kg_m3=density,
            v_m3_kg=1 / density,
            u_j_kg=energy,
            h_j_kg=energy + point.pressure_pa / density,
            s_j_kg_k=point.entropy_j_kg_k,
            energy_mismatch_j_kg=self.energy_mismatch_j_kg,
            volume_mismatch=self.volume_mismatch,
        )
        self.pressure_pa = point.pressure_pa
        self.temperature_k = point.temperature_k
        self._thermo_key = key
        return self.thermo
