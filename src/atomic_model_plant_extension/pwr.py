"""Separate source-informed seven-bank reduced PWR configuration.

Only the aggregate absorber endpoint comes from sequential HZP measurements.
This mixed-provenance lumped model is not a BEAVRS plant reconstruction.
"""

import hashlib
import json
from dataclasses import replace
from importlib.resources import files

from plants.vver import WORTH, rod_integral_worth

from .gravity import GravityFeedPlant


class SourceBoundPWR(GravityFeedPlant):
    def __init__(self, height_m=10.0):
        root = files("atomic_model_plant_extension").joinpath("data/beavrs")
        spec = json.loads(root.joinpath("absorber.json").read_text())
        if (
            spec["commit"] != "1ccae04dc3830a5977f08ed7f5f7e1e51ded0d22"
            or hashlib.sha256(root.joinpath("expdata.tex").read_bytes()).hexdigest()
            != spec["source_sha256"]
            or [r["worth_pcm"] for r in spec["sequential_increments"]]
            != [788, 1203, 1171, 548, 461, 772, 1099]
            or spec["aggregate_worth_pcm"] != 6042
            or spec["aggregate_worth"] != 6042 * 1e-5
        ):
            raise ValueError("Absorber provenance/configuration mismatch")
        self.absorber_worth = spec["aggregate_worth"]
        self.absorber_source = spec
        super().__init__(height_m)
        self.passport = replace(
            self.passport,
            model_id="pwr-reduced-seven-bank",
            version="1.0.0",
            data_sources=(
                *self.passport.data_sources,
                "MIT BEAVRS cycle1 HZP sequential bank increments D,C,B,A,SE,SD,SC; commit"
                + spec["commit"],
            ),
            approximations=(
                *self.passport.approximations,
                "Aggregate absorber endpoint6042pcm; sine-profile interpolation remains a declared reduced assumption",
                "Not a BEAVRS or VVER plant-reference configuration; other nuclear coefficients remain illustrative",
                "Declared reactor-trip interlock closes steam throttle with priority over operator throttle request",
            ),
        )

    def reactivity(self):
        result = super().reactivity()
        return result - (self.absorber_worth - WORTH) * (
            rod_integral_worth(self.rod) - rod_integral_worth(self.rod_ref)
        )

    def _apply_heat_and_actuators(self):
        # Declared protection configuration: reactor trip interlocks the steam
        # throttle closed. The requested operator command remains observable.
        requested = self.throttle
        effective = 0.0 if self.scram_latched else requested
        try:
            self.throttle = effective
            super()._apply_heat_and_actuators()
        finally:
            self.throttle = requested
        self.effective_steam_throttle = effective

    def _hydraulic_torque(self):
        pump = self.network.junctions["pump"]
        if pump.mass_flow_kg_s < 0:
            from core.status import DomainError

            raise DomainError("Primary pump reverse quadrant unsupported")
        density = self.network.volumes["cold_leg"].thermo.rho_kg_m3
        pump.pump_pressure_pa = self.dp_pump_nom * (self.omega / self.omega_nom) ** 2
        return pump.mass_flow_kg_s * self.dp_pump_nom * self.omega / (density * self.omega_nom**2)

    def derivatives(self, time, state):
        rate = super().derivatives(time, state)
        torque = self._hydraulic_torque()
        rate[11] = (
            (self.drive_torque if self.motor_on else 0.0) - torque - self.loss_coef * self.omega
        ) / self.inertia
        return rate
