"""Versioned elevated reduced plant with explicit potential and kinetic accounts.

Fixed reference-density hydraulic kinetic storage is a declared lumped closure.
Water inventory density remains dynamic IF97. No primary boiling promotion.
"""

from dataclasses import replace

from physics.hydraulics.network import BoundaryState, G

from .one_way import OneWayFeedPlant


class GravityFeedPlant(OneWayFeedPlant):
    def __init__(self, height_m=10.0):
        if type(height_m) not in (int, float) or not 0 <= height_m <= 20:
            raise ValueError("Declared reduced geometry height0..20m")
        super().__init__()
        self.height_m = float(height_m)
        heights = dict(
            core=0.0,
            hot_leg=0.5 * height_m,
            sg_primary=height_m,
            cold_leg=0.0,
            pressurizer=height_m,
        )
        for name, z in heights.items():
            self.network.volumes[name].elevation_m = z
        self.primary_links = ["pump", "core_out", "to_sg", "to_cold", "surge", "spray"]
        self.link_density_ref = {
            name: float(
                self.network.volumes[self.network.junctions[name].upstream].thermo.rho_kg_m3
            )
            for name in self.primary_links
        }
        self.passport = replace(
            self.passport,
            model_id="pwr-reduced-elevated-feed",
            version="1.0.0",
            approximations=(
                *self.passport.approximations,
                "versioned illustrative elevated geometry, not surveyed VVER plant geometry",
                "primary hydraulic kinetic storage uses initial reference densities; exact compensating thermal exchange",
                "gravity transported with enthalpy; direction-independent arithmetic endpoint port density in hydrostatics",
            ),
        )
        self.refresh_algebraic(self.pack())

    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not hasattr(self, "link_density_ref"):
            return
        for name in self.primary_links:
            link = self.network.junctions[name]
            up = self.network.volumes[link.upstream]
            down = self.network.volumes[link.downstream]
            donor = self.network._node(
                link.upstream if link.mass_flow_kg_s >= 0 else link.downstream
            )
            dz = down.elevation_m - up.elevation_m
            direction = 1 if link.mass_flow_kg_s >= 0 else -1
            enthalpy = (
                donor.enthalpy_j_kg if isinstance(donor, BoundaryState) else donor.thermo.h_j_kg
            )
            link.delivery_enthalpy_j_kg = enthalpy - G * direction * dz

    def derivatives(self, time, state):
        rate = super().derivatives(time, state)
        offset = 13 + 3 * len(self.network.volumes)
        names = self.network.volume_names()
        # Hydrostatic head must be continuous at flow reversal. Upwind donor
        # density is a transport choice, not a static column density law.
        for name in self.primary_links:
            link = self.network.junctions[name]
            if link.flow_law != "momentum":
                continue
            up = self.network._node(link.upstream)
            down = self.network._node(link.downstream)

            def rho(node):
                return (
                    node.density_kg_m3 if isinstance(node, BoundaryState) else node.thermo.rho_kg_m3
                )

            left, right = rho(up), rho(down)
            donor = left if link.mass_flow_kg_s >= 0 else right
            average = 0.5 * (left + right)
            dz = down.elevation_m - up.elevation_m
            j = self.network.junction_names().index(name)
            rate[offset + j] += (link.area_m2 / link.length_m) * (donor - average) * G * dz
        for name in self.primary_links:
            link = self.network.junctions[name]
            j = self.network.junction_names().index(name)
            inertia = link.length_m / link.area_m2
            work = inertia * link.mass_flow_kg_s * rate[offset + j] / self.link_density_ref[name]
            # Mechanical energy exchange is internal, not another heater.
            rate[13 + 3 * names.index(link.upstream) + 1] -= 0.5 * work
            rate[13 + 3 * names.index(link.downstream) + 1] -= 0.5 * work
        return rate

    def stored_accounts_j(self):
        result = super().stored_accounts_j()
        if hasattr(self, "link_density_ref"):
            result["primary_pipe_kinetic_j"] = sum(
                0.5 * j.length_m / j.area_m2 * j.mass_flow_kg_s**2 / self.link_density_ref[n]
                for n in self.primary_links
                for j in [self.network.junctions[n]]
            )
            result["inventory_gravitational_j"] = sum(
                v.mass_kg * G * v.elevation_m for v in self.network.volumes.values()
            )
        return result

    def account_rates_w(self, dot):
        result = super().account_rates_w(dot)
        offset = 13 + 3 * len(self.network.volumes)
        result["primary_pipe_kinetic_w"] = sum(
            j.length_m
            / j.area_m2
            * j.mass_flow_kg_s
            * float(dot[offset + self.network.junction_names().index(n)])
            / self.link_density_ref[n]
            for n in self.primary_links
            for j in [self.network.junctions[n]]
        )
        result["inventory_gravitational_w"] = sum(
            G * v.elevation_m * float(dot[13 + 3 * i])
            for i, v in enumerate(self.network.volumes.values())
        )
        return result
