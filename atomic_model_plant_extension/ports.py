"""Bottom liquid port of an equilibrium water/steam pressurizer.

Phase selection affects enthalpy, density and dissolved nonvolatile boron,
while mass and energy still belong to the real stored volume.
"""

from core.status import DomainError
from physics.hydraulics.network import BoundaryState, FlowNetwork

from . import water


class PhasePortNetwork(FlowNetwork):
    def _node(self, name):
        node = super()._node(name)
        if name != "pressurizer":
            return node
        state = node.thermo
        x = float(state.quality)
        if state.region == 1:
            return BoundaryState(
                float(state.p_pa),
                float(state.h_j_kg),
                float(state.rho_kg_m3),
                node.elevation_m,
                node.boron_kg / node.mass_kg,
            )
        if state.region != 4 or not 0 <= x < 1:
            raise DomainError("Pressurizer bottom liquid port is dry or outside equilibrium scope")
        liquid, _ = water.saturation(float(state.p_pa))
        liquid_mass = node.mass_kg * (1 - x)
        concentration = node.boron_kg / liquid_mass
        if not 0 <= concentration < 0.05:
            raise DomainError("Liquid boron concentration outside dilute-solution scope")
        return BoundaryState(
            float(state.p_pa),
            liquid.enthalpy_j_kg,
            liquid.density_kg_m3,
            node.elevation_m,
            float(concentration),
        )
