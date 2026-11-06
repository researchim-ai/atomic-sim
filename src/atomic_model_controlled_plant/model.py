"""Three-element feed control acting on head, with real fluid inertance.

Ideal pressure/flow measurements and instantaneous head actuation are declared
reduced assumptions. No feed inventory is projected to a requested flow.
"""

from dataclasses import replace

from atomic_model_plant_extension.pwr import SourceBoundPWR


class ControlledPWR(SourceBoundPWR):
    maximum_feed_head_pa = 12e6

    def __init__(self, height_m=10.0):
        super().__init__(height_m)
        self.passport = replace(
            self.passport,
            model_id="pwr-reduced-flow-controlled",
            version="1.0.0",
            phenomenon="Reduced research PWR loop with controlled feed, finite inventories and point kinetics",
            approximations=(
                *(
                    note
                    for note in self.passport.approximations
                    if not note.startswith("узлы на одной отметке:")
                    and not note.startswith("питание — кривая напора")
                ),
                "Feed controller: steam+dump mass flow, hotwell mass error, measured feed-flow feedback",
                "Head feedforward cancels backpressure/friction; proportional gain I/tau uses original feed tau0.2s",
                "Head command limited0..12MPa; ideal sensing and instantaneous head actuation, no electrical efficiency model",
                "Separate designed controller, not measured or licensed plant-control logic",
            ),
        )

    def _apply_heat_and_actuators(self):
        super()._apply_heat_and_actuators()
        if not hasattr(self, "feed_density_ref"):
            return
        feed = self.network.junctions["feed"]
        source = self.network.boundaries["condenser"]
        _, friction, back = self.feed_terms()
        demand = max(
            0.0,
            self.network.junctions["steam"].mass_flow_kg_s
            + self.network.junctions["dump"].mass_flow_kg_s
            + self.feed_level_gain * (self.hotwell_mass_kg - self.condenser_mass_ref_kg)
            + self.feed_bias_kg_s,
        )
        gain = self.feed_inertance / feed.flow_time_constant_s
        requested = back + friction + gain * (demand - feed.mass_flow_kg_s)
        delivered = min(self.maximum_feed_head_pa, max(0.0, requested))
        feed.pump_pressure_pa = delivered
        feed.flow_target_kg_s = demand
        feed.delivery_enthalpy_j_kg = (
            source.enthalpy_j_kg
            + (back + friction) / self.feed_density_ref
            - delivered / source.density_kg_m3
        )
        self.feed_controller = dict(
            requested_flow_kg_s=demand,
            gain_pa_s_kg=gain,
            requested_head_pa=requested,
            delivered_head_pa=delivered,
            saturated=bool(delivered != requested),
        )

    def observation(self):
        result = super().observation()
        result["feed_controller"] = dict(self.feed_controller)
        return result
