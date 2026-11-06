"""Steady operating-point agent task; a decision is not a physical time step."""

import numpy as np

from .coupled import SpatialSession
from .provenance import seal, unseal


class SpatialAgent:
    def __init__(self, config, feedback):
        self.session = SpatialSession(config, feedback)

    def step(self, action, *, expected_revision):
        if not isinstance(action, dict) or set(action) != {"thermal_power_w"}:
            raise ValueError("Only declared thermal-power action supported")
        power = action["thermal_power_w"]
        if type(power) not in (int, float) or not np.isfinite(power) or not 100 <= power <= 10000:
            raise ValueError("Agent power domain100..10000W")
        result = self.session.advance(power, expected_revision=expected_revision)
        peak = max(result["fuel_temperature_k"])
        return dict(
            observation=result,
            reward=-peak / 1000,
            revision=self.session.revision,
            fidelity=result["fidelity"],
            time_semantics="steady operating-point sequence; no transient time",
            accepted_scope="numerical synthetic-feedback slab/rod/channel",
            validated=False,
            plant_referenced=False,
        )

    def checkpoint(self):
        return seal(dict(session=self.session.checkpoint(), agent="steady-spatial-power-v1"))

    @classmethod
    def restore(cls, saved):
        p = unseal(saved)
        if p["agent"] != "steady-spatial-power-v1":
            raise ValueError("Wrong spatial agent passport")
        result = object.__new__(cls)
        result.session = SpatialSession.restore(p["session"])
        return result
