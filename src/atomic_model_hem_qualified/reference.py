"""Independent absolute-stock RHS/drag; shared IF97 EOS, no production flux/stepper.

Uses the previously frozen independent graph and drag assembly. Chen arithmetic
is evaluated here separately; CHF data are shared and are not experimental truth.
"""

from copy import deepcopy
from math import pi

import numpy as np
from scipy.integrate import solve_ivp

from atomic_model_hem_mixture.reference import Assembly as MixtureReference
from atomic_model_hem_qualified import eos
from physics.materials import iapws


class Assembly(MixtureReference):
    def __init__(self, request):
        super().__init__(dict(model=request["model"], mixture_friction=request["friction"]))

    @staticmethod
    def _state(point):
        if point["kind"] == "pt":
            return eos.pt(point["pressure_pa"], point["temperature_k"])
        if point["kind"] == "px":
            return eos.px(point["pressure_pa"], point["quality"])
        if point["kind"] == "rho_u":
            return eos.rho_u(
                point["density_kg_m3"],
                point["internal_energy_j_kg"],
                point["pressure_seed_pa"],
                point["temperature_seed_k"],
            )
        raise ValueError("Unknown reference chart")

    def thermos(self, y):
        if self._stage_y is y and self._stage_states is not None:
            return self._stage_states
        result = [
            eos.rho_u(
                float(y[i] / self.volumes[i]),
                float(y[2 * self.n + i] / y[i] - (y[self.n + i] / y[i]) ** 2 / 2),
                pressure_seed_pa=s.pressure_pa,
                temperature_seed_k=s.temperature_k,
            )
            for i, s in enumerate(self.seeds)
        ]
        if self._stage_y is y:
            self._stage_states = result
        return result

    def wall_conductance(self, wall, y, state):
        if "boiling" not in wall:
            return wall["conductance_w_k"]
        edge = next(e for e in self.spec["edges"] if wall["cell"].startswith(e["id"] + "/"))
        i = self.names.index(wall["cell"])
        j = self.walls.index(wall)
        tw = y[self.wall_start + j] / wall["capacity_j_k"]
        liquid, gas = iapws.saturation_properties(state.pressure_pa)
        x = state.quality
        G = state.density_kg_m3 * y[self.n + i] / y[i]
        diameter = edge["diameter_m"]
        reynolds = G * (1 - x) * diameter / liquid.viscosity_pa_s
        prandtl = liquid.cp_j_kg_k * liquid.viscosity_pa_s / liquid.conductivity_w_m_k
        martinelli_inverse = (
            (x / (1 - x)) ** 0.9
            * (liquid.rho_kg_m3 / gas.rho_kg_m3) ** 0.5
            * (gas.viscosity_pa_s / liquid.viscosity_pa_s) ** 0.1
            if x
            else 0
        )
        enhancement = (
            1 if martinelli_inverse <= 0.1 else 2.35 * (0.213 + martinelli_inverse) ** 0.736
        )
        suppression = (1 + 2.53e-6 * (reynolds * enhancement**1.25) ** 1.17) ** -1
        convection = 0.023 * reynolds**0.8 * prandtl**0.4 * liquid.conductivity_w_m_k / diameter
        tau = 1 - state.temperature_k / 647.096
        sigma = 0.2358 * tau**1.256 * (1 - 0.625 * tau)
        nucleation = (
            0.00122
            * (liquid.conductivity_w_m_k**0.79 * liquid.cp_j_kg_k**0.45 * liquid.rho_kg_m3**0.49)
            / (
                sigma**0.5
                * liquid.viscosity_pa_s**0.29
                * (gas.h_j_kg - liquid.h_j_kg) ** 0.25
                * gas.rho_kg_m3**0.24
            )
        )
        nucleation *= (tw - state.temperature_k) ** 0.24 * (
            iapws.saturation_pressure_mpa(tw) * 1e6 - state.pressure_pa
        ) ** 0.75
        return float(
            (enhancement * convection + suppression * nucleation)
            * pi
            * diameter
            * edge["length_m"]
            / edge["cells"]
        )


def solve(request, program, method="DOP853", *, max_step=5e-5):
    assembly = Assembly(request)
    y, time = assembly.y0.copy(), 0.0
    observations = [assembly.observation(time, y)]
    for call in program:
        if call["command"] is not None:
            assembly.controls = deepcopy(call["command"]["controls"])
        target = time + call["duration_s"]
        result = solve_ivp(
            assembly.rhs,
            (time, target),
            y,
            method=method,
            rtol=2e-11,
            atol=1e-10,
            max_step=max_step,
        )
        if not result.success or not np.all(np.isfinite(result.y)):
            raise ValueError("Qualified reference IVP failed")
        y, time = result.y[:, -1], target
        observations.append(assembly.observation(time, y))
    return dict(method=method, evaluations=assembly.evaluations, observations=observations)
