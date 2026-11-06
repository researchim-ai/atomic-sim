"""Absolute-stock IVP using the earlier independent graph RHS and own drag law."""

from copy import deepcopy
from math import exp, log, log10

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from atomic_model_hem_mixture.inverse import conserved
from atomic_model_hem_network.reference import Assembly as GraphReference
from physics.materials import iapws


class Assembly(GraphReference):
    def __init__(self, request):
        self.laws = deepcopy(request["mixture_friction"])
        self._stage_y = None
        self._stage_states = None
        self.evaluations = 0
        super().__init__(request["model"])

    def thermos(self, y):
        if self._stage_y is y and self._stage_states is not None:
            return self._stage_states
        states = [
            conserved(
                float(y[i] / self.volumes[i]),
                float(y[2 * self.n + i] / y[i] - 0.5 * (y[self.n + i] / y[i]) ** 2),
                s.pressure_pa,
                s.temperature_k,
            )
            for i, s in enumerate(self.seeds)
        ]
        if self._stage_y is y:
            self._stage_states = states
        return states

    def rhs(self, time, y):
        self.evaluations += 1
        if self.evaluations > 20000:
            raise ValueError("Independent homogeneous IVP evaluation budget exceeded")
        self._stage_y, self._stage_states = y, None
        try:
            return self._rhs(time, y)
        finally:
            self._stage_y, self._stage_states = None, None

    def _rhs(self, time, y):
        result = super().rhs(time, y)
        states = self.thermos(y)
        for edge in self.spec["edges"]:
            if edge["id"] not in self.laws:
                continue
            model = self.laws[edge["id"]]["viscosity_model"]
            for j in range(edge["cells"]):
                i = self.names.index(edge["id"] + "/" + str(j))
                s = states[i]
                if s.region == 4:
                    liquid, vapor = iapws.saturation_properties(s.pressure_pa)
                    x = s.quality
                    if model == "mcadams":
                        mu = (
                            liquid.viscosity_pa_s
                            * vapor.viscosity_pa_s
                            / ((1 - x) * vapor.viscosity_pa_s + x * liquid.viscosity_pa_s)
                        )
                    else:
                        mu = liquid.viscosity_pa_s + x * (
                            vapor.viscosity_pa_s - liquid.viscosity_pa_s
                        )
                else:
                    mu = iapws.viscosity_pa_s(s.density_kg_m3, s.temperature_k)
                velocity = float(y[self.n + i] / y[i])
                diameter = edge["diameter_m"]
                re = s.density_kg_m3 * abs(velocity) * diameter / mu
                if self.laws[edge["id"]].get("darcy_law", "colebrook") == "churchill":
                    if re == 0:
                        drag = 0.0
                    else:
                        # Independent direct-power evaluation over this reference's moderate Re range.
                        # Independent log-sum-exp evaluation of the primary equation.
                        from scipy.special import logsumexp

                        base = abs(
                            2.457
                            * log(1 / ((7 / re) ** 0.9 + 0.27 * edge["roughness_m"] / diameter))
                        )
                        la = float("-inf") if base == 0 else 16 * log(base)
                        lb = 16 * log(37530 / re)
                        factor = 8 * float(
                            exp(logsumexp([12 * log(8 / re), -1.5 * logsumexp([la, lb])]) / 12)
                        )
                        drag = (
                            -factor
                            * self.volumes[i]
                            * s.density_kg_m3
                            * velocity
                            * abs(velocity)
                            / (2 * diameter)
                        )
                elif re < 2300:
                    drag = -32 * mu * self.volumes[i] * velocity / diameter**2
                else:
                    if not 4000 <= re <= 1e8:
                        raise ValueError("Reference homogeneous Re gap")
                    root = brentq(
                        lambda z: (
                            z + 2 * log10(edge["roughness_m"] / diameter / 3.7 + 2.51 * z / re)
                        ),
                        1,
                        100,
                        xtol=1e-13,
                    )
                    drag = (
                        -self.volumes[i]
                        * s.density_kg_m3
                        * velocity
                        * abs(velocity)
                        / (2 * diameter * root**2)
                    )
                result[self.n + i] += drag
                result[self.tail + 5] += -drag * velocity / s.temperature_k
                result[self.tail + 6 + i] += drag
        return result


def solve(request, program, method):
    assembly = Assembly(request)
    y = assembly.y0.copy()
    time = 0.0
    observations = [assembly.observation(time, y)]
    for call in program:
        if call["command"] is not None:
            assembly.controls = deepcopy(call["command"]["controls"])
        target = time + call["duration_s"]
        solved = solve_ivp(
            assembly.rhs, (time, target), y, method=method, rtol=2e-11, atol=1e-10, max_step=5e-5
        )
        if not solved.success:
            raise ValueError("Homogeneous reference failed")
        y = solved.y[:, -1]
        time = target
        observations.append(assembly.observation(time, y))
    return observations
