"""Steady constant-area open-channel IVP: prescribed inlet p/h and mass flux.

Predicts outlet pressure, never prescribes both terminal pressures. Static energy
and momentum include acceleration, Darcy drag and signed gravity. No time dynamics.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from threading import RLock

import numpy as np
from scipy.optimize import brentq

from atomic_model_hem_qualified import closures as drag
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import signature as prior_signature
from atomic_model_workspace.common import digest, keys, number

from .thermo import (
    continued_ph,
    entropy_derivatives,
    ph,
    saturated_enthalpy_tangent,
    volume_derivatives,
)

MAX_STEPS = 16384
MAX_CALLS = 128


def signature():
    return dict(
        prior=prior_signature(),
        throughflow={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


class SteadyChannel:
    def __init__(self, request):
        keys(
            request,
            (
                "schema_version",
                "inlet_pressure_pa",
                "inlet_enthalpy_j_kg",
                "mass_flux_kg_m2_s",
                "diameter_m",
                "length_m",
                "roughness_m",
                "gravity_m_s2",
                "heat_flux_w_m2",
                "viscosity_model",
                "friction_scope",
                "max_step_m",
            ),
        )
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Steady schema1")
        self.request = deepcopy(request)
        self.input_sha256 = digest(request)
        self.G = number(request["mass_flux_kg_m2_s"], 10, 2000, "mass flux")
        self.D = number(request["diameter_m"], 0.005, 0.05, "diameter")
        self.L = number(request["length_m"], 0.1, 10, "length")
        self.rough = number(request["roughness_m"], 0, 0.05 * self.D, "roughness")
        self.gravity = number(request["gravity_m_s2"], -9.80665, 9.80665, "gravity")
        self.max_step = number(request["max_step_m"], 0.0001, 0.5, "step")
        keys(request["friction_scope"], drag.FLAGS)
        if any(request["friction_scope"][k] is not True for k in drag.FLAGS):
            raise ValueError("Explicit homogeneous developed-flow scope required")
        self.friction_scope = deepcopy(request["friction_scope"])
        self.viscosity = request["viscosity_model"]
        if self.viscosity not in ("mcadams", "cicchitti"):
            raise ValueError("Explicit homogeneous viscosity law required")
        q = request["heat_flux_w_m2"]
        if not isinstance(q, list) or not 1 <= len(q) <= 256:
            raise ValueError("Piecewise-constant axial heat profile")
        self.q = np.array([number(x, -5e5, 5e5, "heat flux") for x in q])
        p = number(request["inlet_pressure_pa"], eos.P_MIN, eos.P_MAX, "inlet pressure")
        h = number(request["inlet_enthalpy_j_kg"], 1e4, 5e6, "inlet enthalpy")
        self.y0 = np.array([p, h, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.y = self.y0.copy()
        self.z = 0.0
        self.calls = []
        self.events = []
        self.lock = RLock()
        self.source = signature()
        self.s0 = self.point(self.y)[0]
        self.h00 = h + 0.5 * (self.G / self.s0.density_kg_m3) ** 2
        self.momentum0 = p + self.G**2 / self.s0.density_kg_m3
        self.rate(0.0, self.y, float(self.q[0]))

    def sources(self):
        if signature() != self.source:
            raise ValueError("Steady source changed")

    def contract(self):
        return dict(
            model="steady-open-hem-ivp-v1",
            coordinates=["pressure_pa", "static_enthalpy_j_kg"],
            boundary="Prescribed inlet static pressure/enthalpy and constant mass flux; outlet pressure predicted",
            heat="Prescribed piecewise-constant external heat flux; no automatic wall-temperature/postCHF model",
            momentum="p + G²/rho with Darcy and signed rho*g gravity",
            energy="h + G²/(2rho²) + g*z, explicit external wall heat",
            entropy="Fluid entropy-flux balance; heat term Q/Tfluid, drag production and explicit IF97 mixture Gibbs-fit defect; no wall-reservoir entropy claim",
            friction_scope=deepcopy(self.friction_scope),
            not_supported=[
                "transient storage",
                "choking",
                "slip",
                "pressure-pressure boundary value problem",
            ],
            validated=False,
            plant_referenced=False,
        )

    def point(self, y):
        if y.shape != (7,) or not np.isfinite(y).all():
            raise ValueError("Nonfinite steady coordinates")
        state = ph(float(y[0]), float(y[1]))
        low, high = eos.characteristic_speeds(state)
        velocity = self.G / state.density_kg_m3
        if velocity > 0.2 * low:
            raise ValueError("Steady low-Mach envelope; no choking model")
        return state, velocity

    def rate(self, z, y, q, trial_region=None):
        if trial_region is None:
            s, velocity = self.point(y)
        else:
            s = continued_ph(float(y[0]), float(y[1]), trial_region)
            velocity = self.G / s.density_kg_m3
        friction = drag.friction(
            s,
            velocity,
            self.D,
            self.rough,
            viscosity_model=self.viscosity,
            scope=self.friction_scope,
            darcy_law="churchill",
        )
        force = -friction["force_density_n_m3"]
        heating = 4 * q / (self.G * self.D)
        rho = s.density_kg_m3
        v = 1 / rho
        derivative_state = s
        if trial_region is None:
            for endpoint in (0.0, 1.0):
                boundary = eos.px(s.pressure_pa, endpoint)
                if abs(float(y[1]) - boundary.enthalpy_j_kg) <= 2e-6:
                    # One-sided derivative evaluation at solver resolution only.
                    # Conserved h and the admitted physical state are NOT changed.
                    derivative_state = boundary
                    break
        sides = [None]
        if (
            trial_region is None
            and derivative_state.region == 4
            and derivative_state.quality in (0.0, 1.0)
        ):
            sides = [None, "liquid" if derivative_state.quality == 0 else "vapor"]
        for side in sides:
            vp, vh = volume_derivatives(derivative_state, side)
            matrix = np.array(
                [[1 + self.G**2 * vp, self.G**2 * vh], [self.G**2 * v * vp, 1 + self.G**2 * v * vh]]
            )
            if np.linalg.det(matrix) <= 0:
                raise ValueError("Singular/supercritical steady Jacobian")
            dp, dh = np.linalg.solve(
                matrix, np.array([-force - rho * self.gravity, heating - self.gravity])
            )
            if len(sides) > 1:
                direction = (
                    dh
                    - saturated_enthalpy_tangent(
                        derivative_state.pressure_pa, derivative_state.quality
                    )
                    * dp
                )
                mixture_enters = direction >= 0 if derivative_state.quality == 0 else direction <= 0
                if (side is None) != mixture_enters:
                    continue
            sp, sh = entropy_derivatives(derivative_state, side)
            defect = sp * dp + sh * dh - (dh - v * dp) / s.temperature_k
            return np.array(
                [
                    dp,
                    dh,
                    heating,
                    force + rho * self.gravity,
                    heating / s.temperature_k,
                    force * v / s.temperature_k,
                    defect,
                ]
            )
        raise ValueError("No self-consistent endpoint tangent")

    def observe(self, z, y):
        s, velocity = self.point(y)
        return dict(
            position_m=float(z),
            state=s.as_dict(),
            velocity_m_s=float(velocity),
            mass_flow_kg_s=float(self.G * np.pi * self.D**2 / 4),
            ledgers=dict(
                external_heat_j_kg=float(y[2]),
                momentum_loss_pa=float(y[3]),
                heat_entropy_j_kg_k=float(y[4]),
                drag_entropy_j_kg_k=float(y[5]),
                energy_residual_j_kg=float(
                    s.enthalpy_j_kg + 0.5 * velocity**2 + self.gravity * z - self.h00 - y[2]
                ),
                momentum_residual_pa=float(
                    s.pressure_pa + self.G**2 / s.density_kg_m3 - self.momentum0 + y[3]
                ),
                property_gibbs_defect_j_kg_k=float(y[6]),
                corrected_numerical_entropy_residual_j_kg_k=float(
                    s.entropy_j_kg_k - self.s0.entropy_j_kg_k - y[4] - y[5] - y[6]
                ),
                entropy_residual_j_kg_k=float(
                    s.entropy_j_kg_k - self.s0.entropy_j_kg_k - y[4] - y[5]
                ),
            ),
        )

    def snapshot(self):
        with self.lock:
            self.sources()
            return self.observe(self.z, self.y)

    def advance_to(self, target_m):
        with self.lock:
            self.sources()
            target = number(target_m, 0, self.L, "target position")
            if target <= self.z or len(self.calls) >= MAX_CALLS:
                raise ValueError("Monotone target/call budget")
            z = self.z
            y = self.y.copy()
            steps = 0
            events = self.events.copy()
            while z < target:
                cell = min(int(np.floor(z / self.L * len(self.q) + 1e-12)), len(self.q) - 1)
                boundary = self.L * (cell + 1) / len(self.q)
                step = min(self.max_step, target - z, boundary - z)
                if step <= 0 or z + step <= z or steps >= MAX_STEPS:
                    raise ValueError("Spatial integration budget")
                q = float(self.q[cell])

                def trial(ds, region=None):
                    k1 = self.rate(z, y, q, region)
                    k2 = self.rate(z + ds / 2, y + ds * k1 / 2, q, region)
                    k3 = self.rate(z + ds / 2, y + ds * k2 / 2, q, region)
                    k4 = self.rate(z + ds, y + ds * k3, q, region)
                    return y + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6

                candidate = trial(step)
                crossings = []
                branch = self.point(y)[0].region
                for endpoint in (0.0, 1.0):

                    def gap(point):
                        return float(point[1]) - eos.px(float(point[0]), endpoint).enthalpy_j_kg

                    before, after = gap(y), gap(candidate)
                    if abs(before) > 2e-6 and before * after < 0:
                        root = brentq(lambda ds: gap(trial(ds, branch)), 0.0, step, xtol=1e-12)
                        crossings.append((root, endpoint))
                if crossings:
                    step, endpoint = min(crossings)
                    candidate = trial(step, branch)
                    residual = (
                        float(candidate[1]) - eos.px(float(candidate[0]), endpoint).enthalpy_j_kg
                    )
                    if abs(residual) > 2e-6 or len(events) >= 512:
                        raise ValueError(
                            f"Phase event root residual/budget: {residual}, z={z}, ds={step}"
                        )
                    events.append(
                        dict(
                            position_m=float(z + step),
                            endpoint_quality=endpoint,
                            enthalpy_residual_j_kg=residual,
                        )
                    )
                self.point(candidate)
                y = candidate
                z = min(target, z + step)
                steps += 1
            observation = self.observe(z, y)
            self.sources()
            self.y, self.z, self.calls = y, z, self.calls + [target]
            self.events = events
            return dict(observation, steps=steps)

    def checkpoint(self):
        with self.lock:
            self.sources()
            payload = dict(
                model="steady-open-hem-ivp-v1",
                source=self.source,
                input_sha256=self.input_sha256,
                calls=self.calls.copy(),
                phase_events=deepcopy(self.events),
                position_m=self.z,
                coordinates=self.y.tolist(),
                observation=self.observe(self.z, self.y),
            )
            return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        with self.lock:
            self.sources()
            keys(checkpoint, ("payload", "sha256"))
            payload = checkpoint["payload"]
            if (
                len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2
                or digest(payload) != checkpoint["sha256"]
            ):
                raise ValueError("Checkpoint digest/size")
            keys(
                payload,
                (
                    "model",
                    "source",
                    "input_sha256",
                    "calls",
                    "phase_events",
                    "position_m",
                    "coordinates",
                    "observation",
                ),
            )
            if (
                payload["source"] != self.source
                or payload["input_sha256"] != self.input_sha256
                or not isinstance(payload["calls"], list)
                or len(payload["calls"]) > MAX_CALLS
            ):
                raise ValueError("Checkpoint identity/budget")
            other = SteadyChannel(self.request)
            for target in payload["calls"]:
                other.advance_to(target)
            if other.checkpoint() != checkpoint:
                raise ValueError("Checkpoint full-history replay mismatch")
            self.y, self.z, self.calls = other.y.copy(), other.z, other.calls.copy()
            self.events = deepcopy(other.events)
            return self.snapshot()
