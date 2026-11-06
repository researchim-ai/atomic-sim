"""Additional homogeneous drag on the unchanged source-bound graph family."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_hem_flow.model import characteristic_speeds
from atomic_model_hem_mixture.friction import FLAGS, MODELS, evaluate
from atomic_model_hem_mixture.inverse import conserved
from atomic_model_hem_network.model import GraphHEMNetwork
from atomic_model_hem_network.model import signature as graph_signature
from atomic_model_hem_system.model import MAX_CALLS
from atomic_model_workspace.common import digest, keys


def signature():
    return dict(
        graph=graph_signature(),
        mixture={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


class MixtureHEMNetwork(GraphHEMNetwork):
    def __init__(self, request):
        keys(request, ("schema_version", "model", "mixture_friction"))
        if type(request["schema_version"]) is not int or request["schema_version"] != 1:
            raise ValueError("Mixture network schema1 required")
        laws = request["mixture_friction"]
        if not isinstance(laws, dict) or not laws:
            raise ValueError("Nonempty mixture edge selection required")
        ids = {e["id"]: e for e in request["model"]["edges"]}
        if not set(laws) <= set(ids):
            raise ValueError("Unknown mixture edge")
        for name, law in laws.items():
            keys(law, ("viscosity_model", "scope") + (("darcy_law",) if "darcy_law" in law else ()))
            keys(law["scope"], FLAGS)
            if law["viscosity_model"] not in MODELS or any(
                law["scope"][k] is not True for k in FLAGS
            ):
                raise ValueError("Mixture law/scope")
            if ids[name]["friction"] != "none":
                raise ValueError("Cannot double count Darcy and homogeneous drag")
        self.mixture_laws = deepcopy(laws)
        super().__init__(request["model"])
        self.request = deepcopy(request)
        self.identity = digest(request)
        self.source = signature()
        self.evaluate(self._y, self._controls)

    def primitive(self, y):
        if self._constructing:
            return super().primitive(y)
        if not isinstance(y, np.ndarray) or y.shape != (self.size,) or not np.isfinite(y).all():
            raise ValueError("Invalid mixture coordinates")
        m = self.m0 + y[: self.n]
        P = self.P0 + y[self.n : 2 * self.n]
        C = self.C0 + y[2 * self.n : 3 * self.n]
        if np.any(m <= 0):
            raise ValueError("Nonpositive mixture mass")
        v = P / m
        u = self.eref + (C - 0.5 * m * v * v) / m
        states = []
        speeds = []
        cache = {}
        for i, seed in enumerate(self.states0):
            key = (float(m[i] / self.volumes[i]), float(u[i]), seed.pressure_pa, seed.temperature_k)
            if key not in cache:
                state = conserved(*key)
                cache[key] = (state, *characteristic_speeds(state))
            state, lower, upper = cache[key]
            if abs(v[i]) > 0.2 * lower or abs(v[i]) + upper > self.wave:
                raise ValueError("Mixture Mach/wave bound")
            states.append(state)
            speeds.append(upper)
        tw = self.tw0 + y[self.wall_slice]
        if np.any(tw < 300) or np.any(tw > 900):
            raise ValueError("Mixture wall temperature domain")
        return m, P, C, v, states, tw, np.array(speeds)

    def _sources(self):
        if signature() != self.source:
            raise ValueError("Mixture source changed")

    def _evaluate_point(self, y, controls, point):
        rhs, aux = super()._evaluate_point(y, controls, point)
        if self._constructing:
            return rhs, aux
        _, P, _, velocity, states, _, _ = point
        extra = np.zeros(self.n)
        entropy = 0.0
        for edge, indices in zip(self.edge_specs, self.edge_indices):
            if edge["id"] not in self.mixture_laws:
                continue
            law = self.mixture_laws[edge["id"]]
            for i in indices:
                result = evaluate(
                    states[i], float(velocity[i]), edge["diameter_m"], edge["roughness_m"], **law
                )
                extra[i] = result.force_density_n_m3 * self.volumes[i]
                entropy += result.entropy_production_w_m3_k * self.volumes[i]
        admission = min(self.step_cap, self.cfl * self.minimum_length / self.wave)
        moving = P != 0
        if np.any(admission * abs(extra[moving] / P[moving]) > 0.1):
            raise ValueError("Homogeneous drag relaxation bound; reduce max_step_s")
        rhs[self.n : 2 * self.n] += extra
        rhs[self.drag_slice] += extra
        rhs[self.drag_entropy_index] += entropy
        aux["drag_force_n"] = aux["drag_force_n"] + extra
        return rhs, aux

    def contract(self):
        contract = super().contract()
        contract["model"] = "homogeneous-friction-graph-v1"
        contract["friction_scope"] = (
            "Selected edges: homogeneous no-slip McAdams/Cicchitti viscosity with Darcy law; numerical model choices, not calibrated two-phase accuracy"
        )
        contract["mixture_friction"] = deepcopy(self.mixture_laws)
        return contract

    def _checkpoint(self, *args):
        cp = super()._checkpoint(*args)
        cp["payload"]["model"] = "homogeneous-friction-graph-v1"
        cp["sha256"] = digest(cp["payload"])
        return cp

    def restore(self, checkpoint):
        with self._lock:
            self._sources()
            keys(checkpoint, ("payload", "sha256"))
            p = checkpoint["payload"]
            if (
                len(json.dumps(checkpoint, allow_nan=False).encode()) > 4 * 1024**2
                or digest(p) != checkpoint["sha256"]
                or p.get("source") != self.source
                or p.get("input_sha256") != self.identity
            ):
                raise ValueError("Mixture checkpoint identity/size")
            calls = p.get("calls")
            if not isinstance(calls, list) or len(calls) > MAX_CALLS:
                raise ValueError("Mixture replay budget")
            other = MixtureHEMNetwork(self.request)
            for call in calls:
                keys(call, ("duration_s", "command"))
                c = call["command"]
                if c is None:
                    other.advance(call["duration_s"])
                else:
                    keys(c, ("controls", "expected_revision", "controller_state"))
                    other.step(
                        c["controls"],
                        call["duration_s"],
                        expected_revision=c["expected_revision"],
                        controller_state=c["controller_state"],
                    )
            if other.checkpoint() != checkpoint:
                raise ValueError("Mixture full checkpoint replay differs")
            self._sources()
            self._y, self._time, self._controls, self._controller, self._calls = (
                other._y,
                other._time,
                other._controls,
                other._controller,
                other._calls,
            )
            return self.snapshot()
