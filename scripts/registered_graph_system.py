"""Explicit registry for one supported nonlinear residual system family.

The registered item is a complete graph adapter, not arbitrary composable
nonlinear equipment. No plugin imports, pressure boundaries or equipment are
synthesized from wire data.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import numpy as np

from core.network import NetworkError
from scripts.graph_port_adapter import GraphPortAdapter
from scripts.graph_port_adapter import signature as adapter_signature
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.solve_resistance_feedback import effective_input

BUDGET = Path("validation/ai05_reduced_engine_budget_v1.json")


def signature():
    value = adapter_signature()
    for path in ("scripts/registered_graph_system.py", "scripts/reduced_residual_engine.py"):
        value["drivers"][path] = sha(path)
    value["budgets"][str(BUDGET)] = sha(BUDGET)
    return value


class ThermalGraphSystem:
    def __init__(self, spec):
        self.adapter = GraphPortAdapter(deepcopy(spec))
        self.spec = deepcopy(spec)
        self.identity = digest(spec)
        self.contract = self.adapter.contract()
        self.initial = self.adapter.initialize()
        self.origin = np.array(self.initial["state"])
        self.mass = np.diag(self.adapter.mass_matrix())
        self.differential = np.flatnonzero(self.mass != 0)
        self.algebraic = np.flatnonzero(self.mass == 0)
        self.scales = np.r_[np.ones(self.adapter._n), self.adapter._kernel.scale]
        self.atol_units = tuple(["temperature"] * self.adapter._n + ["energy"])
        self.budget = json.loads(BUDGET.read_text())

    def residual(self, state, rate):
        return self.adapter.residual(state, rate)

    def algebraic_jacobian(self, state):
        return self.adapter.algebraic_jacobian(state)

    def admit(self, full, requested):
        k = self.adapter._kernel
        n = self.adapter._n
        temperatures = full[:n]
        hyd = dict(
            pressures_pa=dict(zip(k.names, full[n : 2 * n])),
            flows_kg_s={e["id"]: full[2 * n + j] for j, e in enumerate(k.graph["edges"])},
        )
        effective, _ = effective_input(k.base, k.graph, dict(zip(k.names, temperatures)), k.laws)
        lo, hi, rl, rh = k._box(temperatures, hyd, effective)
        step = requested
        while not (
            np.all(temperatures + step * np.minimum(rl, 0) > lo)
            and np.all(temperatures + step * np.maximum(rh, 0) < hi)
        ):
            step *= 0.5
            if step < self.budget["minimum_macro_step_s"]:
                raise NetworkError("Reduced interval cannot be admitted")
        return step, (lo, hi)

    def require_box(self, full, box):
        n = self.adapter._n
        lo, hi = box
        if np.any(full[:n] <= lo) or np.any(full[:n] >= hi):
            raise NetworkError("Radau stage left admitted box")

    def energy_error(self, increments):
        return float(
            self.adapter._kernel.capacity @ increments[:-1] - increments[-1] * self.scales[-1]
        )

    def observe(self, full, increments):
        n = self.adapter._n
        k = self.adapter._kernel
        return dict(
            temperatures_k=dict(zip(k.names, map(float, full[:n]))),
            pressures_pa=dict(zip(k.names, map(float, full[n : 2 * n]))),
            flows_kg_s={e["id"]: float(full[2 * n + j]) for j, e in enumerate(k.graph["edges"])},
            temperature_increments_k=dict(zip(k.names, map(float, increments[:-1]))),
            integrated_external_energy_j=float(increments[-1] * self.scales[-1]),
            energy_residual_j=self.energy_error(increments),
        )

    def from_legacy(self, checkpoint):
        lifted = self.adapter.lift_checkpoint(checkpoint)
        kernel = NonlinearThermalGraph(deepcopy(self.spec))
        kernel.restore(deepcopy(checkpoint))
        increments = kernel._state.copy()
        return increments, np.array(lifted["state"]), lifted["time_s"]

    def to_legacy(self, increments, time):
        kernel = NonlinearThermalGraph(deepcopy(self.spec))
        kernel._state = increments.copy()
        kernel._time = time
        # Native DP proposal/history is not Radau state. Restart it explicitly.
        kernel._proposal = kernel.max_step
        checkpoint = kernel.checkpoint()
        verifier = NonlinearThermalGraph(deepcopy(self.spec))
        verifier.restore(checkpoint)
        return checkpoint


SYSTEM_TYPES = MappingProxyType({"constant_rho_thermal_graph_v1": ThermalGraphSystem})


def create_system(request):
    _keys(request, ("format", "schema_version", "kind", "parameters"))
    if (
        request["format"] != "atomic-registered-residual"
        or type(request["schema_version"]) is not int
        or request["schema_version"] != 1
        or not isinstance(request["kind"], str)
        or request["kind"] not in SYSTEM_TYPES
    ):
        raise NetworkError("Unsupported registered residual system")
    return SYSTEM_TYPES[request["kind"]](deepcopy(request["parameters"]))


def graph_request(parameters):
    return dict(
        format="atomic-registered-residual",
        schema_version=1,
        kind="constant_rho_thermal_graph_v1",
        parameters=deepcopy(parameters),
    )
