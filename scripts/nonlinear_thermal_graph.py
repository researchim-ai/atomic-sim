"""Quasi-steady hydraulic feedback in a finite-inventory thermal ODE.

User L(T), constant rho/cp, no fluid momentum inertia. Every RK stage solves
hydraulics and transports pressure enthalpy. Conservative temperature/pressure/
stream bounds admit an entire interval; unproved intervals are refused.
"""

import argparse
import hashlib
import json
import math
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy

from core.network import NetworkError, PortSpec, number
from scripts.solve_hydraulic_graph import _keys, sha, solve_hydraulic_graph
from scripts.solve_resistance_feedback import ResistanceFeedbackError, effective_input
from scripts.solve_resistance_feedback import prepare as prepare_feedback
from sdk.experiment import _source_fingerprint

BUDGET = Path("validation/ai05_nonlinear_graph_budget_v1.json")
DRIVERS = (
    "scripts/nonlinear_thermal_graph.py",
    "scripts/solve_resistance_feedback.py",
    "scripts/solve_thermal_graph.py",
    "scripts/solve_hydraulic_graph.py",
)


def signature():
    budgets = (
        BUDGET,
        Path("validation/ai05_resistance_feedback_budget_v2.json"),
        Path("validation/ai05_thermal_graph_budget_v1.json"),
        Path("validation/ai05_hydraulic_graph_budget_v1.json"),
    )
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={p: sha(p) for p in DRIVERS},
        budgets={str(p): sha(p) for p in budgets},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def digest(value):
    try:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (ValueError, TypeError, OverflowError) as error:
        raise NetworkError("Finite JSON-compatible input/checkpoint required") from error


class NonlinearGraphError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


# Dormand-Prince 5(4); coefficients are part of the source identity.
RK_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
RK_B = np.array((35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0))
RK_LOW = np.array((5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40))


class NonlinearThermalGraph:
    def __init__(self, spec):
        _keys(
            spec,
            (
                "format",
                "schema_version",
                "steady_input",
                "resistance_laws",
                "volumes_m3",
                "initial_temperatures_k",
                "max_step_s",
            ),
        )
        if (
            spec["format"] != "atomic-nonlinear-thermal-graph"
            or type(spec["schema_version"]) is not int
            or spec["schema_version"] != 1
        ):
            raise NetworkError("Expected atomic-nonlinear-thermal-graph schema 1")
        self.budget = json.loads(BUDGET.read_text())
        self.identity = digest(spec)
        descriptor = dict(
            format="atomic-resistance-feedback-steady",
            schema_version=1,
            steady_input=spec["steady_input"],
            resistance_laws=spec["resistance_laws"],
            initial_temperatures_k=spec["initial_temperatures_k"],
        )
        self.base, self.graph, initial, self.laws = prepare_feedback(descriptor)
        if (
            self.base["format"] != "atomic-thermal-hydraulic-steady"
            or type(self.base["schema_version"]) is not int
            or self.base["schema_version"] != 1
        ):
            raise NetworkError("Expected atomic-thermal-hydraulic-steady schema 1")
        self.names = [n["id"] for n in self.graph["nodes"]]
        self.index = {n: i for i, n in enumerate(self.names)}
        n = len(self.names)
        edges = self.graph["edges"]
        if n > self.budget["maximum_nodes"] or len(edges) > self.budget["maximum_edges"]:
            raise NetworkError("Nonlinear graph size exceeds budget")
        if not isinstance(spec["volumes_m3"], dict) or set(spec["volumes_m3"]) != set(self.names):
            raise NetworkError("Positive volume required for every node")
        self.max_step = number(spec["max_step_s"], "max_step_s", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Maximum step exceeds budget")
        self.cfg = self._thermal_contract(self.base["thermal"])
        self.rho = self.graph["density_kg_m3"]
        self.cp = self.cfg["heat_capacity_j_kg_k"]
        self.ref = self.cfg["enthalpy_reference_temperature_k"]
        self.capacity = np.array(
            [
                number(spec["volumes_m3"][k], "volume_m3", positive=True) * self.rho * self.cp
                for k in self.names
            ]
        )
        self.scale = float(self.capacity.sum())
        self.offset = np.array([initial[k] for k in self.names])
        self.initial_energy = float(self.capacity @ (self.offset - self.ref))
        if (
            not np.all(np.isfinite(self.capacity))
            or np.any(self.capacity <= 0)
            or not np.isfinite(self.scale)
            or self.scale <= 0
            or not np.isfinite(self.initial_energy)
        ):
            raise NetworkError("Thermal inventory exceeds finite arithmetic")
        self._state = np.zeros(n + 1)
        self._time = 0.0
        self._proposal = self.max_step
        self.captured = signature()
        self._lower = self.cfg["min_temperature_k"]
        self._upper = self.cfg["max_temperature_k"]
        self._a = np.zeros((n, len(edges)))
        for j, e in enumerate(edges):
            self._a[self.index[e["from"]], j] = 1
            self._a[self.index[e["to"]], j] = -1
        self.free = [i for i, node in enumerate(self.graph["nodes"]) if "injection_kg_s" in node]
        af = self._a[self.free]
        self._projection = (
            af.T @ np.linalg.inv(af @ af.T) if self.free else np.zeros((len(edges), 0))
        )
        self._projection_norm = float(np.linalg.norm(self._projection, 2)) if self.free else 0.0
        self._injection = np.array([self.graph["nodes"][i]["injection_kg_s"] for i in self.free])
        self._hyd_budget = json.loads(
            Path("validation/ai05_hydraulic_graph_budget_v1.json").read_text()
        )
        self._evaluate(self.offset)

    def _thermal_contract(self, cfg):
        _keys(
            cfg,
            (
                "medium",
                "heat_capacity_j_kg_k",
                "enthalpy_reference_temperature_k",
                "min_temperature_k",
                "max_temperature_k",
                "nodes",
            ),
        )
        if not isinstance(cfg["medium"], str) or not cfg["medium"].strip():
            raise NetworkError("Explicit medium required")
        PortSpec(
            "fluid",
            "hydraulic",
            medium=cfg["medium"],
            energy_convention="specific_enthalpy_transport",
        )
        value = {
            k: number(cfg[k], k, positive=True)
            for k in (
                "heat_capacity_j_kg_k",
                "enthalpy_reference_temperature_k",
                "min_temperature_k",
                "max_temperature_k",
            )
        }
        if value["min_temperature_k"] >= value["max_temperature_k"]:
            raise NetworkError("Empty temperature domain")
        if not isinstance(cfg["nodes"], dict) or set(cfg["nodes"]) != set(self.names):
            raise NetworkError("Thermal node map mismatch")
        value.update(medium=cfg["medium"], nodes={})
        for hydraulic in self.graph["nodes"]:
            name = hydraulic["id"]
            node = cfg["nodes"][name]
            _keys(
                node,
                ("heat_power_w", "ambient_ua_w_k", "ambient_temperature_k"),
                ("supply_temperature_k",),
            )
            v = {k: number(node[k], k, positive=k.endswith("temperature_k")) for k in sorted(node)}
            if v["ambient_ua_w_k"] < 0:
                raise NetworkError("Ambient conductance must be nonnegative")
            if (
                "pressure_pa" in hydraulic or hydraulic.get("injection_kg_s", 0) > 0
            ) and "supply_temperature_k" not in v:
                raise NetworkError("Every possible external inflow requires a supplied temperature")
            if (
                "supply_temperature_k" in v
                and not value["min_temperature_k"]
                < v["supply_temperature_k"]
                < value["max_temperature_k"]
            ):
                raise NetworkError("Supply exceeds temperature domain")
            value["nodes"][name] = v
        return value

    def _evaluate(self, t):
        if not np.all(np.isfinite(t)) or np.any(t <= self._lower) or np.any(t >= self._upper):
            raise NonlinearGraphError("temperature_domain", "Temperature exceeds open domain")
        effective, _ = effective_input(self.base, self.graph, dict(zip(self.names, t)), self.laws)
        hyd = solve_hydraulic_graph(effective["hydraulic"])
        energy = np.zeros(len(t))
        mechanical = hyd["powers_w"]["prescribed_gain"] + hyd["powers_w"]["gravity"]
        for edge in effective["hydraulic"]["edges"]:
            q = hyd["flows_kg_s"][edge["id"]]
            donor, receiver = (edge["from"], edge["to"]) if q >= 0 else (edge["to"], edge["from"])
            a, b = self.index[donor], self.index[receiver]
            rate = abs(q)
            loss = (
                edge["linear_pa_s_kg"] * q * q + edge["quadratic_pa_s2_kg2"] * rate**3
            ) / self.rho
            rise = (edge["linear_pa_s_kg"] * rate + edge["quadratic_pa_s2_kg2"] * q * q) / (
                self.rho * self.cp
            )
            if q != 0 and not self._lower < t[a] + rise < self._upper:
                raise NonlinearGraphError(
                    "stream_domain", "Stream exceeds temperature domain before mixing"
                )
            h = self.cp * (t[a] - self.ref)
            energy[a] -= rate * (h + hyd["pressures_pa"][donor] / self.rho)
            energy[b] += rate * (h + hyd["pressures_pa"][receiver] / self.rho) + loss
        external = 0.0
        for name, i in self.index.items():
            node = self.cfg["nodes"][name]
            q = hyd["external_injections_kg_s"][name]
            supply = node["supply_temperature_k"] if q > 0 else t[i]
            power = (
                q * (self.cp * (supply - self.ref) + hyd["pressures_pa"][name] / self.rho)
                + node["heat_power_w"]
                - node["ambient_ua_w_k"] * (t[i] - node["ambient_temperature_k"])
            )
            energy[i] += power
            external += power
        power = external + mechanical
        balance = float(energy.sum()) - power
        if (
            not np.all(np.isfinite(energy))
            or not math.isfinite(power)
            or abs(balance) > self.budget["maximum_instantaneous_energy_error_w"]
        ):
            raise NonlinearGraphError(
                "instantaneous_energy",
                "Instantaneous stock/port power mismatch",
                residual_w=balance if math.isfinite(balance) else None,
            )
        return np.r_[energy / self.capacity, power / self.scale], hyd, effective

    def _flow_bound(self, t, lo, hi, hyd, effective):
        low, _ = effective_input(self.base, self.graph, dict(zip(self.names, lo)), self.laws)
        high, _ = effective_input(self.base, self.graph, dict(zip(self.names, hi)), self.laws)
        llo = np.minimum(
            [e["linear_pa_s_kg"] for e in low["hydraulic"]["edges"]],
            [e["linear_pa_s_kg"] for e in high["hydraulic"]["edges"]],
        )
        lhi = np.maximum(
            [e["linear_pa_s_kg"] for e in low["hydraulic"]["edges"]],
            [e["linear_pa_s_kg"] for e in high["hydraulic"]["edges"]],
        )
        current = np.array([hyd["flows_kg_s"][e["id"]] for e in self.graph["edges"]])
        q0 = current + self._projection @ (self._injection - self._a[self.free] @ current)
        p = np.array([hyd["pressures_pa"][k] for k in self.names])
        linear = np.array([e["linear_pa_s_kg"] for e in effective["hydraulic"]["edges"]])
        r = np.array([e["quadratic_pa_s2_kg2"] for e in self.graph["edges"]])
        gain = np.array(
            [
                e["pressure_gain_pa"]
                - self.rho
                * self.graph["gravity_m_s2"]
                * (
                    self.graph["nodes"][self.index[e["to"]]]["elevation_m"]
                    - self.graph["nodes"][self.index[e["from"]]]["elevation_m"]
                )
                for e in self.graph["edges"]
            ]
        )
        residual = np.abs(self._a.T @ p + gain - linear * q0 - r * q0 * np.abs(q0)) + np.maximum(
            np.abs(llo - linear), np.abs(lhi - linear)
        ) * np.abs(q0)
        minimum = float(llo.min())
        correction = (
            self._projection_norm
            * math.sqrt(len(self.free))
            * self._hyd_budget["maximum_mass_residual_kg_s"]
        )
        slope = 2 * float(r.max()) * correction / minimum
        if not math.isfinite(slope) or slope >= 1:
            raise NonlinearGraphError(
                "hydraulic_bound", "Hydraulic residual bounds cannot be resolved"
            )
        base = (
            float(np.linalg.norm(q0))
            + (
                float(np.linalg.norm(residual))
                + math.sqrt(len(r)) * self._hyd_budget["maximum_pressure_residual_pa"]
            )
            / minimum
        )
        bound = (
            base
            + correction
            + (float(lhi.max()) + 2 * float(r.max()) * correction) * correction / minimum
        ) / (1 - slope)
        change = bound - float(np.linalg.norm(q0)) + float(np.linalg.norm(q0 - current))
        pressure_change = (len(t) - 1) * float(
            np.max(
                (lhi + 2 * r * bound) * change
                + np.maximum(np.abs(llo - linear), np.abs(lhi - linear)) * np.abs(current)
                + 2 * self._hyd_budget["maximum_pressure_residual_pa"]
            )
        )
        for i in self.free:
            node = self.graph["nodes"][i]
            if p[i] - pressure_change <= node.get("min_pressure_pa", 0) or p[
                i
            ] + pressure_change >= node.get("max_pressure_pa", math.inf):
                raise NonlinearGraphError(
                    "pressure_box", "Cannot admit pressure over whole temperature box"
                )
        return bound, lhi, correction

    def _box(self, t, hyd, effective):
        radius = np.minimum(
            self.budget["maximum_box_radius_k"], 0.25 * np.minimum(t - self._lower, self._upper - t)
        )
        for _ in range(self.budget["maximum_box_refinements"]):
            lo, hi = t - radius, t + radius
            try:
                bound, lhi, _ = self._flow_bound(t, lo, hi, hyd, effective)
                lower = np.zeros(len(t))
                upper = np.zeros(len(t))
                degree = np.abs(self._a).sum(axis=1)
                pressure_bound = max(hyd["pressures_pa"].values()) + (len(t) - 1) * max(
                    lhi[j] * bound
                    + e["quadratic_pa_s2_kg2"] * bound * bound
                    + abs(e["pressure_gain_pa"])
                    + self.rho
                    * self.graph["gravity_m_s2"]
                    * abs(
                        self.graph["nodes"][self.index[e["to"]]]["elevation_m"]
                        - self.graph["nodes"][self.index[e["from"]]]["elevation_m"]
                    )
                    for j, e in enumerate(self.graph["edges"])
                )
                for j, e in enumerate(self.graph["edges"]):
                    a, b = self.index[e["from"]], self.index[e["to"]]
                    r = e["quadratic_pa_s2_kg2"]
                    rise = (lhi[j] * bound + r * bound * bound) / (self.rho * self.cp)
                    if max(hi[a], hi[b]) + rise >= self._upper:
                        raise NonlinearGraphError(
                            "stream_box", "Cannot admit all stream temperatures in box"
                        )
                    diss = (lhi[j] * bound * bound + r * bound**3) / self.rho
                    for i, other in ((a, b), (b, a)):
                        lower[i] += self.cp * bound * min(lo[other] - hi[i], 0)
                        upper[i] += self.cp * bound * max(hi[other] - lo[i], 0) + diss
                for name, i in self.index.items():
                    cfg = self.cfg["nodes"][name]
                    node = self.graph["nodes"][i]
                    source = (
                        degree[i] * bound
                        if "pressure_pa" in node
                        else max(node["injection_kg_s"], 0)
                    )
                    if source > 0:
                        supply = cfg["supply_temperature_k"]
                        lower[i] += self.cp * source * min(supply - hi[i], 0)
                        upper[i] += self.cp * source * max(supply - lo[i], 0)
                    lower[i] += cfg["heat_power_w"] - cfg["ambient_ua_w_k"] * (
                        hi[i] - cfg["ambient_temperature_k"]
                    )
                    upper[i] += cfg["heat_power_w"] - cfg["ambient_ua_w_k"] * (
                        lo[i] - cfg["ambient_temperature_k"]
                    )
                    rounding = (
                        self.cp * max(abs(lo[i] - self.ref), abs(hi[i] - self.ref))
                        + pressure_bound / self.rho
                    ) * max(
                        self._hyd_budget["maximum_mass_residual_kg_s"],
                        32 * np.finfo(float).eps * degree[i] * bound,
                    )
                    lower[i] -= rounding
                    upper[i] += rounding
                low_rate, high_rate = lower / self.capacity, upper / self.capacity
                if not np.all(np.isfinite(low_rate)) or not np.all(np.isfinite(high_rate)):
                    raise NonlinearGraphError("nonfinite_bound", "Nonfinite interval rate bound")
                return lo, hi, low_rate, high_rate
            except (
                NonlinearGraphError,
                ResistanceFeedbackError,
                OverflowError,
                FloatingPointError,
            ):
                radius *= 0.5
        raise NonlinearGraphError(
            "interval_box", "Cannot certify a temperature/pressure/stream box"
        )

    def rates(self, temperatures_k=None):
        if temperatures_k is not None and (
            not isinstance(temperatures_k, dict) or set(temperatures_k) != set(self.names)
        ):
            raise NetworkError("Rate temperature map mismatch")
        t = (
            self.offset + self._state[:-1]
            if temperatures_k is None
            else np.array(
                [number(temperatures_k[k], "temperature_k", positive=True) for k in self.names]
            )
        )
        rate, hyd, _ = self._evaluate(t)
        return dict(
            temperature_rates_k_s=dict(zip(self.names, map(float, rate[:-1]))),
            external_power_w=float(rate[-1] * self.scale),
            pressures_pa=deepcopy(hyd["pressures_pa"]),
            flows_kg_s=deepcopy(hyd["flows_kg_s"]),
        )

    def _observation(self, state, time, proposal):
        t = self.offset + state[:-1]
        _, hyd, _ = self._evaluate(t)
        return dict(
            time_s=time,
            temperatures_k=dict(zip(self.names, map(float, t))),
            temperature_increments_k=dict(zip(self.names, map(float, state[:-1]))),
            energy_integral_scaled_k=float(state[-1]),
            integrated_external_energy_j=float(state[-1] * self.scale),
            stored_energy_j=self.initial_energy + float(self.capacity @ state[:-1]),
            energy_residual_j=float(self.capacity @ state[:-1] - state[-1] * self.scale),
            step_proposal_s=proposal,
            pressures_pa=deepcopy(hyd["pressures_pa"]),
            flows_kg_s=deepcopy(hyd["flows_kg_s"]),
        )

    def snapshot(self):
        return self._observation(self._state, self._time, self._proposal)

    def advance(self, duration_s):
        duration = number(duration_s, "duration_s", positive=True)
        target = self._time + duration
        if (
            duration > self.budget["maximum_duration_s"]
            or not math.isfinite(target)
            or target <= self._time
        ):
            raise NetworkError("Duration or representable time budget exceeded")
        if signature() != self.captured:
            raise NonlinearGraphError("source_drift", "Nonlinear sources changed")
        state = self._state.copy()
        remaining = duration
        proposal = self._proposal
        accepted, rejected = 0, 0
        minimum_step = math.inf
        maximum_step = 0.0
        current_evaluation = self._evaluate(self.offset + state[:-1])
        while remaining > 0:
            if accepted + rejected >= self.budget["maximum_attempts"]:
                raise NonlinearGraphError(
                    "attempt_budget", "Step budget exhausted; full advance rolled back"
                )
            t = self.offset + state[:-1]
            first, hyd, effective = current_evaluation
            lo, hi, rl, rh = self._box(t, hyd, effective)
            step = min(proposal, self.max_step, remaining)
            while not (
                np.all(t + step * np.minimum(rl, 0) > lo)
                and np.all(t + step * np.maximum(rh, 0) < hi)
            ):
                step *= 0.5
                if step < self.budget["minimum_step_s"]:
                    raise NonlinearGraphError(
                        "interval_domain",
                        "Full interval cannot be certified; full advance rolled back",
                    )
            stages = [first]
            try:
                for weights in RK_A[1:]:
                    trial = state + step * np.asarray(weights) @ np.asarray(stages)
                    temp = self.offset + trial[:-1]
                    if np.any(temp <= lo) or np.any(temp >= hi):
                        raise NonlinearGraphError("stage_box", "Trial stage left certified box")
                    last_evaluation = self._evaluate(temp)
                    stages.append(last_evaluation[0])
                derivatives = np.asarray(stages)
                candidate = trial
                error = step * ((RK_B - RK_LOW) @ derivatives)
                tolerance = np.r_[
                    np.full(len(t), self.budget["temperature_atol_k"]),
                    self.budget["energy_atol_j"] / self.scale,
                ] + self.budget["rtol"] * np.maximum(np.abs(state), np.abs(candidate))
                norm = float(np.max(np.abs(error) / tolerance))
                if not math.isfinite(norm):
                    raise NonlinearGraphError(
                        "integration_error", "Nonfinite adaptive error estimate"
                    )
            except NetworkError:
                norm = math.inf
            if norm > 1:
                rejected += 1
                proposal = step * 0.5
                if proposal < self.budget["minimum_step_s"]:
                    raise NonlinearGraphError(
                        "minimum_step",
                        "Adaptive stage cannot be admitted; full advance rolled back",
                    )
                continue
            state = candidate
            current_evaluation = last_evaluation
            remaining = max(0.0, remaining - step)
            accepted += 1
            minimum_step = min(minimum_step, step)
            maximum_step = max(maximum_step, step)
            proposal = min(
                self.max_step,
                max(
                    self.budget["minimum_step_s"],
                    step * (5.0 if norm == 0 else min(5.0, max(0.2, 0.9 * norm ** (-0.2)))),
                ),
            )
        self._evaluate(self.offset + state[:-1])
        energy_error = float(self.capacity @ state[:-1] - state[-1] * self.scale)
        if (
            not math.isfinite(energy_error)
            or abs(energy_error) > self.budget["maximum_energy_error_j"]
        ):
            raise NonlinearGraphError(
                "integrated_energy",
                "Energy budget exceeded; full advance rolled back",
                residual_j=energy_error,
            )
        if signature() != self.captured:
            raise NonlinearGraphError("source_drift", "Sources changed during advance; rolled back")
        result = self._observation(state, target, proposal)
        self._state = state
        self._time = target
        self._proposal = proposal
        result.update(accepted_steps=accepted, rejected_steps=rejected)
        result.update(minimum_accepted_step_s=minimum_step, maximum_accepted_step_s=maximum_step)
        return result

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="nonlinear-thermal-graph-v1",
            input_sha256=self.identity,
            signature=deepcopy(self.captured),
            state=self.snapshot(),
        )
        return dict(payload=payload, sha256=digest(payload))

    def restore(self, checkpoint):
        _keys(checkpoint, ("payload", "sha256"))
        p = checkpoint["payload"]
        if digest(p) != checkpoint["sha256"]:
            raise NetworkError("Checkpoint digest mismatch")
        _keys(p, ("schema_version", "model", "input_sha256", "signature", "state"))
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["model"] != "nonlinear-thermal-graph-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
            or signature() != self.captured
        ):
            raise NetworkError("Checkpoint model/input/source mismatch")
        s = p["state"]
        expected = set(self.snapshot())
        _keys(s, expected)
        for key in ("temperatures_k", "temperature_increments_k", "pressures_pa", "flows_kg_s"):
            names = self.names if key != "flows_kg_s" else [e["id"] for e in self.graph["edges"]]
            if not isinstance(s[key], dict) or set(s[key]) != set(names):
                raise NetworkError("Checkpoint field map mismatch")
        temperature = np.array(
            [number(s["temperatures_k"][k], "temperature_k") for k in self.names]
        )
        increment = np.array(
            [number(s["temperature_increments_k"][k], "increment_k") for k in self.names]
        )
        if not np.array_equal(temperature, self.offset + increment):
            raise NetworkError("Checkpoint temperature/increment mismatch")
        _, hyd, _ = self._evaluate(temperature)
        if hyd["pressures_pa"] != s["pressures_pa"] or hyd["flows_kg_s"] != s["flows_kg_s"]:
            raise NetworkError("Checkpoint hydraulic replay differs")
        time = number(s["time_s"], "time_s")
        energy = number(s["integrated_external_energy_j"], "energy_j")
        scaled_energy = number(s["energy_integral_scaled_k"], "scaled_energy")
        stored = number(s["stored_energy_j"], "stored_energy_j")
        residual = number(s["energy_residual_j"], "energy_residual_j")
        proposal = number(s["step_proposal_s"], "step_proposal_s", positive=True)
        change = float(self.capacity @ increment)
        tol = self.budget["maximum_energy_error_j"]
        if (
            time < 0
            or not self.budget["minimum_step_s"] <= proposal <= self.max_step
            or abs(change - energy) > tol
            or abs(scaled_energy * self.scale - energy) > tol
            or abs(change - energy - residual) > tol
            or abs(self.initial_energy + change - stored) > tol
        ):
            raise NetworkError("Checkpoint energy/time/adaptive state mismatch")
        state = np.r_[increment, scaled_energy]
        result = self._observation(state, time, proposal)
        self._state = state
        self._time = time
        self._proposal = proposal
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(spec, indent=2) + "\n")
    try:
        model = NonlinearThermalGraph(spec)
        if args.resume:
            model.restore(json.loads(args.resume.read_text()))
        report = dict(
            passed=True,
            initial=model.snapshot(),
            final=model.advance(args.duration_s),
            signature=model.captured,
            validated=False,
            plant_referenced=False,
        )
        (args.output / "checkpoint.json").write_text(
            json.dumps(model.checkpoint(), indent=2, allow_nan=False) + "\n"
        )
        code = 0
    except NetworkError as error:
        report = dict(passed=False, diagnostics=getattr(error, "report", dict(message=str(error))))
        code = 2
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = dict(
        signature=signature(),
        files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        exit_code=code,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
