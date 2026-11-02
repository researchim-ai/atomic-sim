"""Transactional finite-volume thermal dynamics on a fixed hydraulic graph.

Fixed rho/cp, inventories and flows; an affine ODE is advanced by an augmented
matrix exponential. Domain admission bounds the entire interval, not just its
endpoint. No dynamic hydraulic adapter, EOS feedback or event-time claim.
"""

import argparse
import hashlib
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

from core.network import NetworkError, number
from scripts.solve_hydraulic_graph import _keys, sha
from scripts.solve_thermal_graph import prepare
from sdk.experiment import _source_fingerprint

BUDGET = Path("validation/ai05_transient_graph_budget_v1.json")
DRIVERS = (
    "scripts/transient_thermal_graph.py",
    "scripts/solve_thermal_graph.py",
    "scripts/solve_hydraulic_graph.py",
)
BUDGETS = (
    str(BUDGET),
    "validation/ai05_thermal_graph_budget_v1.json",
    "validation/ai05_hydraulic_graph_budget_v1.json",
)


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={p: sha(p) for p in DRIVERS},
        budgets={p: sha(p) for p in BUDGETS},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


def digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError) as error:
        raise NetworkError("Finite JSON-compatible checkpoint/input data required") from error
    return hashlib.sha256(encoded).hexdigest()


class TransientGraphError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


class ThermalGraph:
    def __init__(self, spec):
        _keys(
            spec,
            (
                "format",
                "schema_version",
                "steady_input",
                "volumes_m3",
                "initial_temperatures_k",
                "max_step_s",
            ),
        )
        if (
            spec["format"] != "atomic-thermal-graph-transient"
            or type(spec["schema_version"]) is not int
            or spec["schema_version"] != 1
        ):
            raise NetworkError("Expected atomic-thermal-graph-transient schema 1")
        self.budget = json.loads(BUDGET.read_text())
        self.spec = deepcopy(spec)
        self.identity = digest(spec)
        self.hydraulic, self.thermal = prepare(spec["steady_input"])
        self.names = [n["id"] for n in self.hydraulic["input"]["nodes"]]
        for key in ("volumes_m3", "initial_temperatures_k"):
            if not isinstance(spec[key], dict) or set(spec[key]) != set(self.names):
                raise NetworkError(
                    "Explicit volumes and initial temperatures must match every node"
                )
        self.max_step = number(spec["max_step_s"], "max_step_s", positive=True)
        if self.max_step > self.budget["maximum_step_s"]:
            raise NetworkError("Transient maximum step exceeds budget")
        volumes = np.array(
            [number(spec["volumes_m3"][k], "volume_m3", positive=True) for k in self.names]
        )
        self.cp = self.thermal["heat_capacity_j_kg_k"]
        self.rho = self.hydraulic["input"]["density_kg_m3"]
        self.ref = self.thermal["enthalpy_reference_temperature_k"]
        self.capacity = volumes * self.rho * self.cp
        if not np.all(np.isfinite(self.capacity)) or np.any(self.capacity <= 0):
            raise NetworkError("Heat capacities exceed finite arithmetic")
        self._temperature = np.array(
            [
                number(spec["initial_temperatures_k"][k], "initial_temperature_k", positive=True)
                for k in self.names
            ]
        )
        self._time = 0.0
        self._offset = self._temperature.copy()
        self._deviation = np.zeros(len(self.names))
        self._energy_input = 0.0
        self._initial_energy = float(self.capacity @ (self._temperature - self.ref))
        self._energy_scale = float(self.capacity.sum())
        if not np.isfinite(self._initial_energy) or not np.isfinite(self._energy_scale):
            raise NetworkError("Initial energy exceeds finite arithmetic")
        self.captured = signature()
        self._cache = {}
        self._assemble()
        self._domain(self._temperature)

    def _assemble(self):
        n = len(self.names)
        index = {name: i for i, name in enumerate(self.names)}
        matrix = np.zeros((n, n))
        forcing = np.zeros(n)
        inflow = np.zeros(n)
        outflow = np.zeros(n)
        deposited = np.zeros(n)
        self._stream_rise = np.zeros(n)
        for edge in self.hydraulic["input"]["edges"]:
            m = self.hydraulic["flows_kg_s"][edge["id"]]
            donor, receiver = (
                (index[edge["from"]], index[edge["to"]])
                if m >= 0
                else (index[edge["to"]], index[edge["from"]])
            )
            rate = abs(m)
            matrix[donor, donor] -= rate * self.cp
            matrix[receiver, donor] += rate * self.cp
            outflow[donor] += rate
            inflow[receiver] += rate
            loss = (
                edge["linear_pa_s_kg"] * m * m + edge["quadratic_pa_s2_kg2"] * abs(m) ** 3
            ) / self.rho
            deposited[receiver] += loss
            if m != 0:
                self._stream_rise[donor] = max(
                    self._stream_rise[donor],
                    (edge["linear_pa_s_kg"] * rate + edge["quadratic_pa_s2_kg2"] * m * m)
                    / (self.rho * self.cp),
                )
        for name, i in index.items():
            node = self.thermal["nodes"][name]
            m = self.hydraulic["external_injections_kg_s"][name]
            matrix[i, i] -= max(-m, 0) * self.cp + node["ambient_ua_w_k"]
            forcing[i] = (
                node["heat_power_w"]
                + node["ambient_ua_w_k"] * node["ambient_temperature_k"]
                + deposited[i]
            )
            if m > 0:
                forcing[i] += m * self.cp * node["supply_temperature_k"]
            # Absolute-temperature matrix includes sensible-reference and pressure
            # transport terms from the small admitted hydraulic mass imbalance.
            imbalance = inflow[i] - outflow[i] + m
            forcing[i] += (
                self.hydraulic["pressures_pa"][name] / self.rho - self.cp * self.ref
            ) * imbalance
        self._a = matrix / self.capacity[:, None]
        self._b = forcing / self.capacity
        if not np.all(np.isfinite(self._a)) or not np.all(np.isfinite(self._b)):
            raise NetworkError("Nonfinite transient equations")
        self._power_linear = matrix.sum(axis=0)
        self._power_constant = float(forcing.sum())
        self._lower = np.full(n, self.thermal["min_temperature_k"])
        self._upper = self.thermal["max_temperature_k"] - self._stream_rise
        if np.any(self._upper <= self._lower):
            raise TransientGraphError(
                "stream_domain", "No admitted fluid temperature exists before stream heating"
            )
        self._positive = np.maximum(self._a, 0)
        self._negative = np.minimum(self._a, 0)
        # On the entire open domain, these are conservative componentwise rate
        # bounds. They are used solely to certify that no interior excursion can
        # be hidden by an admissible final endpoint.
        self._rate_low = self._positive @ self._lower + self._negative @ self._upper + self._b
        self._rate_high = self._positive @ self._upper + self._negative @ self._lower + self._b

    def _domain(self, value):
        if (
            not np.all(np.isfinite(value))
            or np.any(value <= self._lower)
            or np.any(value >= self._upper)
        ):
            raise TransientGraphError(
                "temperature_domain", "Node or outgoing stream exceeds its open temperature domain"
            )

    def _certified(self, value, step):
        return bool(
            np.all(value + step * np.minimum(self._rate_low, 0) > self._lower)
            and np.all(value + step * np.maximum(self._rate_high, 0) < self._upper)
        )

    def _transition(self, step):
        if step not in self._cache:
            n = len(self.names)
            augmented = np.zeros((n + 2, n + 2))
            augmented[:n, :n] = self._a
            augmented[:n, n] = self._a @ self._offset + self._b
            augmented[n + 1, :n] = self._power_linear / self._energy_scale
            augmented[n + 1, n] = (
                float(self._power_linear @ self._offset + self._power_constant) / self._energy_scale
            )
            if len(self._cache) >= 64:
                self._cache.clear()
            self._cache[step] = expm(augmented * step)
        return self._cache[step]

    def rates(self, temperatures_k=None):
        if temperatures_k is not None and (
            not isinstance(temperatures_k, dict) or set(temperatures_k) != set(self.names)
        ):
            raise NetworkError("Explicit rate-evaluation temperatures must match every node")
        value = (
            self._temperature
            if temperatures_k is None
            else np.array([number(temperatures_k[k], "temperature_k") for k in self.names])
        )
        self._domain(value)
        return {name: float(v) for name, v in zip(self.names, self._a @ value + self._b)}

    def snapshot(self):
        return dict(
            time_s=self._time,
            temperatures_k={name: float(v) for name, v in zip(self.names, self._temperature)},
            temperature_increments_k={
                name: float(v) for name, v in zip(self.names, self._deviation)
            },
            stored_energy_j=self._initial_energy + float(self.capacity @ self._deviation),
            integrated_external_energy_j=self._energy_input,
            energy_residual_j=float(self.capacity @ self._deviation) - self._energy_input,
        )

    def advance(self, duration_s):
        duration = number(duration_s, "duration_s", positive=True)
        if duration > self.budget["maximum_duration_s"]:
            raise NetworkError("Duration exceeds transient budget")
        target_time = self._time + duration
        if not np.isfinite(target_time) or target_time <= self._time:
            raise NetworkError("Requested time increment is not representable at current time")
        if signature() != self.captured:
            raise TransientGraphError("source_drift", "Transient drivers or budgets changed")
        value = self._temperature.copy()
        deviation = self._deviation.copy()
        energy = self._energy_input
        remaining = duration
        steps = 0
        while remaining > 0:
            if steps >= self.budget["maximum_internal_steps"]:
                raise TransientGraphError(
                    "step_budget",
                    "Certified interval step budget exhausted; full advance rolled back",
                )
            step = min(remaining, self.max_step)
            while not self._certified(value, step):
                step *= 0.5
                if step < self.budget["minimum_internal_step_s"]:
                    raise TransientGraphError(
                        "interval_domain",
                        "Cannot certify complete interval inside domain; full advance rolled back",
                        candidate_elapsed_s=duration - remaining,
                    )
            extended = self._transition(step) @ np.r_[deviation, 1.0, 0.0]
            deviation = extended[: len(self.names)]
            value = self._offset + deviation
            energy += float(extended[-1]) * self._energy_scale
            self._domain(value)
            remaining = max(0.0, remaining - step)
            steps += 1
        residual = float(self.capacity @ deviation) - energy
        if not np.isfinite(residual) or abs(residual) > self.budget["maximum_energy_error_j"]:
            raise TransientGraphError(
                "energy_residual",
                "Transient energy exceeds absolute budget; full advance rolled back",
                residual_j=residual,
            )
        if signature() != self.captured:
            raise TransientGraphError(
                "source_drift", "Transient sources changed during advance; full advance rolled back"
            )
        self._temperature = value
        self._deviation = deviation
        self._time = target_time
        self._energy_input = energy
        result = self.snapshot()
        result["internal_steps"] = steps
        return result

    def checkpoint(self):
        payload = dict(
            schema_version=1,
            model="transient-thermal-graph-v1",
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
            or p["model"] != "transient-thermal-graph-v1"
            or p["input_sha256"] != self.identity
            or p["signature"] != self.captured
            or signature() != self.captured
        ):
            raise NetworkError("Checkpoint model/input/source identity mismatch")
        state = p["state"]
        _keys(
            state,
            (
                "time_s",
                "temperatures_k",
                "temperature_increments_k",
                "stored_energy_j",
                "integrated_external_energy_j",
                "energy_residual_j",
            ),
        )
        if not isinstance(state["temperatures_k"], dict) or set(state["temperatures_k"]) != set(
            self.names
        ):
            raise NetworkError("Checkpoint temperature map mismatch")
        value = np.array(
            [number(state["temperatures_k"][k], "checkpoint_temperature") for k in self.names]
        )
        if not isinstance(state["temperature_increments_k"], dict) or set(
            state["temperature_increments_k"]
        ) != set(self.names):
            raise NetworkError("Checkpoint increment map mismatch")
        deviation = np.array(
            [
                number(state["temperature_increments_k"][k], "checkpoint_increment")
                for k in self.names
            ]
        )
        if not np.array_equal(value, self._offset + deviation):
            raise NetworkError("Checkpoint temperatures disagree with stored increments")
        self._domain(value)
        time = number(state["time_s"], "checkpoint_time")
        energy = number(state["integrated_external_energy_j"], "checkpoint_energy")
        stored = number(state["stored_energy_j"], "stored_energy")
        residual = number(state["energy_residual_j"], "energy_residual")
        change = float(self.capacity @ deviation)
        actual = self._initial_energy + change
        tol = self.budget["maximum_energy_error_j"]
        if (
            time < 0
            or abs(actual - stored) > tol
            or abs(change - energy) > tol
            or abs(change - energy - residual) > tol
        ):
            raise NetworkError("Checkpoint energy consistency failed")
        self._temperature = value.copy()
        self._deviation = deviation.copy()
        self._time = time
        self._energy_input = energy
        return self.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(spec, indent=2) + "\n")
    try:
        model = ThermalGraph(spec)
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
