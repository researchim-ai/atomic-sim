"""Executable closed thermal network for the typed linear assembly contract.

Constant heat capacities, reciprocal positive conductances, nonnegative external
powers. No hydraulics, radiation, phase change, shafts or whole-plant claim.
"""

import numpy as np
from scipy.linalg import expm

from core.network import (
    LinearComponentType,
    NetworkError,
    PortSpec,
    RowSpec,
    StateSpec,
    compile_network,
    number,
)
from core.status import ModelPassport, ModelStatus

CAPACITY = ModelPassport(
    model_id="linear-thermal-capacity",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Constant heat capacity with one conservative thermal port",
    approximations=(
        "E=C*T relative to fixed reference",
        "constant positive C; nonnegative external power",
    ),
    applicability="Closed lumped thermal networks; positive absolute temperatures",
    data_sources=("C dT/dt = Q_port + P_external",),
)
CONDUCTANCE = ModelPassport(
    model_id="linear-thermal-conductance",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Reciprocal linear heat conductor without storage",
    approximations=("constant positive UA", "no storage or contact resistance dynamics"),
    applicability="Two conservative thermal ports in lumped networks",
    data_sources=("Q_left=UA*(T_left-T_right); Q_right=-Q_left",),
)

JUNCTION = ModelPassport(
    model_id="linear-thermal-junction3",
    version="1.0.0",
    status=ModelStatus.IMPLEMENTED,
    phenomenon="Three-port zero-storage thermal junction",
    approximations=("common interface temperature", "sum of inward heat powers is zero"),
    applicability="Closed lumped thermal networks; no hydraulic mixing model",
    data_sources=("T_a=T_b=T_c; Q_a+Q_b+Q_c=0",),
)


def junction_matrices(_params):
    return (
        np.zeros((3, 6)),
        np.array([[-1, 0, 1, 0, 0, 0], [-1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1]]),
        np.zeros(3),
    )


def capacity_matrices(params):
    capacity = number(params["heat_capacity_j_k"], "heat_capacity_j_k", positive=True)
    power = number(params["source_w"], "source_w")
    if power < 0:
        raise NetworkError("Thermal v1 requires nonnegative external powers")
    # Local columns: state.T, port.thermal.T, port.thermal.Q (inward).
    return (
        np.array([[capacity, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, -1], [-1, 1, 0]]),
        np.array([power, 0]),
    )


def conductance_matrices(params):
    ua = number(params["ua_w_k"], "ua_w_k", positive=True)
    # left.T,left.Q,right.T,right.Q; both Q are inward to conductor.
    return (np.zeros((2, 4)), np.array([[-ua, 1, ua, 0], [0, 1, 0, 1]]), np.zeros(2))


def thermal_catalog():
    return {
        JUNCTION.model_id: LinearComponentType(
            JUNCTION,
            tuple(PortSpec(p, "thermal") for p in ("a", "b", "c")),
            (),
            (
                RowSpec("temperature_b", "K", 300, 1e-9),
                RowSpec("temperature_c", "K", 300, 1e-9),
                RowSpec("balance", "W", 100, 1e-7),
            ),
            (),
            junction_matrices,
        ),
        CAPACITY.model_id: LinearComponentType(
            CAPACITY,
            (PortSpec("thermal", "thermal"),),
            (StateSpec("temperature_k", "K", True, 300),),
            (RowSpec("energy", "W", 100, 1e-7), RowSpec("port_temperature", "K", 300, 1e-9)),
            ("heat_capacity_j_k", "source_w"),
            capacity_matrices,
        ),
        CONDUCTANCE.model_id: LinearComponentType(
            CONDUCTANCE,
            (PortSpec("left", "thermal"), PortSpec("right", "thermal")),
            (),
            (RowSpec("transfer", "W", 100, 1e-7), RowSpec("balance", "W", 100, 1e-7)),
            ("ua_w_k",),
            conductance_matrices,
        ),
    }


class LinearThermalNetwork:
    model_id = "compiled-linear-thermal-network"
    model_version = "1.0.0"

    def __init__(self, topology, initial_temperature_k, *, energy_tolerance_j=1e-5):
        self.compiled = compile_network(topology, thermal_catalog())
        self._topology = self.compiled.specification()
        capacities = [n for n in self._topology["nodes"] if n["model_id"] == CAPACITY.model_id]
        self._names = tuple(n["id"] for n in capacities)
        if (
            not self._names
            or not isinstance(initial_temperature_k, dict)
            or set(initial_temperature_k) != set(self._names)
        ):
            raise NetworkError("Initial temperature required for every and only capacity node")
        self._capacity = np.array([n["parameters"]["heat_capacity_j_k"] for n in capacities])
        self._power = float(sum(n["parameters"]["source_w"] for n in capacities))
        self.energy_tolerance_j = number(energy_tolerance_j, "energy_tolerance_j", positive=True)
        self._initial = np.array(
            [number(initial_temperature_k[n], n, positive=True) for n in self._names]
        )
        self._temperature = self._initial.copy()
        self.time_s = 0.0
        self._matrix, self._drive, self._projection, self._offset = self.compiled.reduction()
        # Assert the physical conservation property independently of the graph pattern.
        if (
            np.max(abs(self._capacity @ self._matrix)) > 1e-7
            or abs(self._capacity @ self._drive - self._power) > 1e-7
        ):
            raise NetworkError("Thermal reduction violates global energy conservation")
        off_diagonal = self._matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        if np.any(off_diagonal < -1e-12) or np.any(self._drive < -1e-12):
            raise NetworkError("Thermal v1 reduction is outside its positive-system scope")
        self.initial_energy_j = float(self._capacity @ self._initial)
        self._observe_at(self._temperature, self.time_s)

    @property
    def temperature_k(self):
        return dict(zip(self._names, map(float, self._temperature)))

    def _observe_at(self, temperature, time_s):
        if (
            temperature.shape != self._initial.shape
            or not np.isfinite(temperature).all()
            or np.any(temperature <= 0)
        ):
            raise NetworkError("Invalid absolute temperatures")
        state = self._projection @ temperature + self._offset
        rate = self._projection @ (self._matrix @ temperature + self._drive)
        residual = self.compiled.require_residual(state, rate)
        energy = float(self._capacity @ temperature)
        defect = energy - self.initial_energy_j - self._power * time_s
        if not np.isfinite(defect) or abs(defect) > self.energy_tolerance_j:
            raise NetworkError("Global energy ledger tolerance exceeded")
        return dict(
            time_s=time_s,
            stored_energy_j=energy,
            external_energy_j=self._power * time_s,
            energy_residual_j=defect,
            temperature_k=dict(zip(self._names, map(float, temperature))),
            variables={v[0]: float(x) for v, x in zip(self.compiled.variables, state)},
            residuals={r[0]: float(x) for r, x in zip(self.compiled.rows, residual)},
        )

    def observation(self):
        return self._observe_at(self._temperature, self.time_s)

    def advance(self, dt_s):
        dt_s = number(dt_s, "dt_s", positive=True)
        target = self.time_s + dt_s
        if not np.isfinite(target) or target <= self.time_s:
            raise NetworkError("Target time is nonfinite or cannot advance at requested precision")
        size = len(self._temperature)
        augmented = np.zeros((size + 1, size + 1))
        augmented[:size, :size] = self._matrix
        augmented[:size, -1] = self._drive
        with np.errstate(over="ignore", invalid="ignore"):
            proposed = (expm(augmented * dt_s) @ np.r_[self._temperature, 1.0])[:size]
        observed = self._observe_at(proposed, target)
        # Commit only after all algebraic, domain and dimensional checks succeed.
        self._temperature, self.time_s = proposed, target
        return observed

    def snapshot(self):
        return dict(
            schema_version=1,
            model_id=self.model_id,
            model_version=self.model_version,
            topology=self.compiled.specification(),
            assembly_sha256=self.compiled.assembly_sha256,
            initial_temperature_k=dict(zip(self._names, map(float, self._initial))),
            temperature_k=self.temperature_k,
            time_s=self.time_s,
            energy_tolerance_j=self.energy_tolerance_j,
        )

    @classmethod
    def from_snapshot(cls, payload):
        fields = {
            "schema_version",
            "model_id",
            "model_version",
            "topology",
            "assembly_sha256",
            "initial_temperature_k",
            "temperature_k",
            "time_s",
            "energy_tolerance_j",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["model_id"] != cls.model_id
            or payload["model_version"] != cls.model_version
        ):
            raise NetworkError("Incompatible thermal network snapshot")
        model = cls(
            payload["topology"],
            payload["initial_temperature_k"],
            energy_tolerance_j=payload["energy_tolerance_j"],
        )
        if model.compiled.assembly_sha256 != payload["assembly_sha256"]:
            raise NetworkError("Checkpoint assembly fingerprint mismatch")
        if not isinstance(payload["temperature_k"], dict) or set(payload["temperature_k"]) != set(
            model._names
        ):
            raise NetworkError("Checkpoint capacity states mismatch")
        time_s = number(payload["time_s"], "time_s")
        if time_s < 0:
            raise NetworkError("Negative checkpoint time")
        proposed = np.array(
            [number(payload["temperature_k"][n], n, positive=True) for n in model._names]
        )
        model._observe_at(proposed, time_s)
        model._temperature, model.time_s = proposed, time_s
        return model
