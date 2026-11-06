"""Versioned typed topology and constant-coefficient residual compiler.

Physical connector flows are positive INTO their owning component. Structural
matching is necessary, not a proof of numerical rank or of a general DAE index.
No equipment, pressure boundary or closure is synthesized by this compiler.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from core.status import ModelPassport


class NetworkError(ValueError):
    """Invalid contract, topology or residual before any integration."""


FIELDS = MappingProxyType(
    {
        "hydraulic": (
            ("pressure_pa", "Pa", False, 1e6),
            ("mass_flow_kg_s", "kg/s", True, 1.0),
            ("energy_flow_w", "W", True, 1e6),
        ),
        "thermal": (("temperature_k", "K", False, 300.0), ("heat_flow_w", "W", True, 100.0)),
        "mechanical": (("omega_rad_s", "rad/s", False, 100.0), ("torque_nm", "N m", True, 100.0)),
    }
)


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NetworkError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise NetworkError(f"{name} must be positive")
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise NetworkError(f"Invalid identifier: {value!r}")
    return value


@dataclass(frozen=True)
class PortSpec:
    name: str
    kind: str
    direction: str = "bidirectional"
    required: bool = True
    medium: str | None = None
    energy_convention: str | None = None
    signal_unit: str | None = None

    def __post_init__(self):
        identifier(self.name)
        if type(self.required) is not bool:
            raise NetworkError("Port required flag must be boolean")
        if self.kind == "signal":
            if (
                self.direction not in ("input", "output")
                or not isinstance(self.signal_unit, str)
                or not self.signal_unit.strip()
            ):
                raise NetworkError("Signal requires input/output direction and exact unit")
        elif (
            self.kind not in FIELDS
            or self.direction != "bidirectional"
            or self.signal_unit is not None
        ):
            raise NetworkError(
                "Physical ports require a known kind and bidirectional inward-flow convention"
            )
        if self.kind == "hydraulic":
            if (
                not isinstance(self.medium, str)
                or not self.medium
                or self.energy_convention not in ("total_enthalpy", "specific_enthalpy_transport")
            ):
                raise NetworkError("Hydraulic medium and energy convention must be explicit")
        elif self.medium is not None or self.energy_convention is not None:
            raise NetworkError("Medium and energy convention belong to hydraulic ports")

    @property
    def fields(self):
        return (
            (("value", self.signal_unit, False, 1.0),)
            if self.kind == "signal"
            else FIELDS[self.kind]
        )


@dataclass(frozen=True)
class StateSpec:
    name: str
    unit: str
    differential: bool
    nominal: float

    def __post_init__(self):
        identifier(self.name)
        if (
            not isinstance(self.unit, str)
            or not self.unit.strip()
            or type(self.differential) is not bool
        ):
            raise NetworkError("Invalid variable unit/differential flag")
        number(self.nominal, "Variable nominal", positive=True)


@dataclass(frozen=True)
class RowSpec:
    name: str
    unit: str
    nominal: float
    tolerance: float

    def __post_init__(self):
        identifier(self.name)
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise NetworkError("Residual unit required")
        number(self.nominal, "Residual nominal", positive=True)
        number(self.tolerance, "Residual absolute tolerance", positive=True)


@dataclass(frozen=True)
class LinearComponentType:
    passport: ModelPassport
    ports: tuple
    states: tuple
    rows: tuple
    parameters: tuple
    build: object

    def __post_init__(self):
        for collection in (self.ports, self.states, self.rows):
            if not isinstance(collection, tuple) or len({v.name for v in collection}) != len(
                collection
            ):
                raise NetworkError("Contract members must be unique immutable tuples")
        if not isinstance(self.parameters, tuple) or len(set(self.parameters)) != len(
            self.parameters
        ):
            raise NetworkError("Parameter names must be unique")
        for name in self.parameters:
            identifier(name)
        if not callable(self.build):
            raise NetworkError("Linear matrix builder required")

    @property
    def variables(self):
        result = [("state." + s.name, s.unit, s.differential, s.nominal) for s in self.states]
        for port in self.ports:
            result.extend(
                (f"port.{port.name}.{field}", unit, False, nominal)
                for field, unit, _, nominal in port.fields
            )
        return tuple(result)


class CompiledNetwork:
    """Immutable layout with a pure F=M*ydot+K*y-b evaluation interface."""

    def __init__(self, canonical, variables, rows, ports, links, mass, stiffness, source):
        self._canonical = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.topology_sha256 = hashlib.sha256(self._canonical.encode()).hexdigest()
        self.variables = tuple(variables)
        self.rows = tuple(rows)
        self.index = MappingProxyType({v[0]: i for i, v in enumerate(variables)})
        self.ports = MappingProxyType(dict(ports))
        self.links = tuple(links)
        self._mass, self._stiffness, self._source = mass.copy(), stiffness.copy(), source.copy()
        self.differential = tuple(v[2] for v in variables)
        signature = dict(
            topology=canonical,
            variables=variables,
            rows=rows,
            ports={key: asdict(port) for key, port in ports.items()},
            mass=mass.tolist(),
            stiffness=stiffness.tolist(),
            source=source.tolist(),
        )
        self.assembly_sha256 = hashlib.sha256(
            json.dumps(signature, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.reduction()  # Constant-coefficient numerical initialization rank, not just pattern.

    def specification(self):
        return json.loads(self._canonical)

    def matrices(self):
        return self._mass.copy(), self._stiffness.copy(), self._source.copy()

    def residual(self, state, rate):
        state, rate = np.asarray(state, dtype=float), np.asarray(rate, dtype=float)
        n = len(self.variables)
        if (
            state.shape != (n,)
            or rate.shape != (n,)
            or not np.isfinite(state).all()
            or not np.isfinite(rate).all()
        ):
            raise NetworkError("Residual vectors have invalid shape or nonfinite values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self._mass @ rate + self._stiffness @ state - self._source
        if not np.isfinite(result).all():
            raise NetworkError("Nonfinite residual")
        return result

    def require_residual(self, state, rate):
        residual = self.residual(state, rate)
        failures = {
            row[0]: float(value) for row, value in zip(self.rows, residual) if abs(value) > row[3]
        }
        if failures:
            raise NetworkError(f"Dimensional residual tolerance exceeded: {failures}")
        return residual

    def reduction(self):
        """Eliminate algebraic unknowns for a constant, uniquely initializable system.

        Solve [K_A | M_D] [y_A; ydot_D] = b-K_D*y_D, with dimensional
        row/column scaling. A singular initialization problem is rejected.
        """
        d = np.flatnonzero(self.differential)
        a = np.flatnonzero(~np.asarray(self.differential))
        jac = np.column_stack((self._stiffness[:, a], self._mass[:, d]))
        row_scale = np.array([r[2] for r in self.rows])
        col_scale = np.array([self.variables[i][3] for i in np.r_[a, d]])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = jac * col_scale[None, :] / row_scale[:, None]
        if not np.isfinite(scaled).all():
            raise NetworkError("Initialization scaling overflow")
        singular = np.linalg.svd(scaled, compute_uv=False)
        if singular[-1] <= singular[0] * 1e-12:
            raise NetworkError("Numerically singular or ill-conditioned initialization Jacobian")
        rhs = np.column_stack((self._source, -self._stiffness[:, d])) / row_scale[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            solved = np.linalg.solve(scaled, rhs) * col_scale[:, None]
        if not np.isfinite(solved).all():
            raise NetworkError("Nonfinite initialization solution")
        offset = np.zeros(len(self.variables))
        projection = np.zeros((len(self.variables), len(d)))
        offset[a] = solved[: len(a), 0]
        projection[a, :] = solved[: len(a), 1:]
        projection[d, :] = np.eye(len(d))
        return solved[len(a) :, 1:].copy(), solved[len(a) :, 0].copy(), projection, offset

    def initialize(self, fixed):
        keys = [v[0] for v in self.variables if v[2]]
        if not isinstance(fixed, dict) or set(fixed) != set(keys):
            raise NetworkError("All and only differential initial states must be supplied")
        z = np.array([number(fixed[k], k) for k in keys])
        matrix, drive, projection, offset = self.reduction()
        state = projection @ z + offset
        rate = projection @ (matrix @ z + drive)
        self.require_residual(state, rate)
        return state, rate

    def describe(self):
        return dict(
            schema_version=1,
            topology_sha256=self.topology_sha256,
            assembly_sha256=self.assembly_sha256,
            variables=[
                dict(name=v[0], unit=v[1], differential=v[2], nominal=v[3]) for v in self.variables
            ],
            equations=[
                dict(name=r[0], unit=r[1], nominal=r[2], absolute_tolerance=r[3]) for r in self.rows
            ],
            ports={
                key: dict(**asdict(port), fields=port.fields) for key, port in self.ports.items()
            },
            structural_matching="complete",
            initialization_rank="full at declared 1e-12 scaled threshold",
            scope="Registered constant-coefficient models only; no generic nonlinear DAE qualification",
        )

    def route_flux(self, link_id, values):
        matches = [v for v in self.links if v[0] == link_id]
        if len(matches) != 1:
            raise NetworkError("Unknown connection")
        _, left, right = matches[0]
        port = self.ports[left]
        if port.kind == "signal":
            raise NetworkError("Signals are not conserved fluxes")
        fields = [f for f, _, flow, _ in port.fields if flow]
        if not isinstance(values, dict) or set(values) != set(fields):
            raise NetworkError("All and only conserved connector fields required")
        checked = {f: number(values[f], f) for f in fields}
        return {left: {f: -v for f, v in checked.items()}, right: checked.copy()}


def compile_network(spec, catalog):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"schema_version", "nodes", "connections"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise NetworkError("Topology schema 1 with nodes and connections required")
    if (
        not isinstance(spec["nodes"], list)
        or not spec["nodes"]
        or not isinstance(spec["connections"], list)
    ):
        raise NetworkError("Nonempty node list and connection list required")
    # Explicit v1 resource bound; no claim of a large sparse equipment solver.
    if len(spec["nodes"]) > 64 or len(spec["connections"]) > 128:
        raise NetworkError("Linear compiler v1 supports at most 64 nodes / 128 connections")
    nodes, ports, variables, rows, blocks = {}, {}, {}, [], []
    canonical_nodes = []
    for raw in spec["nodes"]:
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "model_id",
            "model_version",
            "parameters",
        }:
            raise NetworkError("Unknown or missing node fields")
        name = identifier(raw["id"])
        if name in nodes:
            raise NetworkError("Duplicate node: " + name)
        if not isinstance(raw["model_id"], str) or not isinstance(raw["model_version"], str):
            raise NetworkError("Component passport ID/version must be strings")
        contract = catalog.get(raw["model_id"])
        if (
            contract is None
            or raw["model_version"] != contract.passport.version
            or raw["model_id"] != contract.passport.model_id
        ):
            raise NetworkError(
                "Unknown or incompatible component passport: " + str(raw["model_id"])
            )
        params = raw["parameters"]
        if not isinstance(params, dict) or set(params) != set(contract.parameters):
            raise NetworkError("Component parameters must match its registered contract")
        params = {k: number(v, k) for k, v in params.items()}
        local = contract.variables
        if len(variables) + len(local) > 512 or len(rows) + len(contract.rows) > 512:
            raise NetworkError("Linear compiler v1 supports at most 512 unknowns/equations")
        matrices = tuple(
            np.asarray(v, dtype=float).copy() for v in contract.build(MappingProxyType(params))
        )
        if len(matrices) != 3:
            raise NetworkError("Builder must return M,K,b")
        mass, stiffness, source = matrices
        shape = (len(contract.rows), len(local))
        if (
            mass.shape != shape
            or stiffness.shape != shape
            or source.shape != (len(contract.rows),)
            or not all(np.isfinite(v).all() for v in matrices)
        ):
            raise NetworkError("Builder matrix shape/nonfinite contract violation")
        if any(np.any(mass[:, i] != 0) for i, v in enumerate(local) if not v[2]):
            raise NetworkError("Algebraic variables cannot carry undeclared derivatives")
        indices = []
        for key, unit, diff, nominal in local:
            global_key = f"{name}.{key}"
            variables[global_key] = (global_key, unit, diff, nominal)
            indices.append(len(variables) - 1)
        start = len(rows)
        rows.extend((f"{name}.eq.{r.name}", r.unit, r.nominal, r.tolerance) for r in contract.rows)
        blocks.append((start, indices, mass, stiffness, source))
        nodes[name] = contract
        for port in contract.ports:
            ports[f"{name}.{port.name}"] = port
        canonical_nodes.append(
            dict(
                id=name,
                model_id=contract.passport.model_id,
                model_version=contract.passport.version,
                parameters=params,
            )
        )
    variable_list = list(variables.values())
    if len(variable_list) > 512:
        raise NetworkError("Linear compiler v1 supports at most 512 unknowns")
    index = {v[0]: i for i, v in enumerate(variable_list)}
    usage = {p: 0 for p in ports}
    links = []
    ids = set()
    adjacency = {n: set() for n in nodes}
    connector_rows = []
    for raw in spec["connections"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "from", "to"}:
            raise NetworkError("Unknown or missing connection fields")
        name = identifier(raw["id"])
        if name in ids:
            raise NetworkError("Duplicate connection: " + name)
        ids.add(name)
        left, right = raw["from"], raw["to"]
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left not in ports
            or right not in ports
        ):
            raise NetworkError("Connection references unknown port")
        a, b = ports[left], ports[right]
        na, nb = left.split(".")[0], right.split(".")[0]
        if na == nb:
            raise NetworkError("Self connections unsupported; use an explicit component")
        if (a.kind, a.fields, a.medium, a.energy_convention) != (
            b.kind,
            b.fields,
            b.medium,
            b.energy_convention,
        ):
            raise NetworkError("Port kind, unit, medium or energy convention mismatch")
        if a.kind == "signal" and (a.direction, b.direction) != ("output", "input"):
            raise NetworkError("Signal must connect output to input")
        for endpoint, port in ((left, a), (right, b)):
            usage[endpoint] += 1
            if usage[endpoint] > 1 and not (port.kind == "signal" and port.direction == "output"):
                raise NetworkError("Port has multiple drivers/connections: " + endpoint)
        adjacency[na].add(nb)
        adjacency[nb].add(na)
        links.append((name, left, right))
        for field, unit, flow, nominal in a.fields:
            ia = index[f"{na}.port.{left.split('.')[1]}.{field}"]
            ib = index[f"{nb}.port.{right.split('.')[1]}.{field}"]
            connector_rows.append((ia, ib, 1.0 if flow else -1.0))
            rows.append((f"connection.{name}.{field}", unit, nominal, 1e-7 if flow else 1e-9))
    missing = [p for p in ports if ports[p].required and usage[p] == 0]
    if missing:
        raise NetworkError("Unconnected required ports: " + str(missing))
    reached = set()
    pending = [next(iter(nodes))]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(adjacency[node] - reached)
    if reached != set(nodes):
        raise NetworkError("Disconnected component islands")
    n = len(variable_list)
    if len({row[0] for row in rows}) != len(rows):
        raise NetworkError("Duplicate global equation identifier; rename component or connection")
    if len(rows) != n or n == 0:
        raise NetworkError(f"Residual is not square: {len(rows)} equations / {n} unknowns")
    mass = np.zeros((n, n))
    stiffness = np.zeros((n, n))
    source = np.zeros(n)
    for start, columns, m, k, b in blocks:
        sl = np.arange(start, start + len(b))
        mass[np.ix_(sl, columns)] = m
        stiffness[np.ix_(sl, columns)] = k
        source[sl] = b
    start = sum(len(b[-1]) for b in blocks)
    for i, (a, b, sign) in enumerate(connector_rows):
        stiffness[start + i, a] = 1
        stiffness[start + i, b] = sign
    pattern = csr_matrix((mass != 0) | (stiffness != 0))
    if np.any(maximum_bipartite_matching(pattern, perm_type="column") < 0):
        raise NetworkError("Structurally rank-deficient residual pattern")
    canonical = dict(
        schema_version=1,
        nodes=canonical_nodes,
        connections=[dict(id=i, **{"from": a, "to": b}) for i, a, b in links],
    )
    return CompiledNetwork(canonical, variable_list, rows, ports, links, mass, stiffness, source)
