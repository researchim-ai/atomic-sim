"""Steady, constant-density branched hydraulic graphs with fixed boundaries.

Engineering companion, not a dynamic network backend. Edge flows are positive
from->to; node injection is positive into the network. Strictly positive linear
resistance admits a differentiable monotone pressure solve, including zero flow.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, identifier, number

BUDGET = Path("validation/ai05_hydraulic_graph_budget_v1.json")


class HydraulicSolveError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


def _reject(criterion, message, **details):
    raise HydraulicSolveError(criterion, message, **details)


def sha(path):
    source = Path(path)
    if (
        not source.is_absolute()
        and source.parts
        and (
            source.parts[0] == "atomic_model_runtime"
            or (source.parts[0] == "scripts" and source.suffix == ".py")
        )
    ):
        from atomic_model_runtime.resources import resolve

        source = resolve(source)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _keys(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise NetworkError("Missing or unknown hydraulic graph fields")


def compile_graph(spec):
    """Validate a separate engineering wire format; never mutate caller input."""
    budget = json.loads(BUDGET.read_text())
    _keys(spec, ("format", "schema_version", "density_kg_m3", "gravity_m_s2", "nodes", "edges"))
    if (
        spec["format"] != "atomic-hydraulic-steady"
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise NetworkError("Expected atomic-hydraulic-steady schema 1")
    rho = number(spec["density_kg_m3"], "density_kg_m3", positive=True)
    gravity = number(spec["gravity_m_s2"], "gravity_m_s2")
    if gravity < 0:
        raise NetworkError("Gravity must be nonnegative")
    nodes, edges = spec["nodes"], spec["edges"]
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= budget["maximum_nodes"]:
        raise NetworkError("Hydraulic graph node limit exceeded or fewer than two nodes")
    if not isinstance(edges, list) or not 1 <= len(edges) <= budget["maximum_edges"]:
        raise NetworkError("Hydraulic graph edge limit exceeded or empty edges")
    names, normalized = {}, []
    for node in nodes:
        _keys(
            node,
            ("id", "elevation_m"),
            (
                "pressure_pa",
                "injection_kg_s",
                "initial_pressure_pa",
                "min_pressure_pa",
                "max_pressure_pa",
            ),
        )
        name = identifier(node["id"])
        if name in names:
            raise NetworkError("Duplicate hydraulic node")
        names[name] = len(names)
        if ("pressure_pa" in node) == ("injection_kg_s" in node):
            raise NetworkError("Node requires exactly one pressure or mass injection condition")
        v = {"id": name, "elevation_m": number(node["elevation_m"], "elevation_m")}
        for key in sorted(node.keys() - {"id", "elevation_m"}):
            v[key] = number(node[key], key, positive=key != "injection_kg_s")
        if "pressure_pa" in v and "initial_pressure_pa" in v:
            raise NetworkError("A fixed-pressure boundary cannot have a pressure guess")
        if (
            "min_pressure_pa" in v
            and "max_pressure_pa" in v
            and v["min_pressure_pa"] >= v["max_pressure_pa"]
        ):
            raise NetworkError("Empty pressure domain")
        normalized.append(v)
    n, m = len(nodes), len(edges)
    incidence = np.zeros((n, m))
    linear, quadratic, gain = np.empty(m), np.empty(m), np.empty(m)
    edge_names, normalized_edges, adjacency = set(), [], [set() for _ in nodes]
    for j, edge in enumerate(edges):
        _keys(
            edge, ("id", "from", "to", "linear_pa_s_kg", "quadratic_pa_s2_kg2", "pressure_gain_pa")
        )
        name = identifier(edge["id"])
        if (
            name in edge_names
            or not isinstance(edge["from"], str)
            or not isinstance(edge["to"], str)
            or edge["from"] not in names
            or edge["to"] not in names
            or edge["from"] == edge["to"]
        ):
            raise NetworkError("Duplicate edge, unknown endpoint or self-loop")
        edge_names.add(name)
        a, b = names[edge["from"]], names[edge["to"]]
        incidence[a, j], incidence[b, j] = 1, -1
        adjacency[a].add(b)
        adjacency[b].add(a)
        linear[j] = number(edge["linear_pa_s_kg"], "linear_pa_s_kg", positive=True)
        quadratic[j] = number(edge["quadratic_pa_s2_kg2"], "quadratic_pa_s2_kg2")
        head = number(edge["pressure_gain_pa"], "pressure_gain_pa")
        if quadratic[j] < 0:
            raise NetworkError("Quadratic resistance must be nonnegative")
        gain[j] = head - rho * gravity * (
            normalized[b]["elevation_m"] - normalized[a]["elevation_m"]
        )
        normalized_edges.append(
            dict(
                id=name,
                **{"from": edge["from"], "to": edge["to"]},
                linear_pa_s_kg=float(linear[j]),
                quadratic_pa_s2_kg2=float(quadratic[j]),
                pressure_gain_pa=head,
            )
        )
    seen, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(adjacency[i] - seen)
    if len(seen) != n:
        _reject("disconnected_graph", "Every node must belong to one connected graph")
    fixed = [i for i, v in enumerate(normalized) if "pressure_pa" in v]
    if not fixed:
        _reject("pressure_gauge", "At least one fixed pressure is required; no gauge is invented")
    free = [i for i in range(n) if i not in fixed]
    p = np.full(n, np.mean([normalized[i]["pressure_pa"] for i in fixed]))
    for i, v in enumerate(normalized):
        p[i] = v.get("pressure_pa", v.get("initial_pressure_pa", p[i]))
    if not np.all(np.isfinite(gain)) or not np.all(np.isfinite(p)):
        raise NetworkError("Hydraulic parameters exceed finite arithmetic")
    value = dict(
        format=spec["format"],
        schema_version=1,
        density_kg_m3=rho,
        gravity_m_s2=gravity,
        nodes=normalized,
        edges=normalized_edges,
    )
    return value, incidence, linear, quadratic, gain, fixed, free, p, budget


def edge_flows(pressure_drop, linear, quadratic):
    """Stable inverse of L*m + R*m*abs(m) = pressure_drop."""
    denominator = linear + np.hypot(linear, 2 * np.sqrt(quadratic) * np.sqrt(np.abs(pressure_drop)))
    return 2 * (pressure_drop / denominator)


def solve_hydraulic_graph(spec):
    value, a, linear, quadratic, gain, fixed, free, p, budget = compile_graph(spec)
    af = a[free]
    injections = np.array([value["nodes"][i]["injection_kg_s"] for i in free])
    initial = p.copy()
    conditions, backtracks, iterations = [], 0, 0

    def evaluate(pressure):
        flow = edge_flows(a.T @ pressure + gain, linear, quadratic)
        residual = af @ flow - injections
        if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(residual)):
            _reject("nonfinite", "Hydraulic evaluation left finite arithmetic")
        return flow, residual

    for iterations in range(budget["maximum_iterations"] + 1):
        flow, residual = evaluate(p)
        if free:
            conductance = 1 / (linear + 2 * quadratic * np.abs(flow))
            jacobian = (af * conductance) @ af.T
            diagonal = np.sqrt(np.diag(jacobian))
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0):
                _reject("hydraulic_rank", "Pressure Jacobian has an unanchored or unresolved row")
            scaled = jacobian / diagonal[:, None] / diagonal[None, :]
            condition = float(np.linalg.cond(scaled))
            conditions.append(condition)
            if not np.isfinite(condition) or condition > budget["maximum_scaled_condition"]:
                _reject(
                    "hydraulic_rank",
                    "Scaled pressure Jacobian exceeds condition budget",
                    condition=condition,
                )
        if not free or (
            np.max(np.abs(residual)) <= budget["maximum_mass_residual_kg_s"]
            and abs(float(residual.sum())) <= budget["maximum_global_mass_residual_kg_s"]
        ):
            break
        if iterations == budget["maximum_iterations"]:
            _reject(
                "iteration_limit", "Hydraulic mass residual did not converge", iterations=iterations
            )
        try:
            step = np.linalg.solve(scaled, -residual / diagonal) / diagonal
        except np.linalg.LinAlgError as exc:
            _reject("hydraulic_rank", "Pressure correction could not be solved", error=str(exc))
        norm = np.linalg.norm(residual)
        for k in range(budget["maximum_backtracks"]):
            trial = p.copy()
            trial[free] += step * (0.5**k)
            _, candidate = evaluate(trial)
            if (
                np.linalg.norm(candidate)
                < (1 - budget["line_search_sufficient_decrease"] * 0.5**k) * norm
            ):
                p = trial
                backtracks += k
                break
        else:
            _reject(
                "line_search", "Pressure correction cannot reduce mass residual without clipping"
            )

    for i, node in enumerate(value["nodes"]):
        if p[i] <= node.get("min_pressure_pa", 0) or p[i] >= node.get("max_pressure_pa", np.inf):
            _reject(
                "pressure_domain",
                "Solved pressure lies outside the declared open domain",
                node=node["id"],
                pressure_pa=float(p[i]),
            )
    pressure_residual = a.T @ p + gain - linear * flow - quadratic * flow * np.abs(flow)
    actual = a @ flow
    external = actual.copy()
    external[free] = injections
    rho, gravity = value["density_kg_m3"], value["gravity_m_s2"]
    heights = np.array([n["elevation_m"] for n in value["nodes"]])
    heads = np.array([e["pressure_gain_pa"] for e in value["edges"]])
    boundary_power = float(p @ external / rho)
    gain_power = float(heads @ flow / rho)
    gravity_power = float(gravity * heights @ (a @ flow))
    dissipation = float((linear * flow**2 + quadratic * np.abs(flow) ** 3).sum() / rho)
    metrics = dict(
        maximum_global_mass_residual_kg_s=abs(float(external.sum())),
        maximum_mass_residual_kg_s=float(np.max(np.abs(residual))) if free else 0.0,
        maximum_pressure_residual_pa=float(np.max(np.abs(pressure_residual))),
        maximum_power_residual_w=abs(boundary_power + gain_power + gravity_power - dissipation),
    )
    if any(not np.isfinite(v) or v > budget[k] for k, v in metrics.items()):
        _reject(
            "dimensional_residual",
            "Original dimensional balances exceed their budgets",
            residuals=metrics,
        )
    return dict(
        schema_version=1,
        accepted=True,
        model="steady-incompressible-resistor-graph-v1",
        input=value,
        pressures_pa={n["id"]: float(p[i]) for i, n in enumerate(value["nodes"])},
        flows_kg_s={e["id"]: float(flow[i]) for i, e in enumerate(value["edges"])},
        external_injections_kg_s={
            n["id"]: float(external[i]) for i, n in enumerate(value["nodes"])
        },
        residuals=metrics,
        powers_w=dict(
            pressure_boundary_and_injection=boundary_power,
            prescribed_gain=gain_power,
            gravity=gravity_power,
            dissipation=dissipation,
        ),
        iterations=iterations,
        backtracks=backtracks,
        maximum_scaled_condition=max(conditions, default=1.0),
        initial_pressures_pa=initial.tolist(),
        warmup_steps=0,
        commands_changed=False,
        uniqueness="Strictly monotone edge laws with fixed pressure gauge; finite solution is unique within this graph formulation",
        budget_sha256=sha(BUDGET),
        driver_sha256=sha(__file__),
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(deepcopy(spec), indent=2) + "\n")
    try:
        report = solve_hydraulic_graph(spec)
        name, code = "solution.json", 0
    except HydraulicSolveError as error:
        report, name, code = error.report, "diagnostics.json", 2
    except NetworkError as error:
        report, name, code = (
            dict(accepted=False, criterion="invalid_input", message=str(error)),
            "diagnostics.json",
            2,
        )
    (args.output / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = dict(
        schema_version=1,
        driver_sha256=sha(__file__),
        budget_sha256=sha(BUDGET),
        files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        exit_code=code,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
