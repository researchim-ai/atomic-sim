"""Steady two-way temperature/resistance feedback on a conservative graph.

rho and cp remain constant. L(T)=L_ref*exp(beta*(mean_endpoint_T-T_ref)) is
a declared user model, not an empirical water viscosity correlation. Existing
hydraulic/thermal budgets are retained. No global uniqueness is asserted.
"""

import argparse
import json
import math
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy

from core.network import NetworkError, number
from scripts.solve_hydraulic_graph import HydraulicSolveError, _keys, compile_graph, sha
from scripts.solve_thermal_graph import ThermalSolveError, solve_thermal_graph
from sdk.experiment import _source_fingerprint

BUDGET = Path("validation/ai05_resistance_feedback_budget_v2.json")
DRIVERS = (
    "scripts/solve_resistance_feedback.py",
    "scripts/solve_thermal_graph.py",
    "scripts/solve_hydraulic_graph.py",
)


def signature():
    return dict(
        source_sha256=_source_fingerprint(),
        drivers={p: sha(p) for p in DRIVERS},
        budgets={
            str(p): sha(p)
            for p in (
                BUDGET,
                Path("validation/ai05_thermal_graph_budget_v1.json"),
                Path("validation/ai05_hydraulic_graph_budget_v1.json"),
            )
        },
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )


class ResistanceFeedbackError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


def prepare(spec):
    _keys(
        spec,
        ("format", "schema_version", "steady_input", "resistance_laws", "initial_temperatures_k"),
    )
    if (
        spec["format"] != "atomic-resistance-feedback-steady"
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise NetworkError("Expected atomic-resistance-feedback-steady schema 1")
    base = deepcopy(spec["steady_input"])
    _keys(base, ("format", "schema_version", "hydraulic", "thermal"))
    graph, *_ = compile_graph(base["hydraulic"])
    names = [n["id"] for n in graph["nodes"]]
    edges = [e["id"] for e in graph["edges"]]
    if not isinstance(spec["resistance_laws"], dict) or set(spec["resistance_laws"]) != set(edges):
        raise NetworkError("Explicit law required for every hydraulic edge")
    if not isinstance(spec["initial_temperatures_k"], dict) or set(
        spec["initial_temperatures_k"]
    ) != set(names):
        raise NetworkError("Explicit temperature guess required for every node")
    if not isinstance(base["thermal"], dict):
        raise NetworkError("Explicit thermal model required")
    try:
        lower = number(base["thermal"]["min_temperature_k"], "min_temperature_k", positive=True)
        upper = number(base["thermal"]["max_temperature_k"], "max_temperature_k", positive=True)
    except KeyError as error:
        raise NetworkError("Explicit thermal domain required") from error
    if lower >= upper:
        raise NetworkError("Empty thermal domain")
    temperature = {
        name: number(spec["initial_temperatures_k"][name], "initial_temperature_k", positive=True)
        for name in names
    }
    if any(not lower < t < upper for t in temperature.values()):
        raise ResistanceFeedbackError(
            "initial_temperature_domain", "Temperature guess exceeds declared open domain"
        )
    laws = {}
    for edge in edges:
        law = spec["resistance_laws"][edge]
        _keys(
            law,
            (
                "kind",
                "reference_temperature_k",
                "temperature_coefficient_per_k",
                "minimum_multiplier",
                "maximum_multiplier",
            ),
        )
        if law["kind"] != "exponential_endpoint_mean_v1":
            raise NetworkError("Unsupported resistance law; no implicit correlation is selected")
        value = {
            k: number(law[k], k, positive=k != "temperature_coefficient_per_k")
            for k in sorted(law)
            if k != "kind"
        }
        if value["minimum_multiplier"] >= value["maximum_multiplier"]:
            raise NetworkError("Empty resistance multiplier interval")
        laws[edge] = dict(kind=law["kind"], **value)
    return base, graph, temperature, laws


def effective_input(base, graph, temperatures, laws):
    value = deepcopy(base)
    multipliers = {}
    for edge, reference in zip(value["hydraulic"]["edges"], graph["edges"]):
        law = laws[reference["id"]]
        mean = temperatures[reference["from"]] * 0.5 + temperatures[reference["to"]] * 0.5
        exponent = law["temperature_coefficient_per_k"] * (mean - law["reference_temperature_k"])
        try:
            factor = math.exp(exponent)
        except OverflowError as error:
            raise ResistanceFeedbackError(
                "resistance_law_domain",
                "Resistance exponent exceeds finite arithmetic",
                edge=reference["id"],
            ) from error
        if (
            not math.isfinite(factor)
            or not law["minimum_multiplier"] <= factor <= law["maximum_multiplier"]
        ):
            raise ResistanceFeedbackError(
                "resistance_law_domain",
                "Resistance law is outside its declared multiplier interval; no clipping or extrapolation",
                edge=reference["id"],
                temperature_k=mean,
            )
        edge["linear_pa_s_kg"] = reference["linear_pa_s_kg"] * factor
        if not math.isfinite(edge["linear_pa_s_kg"]) or edge["linear_pa_s_kg"] <= 0:
            raise ResistanceFeedbackError(
                "resistance_law_domain",
                "Effective resistance left finite positive arithmetic",
                edge=reference["id"],
            )
        multipliers[reference["id"]] = factor
    return value, multipliers


def solve_resistance_feedback(spec):
    base, graph, temperature, laws = prepare(spec)
    budget = json.loads(BUDGET.read_text())
    trace = []
    captured = signature()
    byname = {n["id"]: n for n in graph["nodes"]}
    budget_hash = sha(BUDGET)
    for iteration in range(1, budget["maximum_iterations"] + 1):
        effective, multipliers = effective_input(base, graph, temperature, laws)
        solution = solve_thermal_graph(effective)
        actual = solution["temperatures_k"]
        target, target_factors = effective_input(base, graph, actual, laws)
        error = max(abs(actual[n] - temperature[n]) for n in actual)
        relative = max(abs(target_factors[e] / multipliers[e] - 1) for e in multipliers)
        residual = 0.0
        node_energy = deepcopy(solution["node_energy_residuals_w"])
        edge_energy = 0.0
        effective_edges = {e["id"]: e for e in effective["hydraulic"]["edges"]}
        streams_admitted = True
        for edge in target["hydraulic"]["edges"]:
            flow = solution["hydraulic"]["flows_kg_s"][edge["id"]]
            pressure = solution["hydraulic"]["pressures_pa"]
            drop = (
                pressure[edge["from"]]
                - pressure[edge["to"]]
                + edge["pressure_gain_pa"]
                - graph["density_kg_m3"]
                * graph["gravity_m_s2"]
                * (byname[edge["to"]]["elevation_m"] - byname[edge["from"]]["elevation_m"])
            )
            momentum = (
                drop
                - edge["linear_pa_s_kg"] * flow
                - edge["quadratic_pa_s2_kg2"] * flow * abs(flow)
            )
            residual = max(residual, abs(momentum))
            rho = graph["density_kg_m3"]
            edge_energy = max(edge_energy, abs(momentum * flow / rho))
            donor, receiver = (
                (edge["from"], edge["to"]) if flow >= 0 else (edge["to"], edge["from"])
            )
            delta_loss = (
                (edge["linear_pa_s_kg"] - effective_edges[edge["id"]]["linear_pa_s_kg"])
                * flow
                * flow
                / rho
            )
            node_energy[receiver] += delta_loss
            outlet = actual[donor] + (
                edge["linear_pa_s_kg"] * abs(flow) + edge["quadratic_pa_s2_kg2"] * flow * flow
            ) / (rho * solution["thermal"]["heat_capacity_j_kg_k"])
            streams_admitted = (
                streams_admitted
                and solution["thermal"]["min_temperature_k"]
                < outlet
                < solution["thermal"]["max_temperature_k"]
            )
        metrics = dict(
            temperature_fixed_point_error_k=error,
            relative_resistance_error=relative,
            constitutive_pressure_residual_pa=residual,
            constitutive_node_energy_residual_w=max(abs(v) for v in node_energy.values()),
            constitutive_edge_energy_residual_w=edge_energy,
        )
        if any(not math.isfinite(v) for v in metrics.values()):
            raise ResistanceFeedbackError(
                "coupling_arithmetic", "Feedback residual left finite arithmetic; no point accepted"
            )
        trace.append(dict(iteration=iteration, **metrics))
        if (
            error <= budget["maximum_temperature_fixed_point_error_k"]
            and relative <= budget["maximum_relative_resistance_error"]
            and residual <= budget["maximum_constitutive_pressure_residual_pa"]
            and metrics["constitutive_node_energy_residual_w"]
            <= budget["maximum_constitutive_node_energy_residual_w"]
            and edge_energy <= budget["maximum_constitutive_edge_energy_residual_w"]
        ):
            if not streams_admitted:
                raise ResistanceFeedbackError(
                    "coupled_stream_temperature_domain",
                    "Outlet temperature at the returned-temperature resistance law exceeds its declared domain",
                )
            if captured != signature() or budget_hash != sha(BUDGET):
                raise ResistanceFeedbackError(
                    "source_drift", "Feedback sources changed before verdict"
                )
            return dict(
                schema_version=1,
                accepted=True,
                model="steady-temperature-resistance-feedback-v1",
                solution=solution,
                resistance_laws=laws,
                multipliers=multipliers,
                target_multipliers=target_factors,
                metrics=metrics,
                iterations=iteration,
                trace=trace,
                commands_changed=False,
                warmup_steps=0,
                global_uniqueness_proven=False,
                signature=captured,
                input=deepcopy(spec),
                budget_sha256=budget_hash,
                validated=False,
                plant_referenced=False,
            )
        weight = budget["relaxation"]
        temperature = {n: temperature[n] + weight * (actual[n] - temperature[n]) for n in actual}
    raise ResistanceFeedbackError(
        "coupling_iteration_limit",
        "Temperature/resistance fixed point did not meet dimensional criteria; no result accepted",
        iterations=budget["maximum_iterations"],
        last_metrics=trace[-1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_text(json.dumps(spec, indent=2) + "\n")
    try:
        report = solve_resistance_feedback(spec)
        filename, code = "solution.json", 0
    except (ResistanceFeedbackError, HydraulicSolveError, ThermalSolveError) as error:
        report, filename, code = error.report, "diagnostics.json", 2
    except NetworkError as error:
        report, filename, code = (
            dict(accepted=False, criterion="invalid_input", message=str(error)),
            "diagnostics.json",
            2,
        )
    (args.output / filename).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = dict(
        signature=signature(),
        budget_sha256=sha(BUDGET),
        files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        exit_code=code,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
