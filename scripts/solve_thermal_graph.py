"""Conservative enthalpy mixing on the qualified constant-density hydraulic graph.

Constant cp/rho: hydraulic state is solved first, then the exact steady linear
enthalpy equations. No temperature feedback, thermal storage or implicit thermal
gauge. Endpoint flow convention is positive INTO each component owner.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, PortSpec, number
from scripts.solve_hydraulic_graph import (
    HydraulicSolveError,
    _keys,
    sha,
    solve_hydraulic_graph,
)

BUDGET = Path("validation/ai05_thermal_graph_budget_v1.json")
DRIVERS = ("scripts/solve_thermal_graph.py", "scripts/solve_hydraulic_graph.py")


class ThermalSolveError(NetworkError):
    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(accepted=False, criterion=criterion, message=message, details=details)


def reject(criterion, message, **details):
    raise ThermalSolveError(criterion, message, **details)


def prepare(spec):
    budget = json.loads(BUDGET.read_text())
    _keys(spec, ("format", "schema_version", "hydraulic", "thermal"))
    if (
        spec["format"] != "atomic-thermal-hydraulic-steady"
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise NetworkError("Expected atomic-thermal-hydraulic-steady schema 1")
    thermal = spec["thermal"]
    _keys(
        thermal,
        (
            "medium",
            "heat_capacity_j_kg_k",
            "enthalpy_reference_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
            "nodes",
        ),
    )
    if not isinstance(thermal["medium"], str) or not thermal["medium"].strip():
        raise NetworkError("An explicit single-fluid medium identifier is required")
    values = {
        key: number(thermal[key], key, positive=True)
        for key in (
            "heat_capacity_j_kg_k",
            "enthalpy_reference_temperature_k",
            "min_temperature_k",
            "max_temperature_k",
        )
    }
    lower, upper = values["min_temperature_k"], values["max_temperature_k"]
    if lower >= upper:
        raise NetworkError("Empty thermal temperature domain")
    if not isinstance(thermal["nodes"], dict) or not isinstance(spec["hydraulic"], dict):
        raise NetworkError("Explicit hydraulic graph and thermal node map required")
    hydraulic = solve_hydraulic_graph(spec["hydraulic"])
    names = [n["id"] for n in hydraulic["input"]["nodes"]]
    if set(thermal["nodes"]) != set(names):
        raise NetworkError("Thermal node names must exactly match hydraulic nodes")
    normalized = {}
    for name in names:
        node = thermal["nodes"][name]
        _keys(
            node,
            ("heat_power_w", "ambient_ua_w_k", "ambient_temperature_k"),
            ("supply_temperature_k",),
        )
        value = {
            key: number(node[key], key, positive=key.endswith("temperature_k"))
            for key in sorted(node)
        }
        if value["ambient_ua_w_k"] < 0:
            raise NetworkError("Ambient conductance must be nonnegative")
        source = hydraulic["external_injections_kg_s"][name]
        if 0 < abs(source) < budget["minimum_resolved_external_flow_kg_s"]:
            reject(
                "unresolved_external_flow",
                "External exchange is below the declared resolution; it cannot anchor a thermal state",
                node=name,
                injection_kg_s=source,
            )
        if source > 0 and "supply_temperature_k" not in value:
            reject(
                "missing_supply_temperature",
                "Positive external injection requires its supplied temperature",
                node=name,
                injection_kg_s=source,
            )
        if "supply_temperature_k" in value and not lower < value["supply_temperature_k"] < upper:
            reject(
                "supply_temperature_domain",
                "Supply temperature exceeds the open thermal domain",
                node=name,
            )
        normalized[name] = value
    values.update(medium=thermal["medium"], nodes=normalized)
    return hydraulic, values


def solve_thermal_graph(spec):
    hydraulic, thermal = prepare(spec)
    budget = json.loads(BUDGET.read_text())
    nodes, edges = hydraulic["input"]["nodes"], hydraulic["input"]["edges"]
    names = {n["id"]: i for i, n in enumerate(nodes)}
    cp, ref = thermal["heat_capacity_j_kg_k"], thermal["enthalpy_reference_temperature_k"]
    rho, gravity = hydraulic["input"]["density_kg_m3"], hydraulic["input"]["gravity_m_s2"]
    matrix, rhs = np.zeros((len(nodes), len(nodes))), np.zeros(len(nodes))
    inflow, outflow, deposited = np.zeros(len(nodes)), np.zeros(len(nodes)), np.zeros(len(nodes))
    for edge in edges:
        m = hydraulic["flows_kg_s"][edge["id"]]
        donor, receiver = (
            (names[edge["from"]], names[edge["to"]])
            if m >= 0
            else (names[edge["to"]], names[edge["from"]])
        )
        rate = abs(m)
        matrix[donor, donor] += rate * cp
        matrix[receiver, donor] -= rate * cp
        outflow[donor] += rate
        inflow[receiver] += rate
        loss = (edge["linear_pa_s_kg"] * m * m + edge["quadratic_pa_s2_kg2"] * abs(m) ** 3) / rho
        deposited[receiver] += loss
    for name, i in names.items():
        node = thermal["nodes"][name]
        source = hydraulic["external_injections_kg_s"][name]
        matrix[i, i] += cp * max(-source, 0) + node["ambient_ua_w_k"]
        rhs[i] = (
            node["heat_power_w"]
            + node["ambient_ua_w_k"] * (node["ambient_temperature_k"] - ref)
            + deposited[i]
        )
        if source > 0:
            rhs[i] += source * cp * (node["supply_temperature_k"] - ref)
        # Use total specific enthalpy h=cp*(T-ref)+p/rho, including the tiny
        # hydraulic mass residual rather than silently erasing pressure work.
        rhs[i] += hydraulic["pressures_pa"][name] / rho * (inflow[i] - outflow[i] + source)
    scale = np.max(np.abs(matrix), axis=1)
    if np.any(scale <= 0) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        reject("thermal_rank", "Unanchored zero-flow thermal node or nonfinite thermal equations")
    scaled = matrix / scale[:, None]
    condition = float(np.linalg.cond(scaled))
    if not np.isfinite(condition) or condition > budget["maximum_scaled_condition"]:
        reject(
            "thermal_rank",
            "Thermal mixing matrix is singular or numerically unresolved; no temperature gauge is invented",
            condition=condition,
        )
    try:
        temperature = ref + np.linalg.solve(scaled, rhs / scale)
    except np.linalg.LinAlgError as error:
        reject("thermal_rank", "Thermal mixing system could not be solved", error=str(error))
    if not np.all(np.isfinite(temperature)):
        reject("nonfinite", "Thermal solution left finite arithmetic")
    for name, i in names.items():
        if not thermal["min_temperature_k"] < temperature[i] < thermal["max_temperature_k"]:
            reject(
                "temperature_domain",
                "Solved mixing temperature exceeds its open domain; no clipping",
                node=name,
                temperature_k=float(temperature[i]),
            )
    # PortSpec checks medium, units and the existing enthalpy convention. These
    # are steady endpoint records, not dynamic core/network component adapters.
    PortSpec(
        "fluid",
        "hydraulic",
        medium=thermal["medium"],
        energy_convention="specific_enthalpy_transport",
    )
    node_balance, ports, edge_errors = np.zeros(len(nodes)), [], []
    for edge in edges:
        left, right = names[edge["from"]], names[edge["to"]]
        m = hydraulic["flows_kg_s"][edge["id"]]
        donor = left if m >= 0 else right
        outlet = temperature[donor] + (
            edge["linear_pa_s_kg"] * abs(m) + edge["quadratic_pa_s2_kg2"] * m * m
        ) / (rho * cp)
        if not thermal["min_temperature_k"] < outlet < thermal["max_temperature_k"]:
            reject(
                "edge_temperature_domain",
                "Stream temperature before node mixing exceeds its domain",
                edge=edge["id"],
                temperature_k=float(outlet),
            )
        sensible = cp * (temperature[donor] - ref)
        loss = (edge["linear_pa_s_kg"] * m * m + edge["quadratic_pa_s2_kg2"] * abs(m) ** 3) / rho
        left_power = m * (sensible + hydraulic["pressures_pa"][edge["from"]] / rho) - (
            loss if m < 0 else 0
        )
        right_power = -m * (sensible + hydraulic["pressures_pa"][edge["to"]] / rho) - (
            loss if m >= 0 else 0
        )
        gain_power = m * edge["pressure_gain_pa"] / rho
        gravity_power = gravity * m * (nodes[left]["elevation_m"] - nodes[right]["elevation_m"])
        edge_errors.append(abs(left_power + right_power + gain_power + gravity_power))
        for endpoint, i, mass, power in [
            ("from", left, m, left_power),
            ("to", right, -m, right_power),
        ]:
            node_balance[i] -= power
            ports.append(
                dict(
                    edge=edge["id"],
                    endpoint=endpoint,
                    node=nodes[i]["id"],
                    medium=thermal["medium"],
                    energy_convention="specific_enthalpy_transport",
                    pressure_pa=hydraulic["pressures_pa"][nodes[i]["id"]],
                    stream_temperature_k=(
                        float(temperature[donor] if i == donor else outlet) if m != 0 else None
                    ),
                    edge_inward_mass_kg_s=float(mass),
                    node_inward_mass_kg_s=float(-mass),
                    edge_inward_energy_w=float(power),
                    node_inward_energy_w=float(-power),
                )
            )
    external_power, heat, ambient = 0.0, 0.0, 0.0
    for name, i in names.items():
        node = thermal["nodes"][name]
        source = hydraulic["external_injections_kg_s"][name]
        supply = node["supply_temperature_k"] if source > 0 else temperature[i]
        power = source * (cp * (supply - ref) + hydraulic["pressures_pa"][name] / rho)
        removal = node["ambient_ua_w_k"] * (temperature[i] - node["ambient_temperature_k"])
        node_balance[i] += power + node["heat_power_w"] - removal
        external_power += power
        heat += node["heat_power_w"]
        ambient += removal
    metrics = dict(
        maximum_node_energy_residual_w=float(np.max(np.abs(node_balance))),
        maximum_edge_energy_residual_w=max(edge_errors),
        maximum_global_energy_residual_w=abs(
            external_power
            + heat
            - ambient
            + hydraulic["powers_w"]["prescribed_gain"]
            + hydraulic["powers_w"]["gravity"]
        ),
    )
    if any(not np.isfinite(v) or v > budget[k] for k, v in metrics.items()):
        reject(
            "energy_residual",
            "Conservative dimensional energy residual exceeds budget",
            residuals=metrics,
        )
    return dict(
        schema_version=1,
        accepted=True,
        model="steady-incompressible-enthalpy-graph-v1",
        hydraulic=hydraulic,
        thermal=thermal,
        temperatures_k={name: float(temperature[i]) for name, i in names.items()},
        node_energy_residuals_w={name: float(node_balance[i]) for name, i in names.items()},
        ports=ports,
        units=dict(pressure_pa="Pa", edge_inward_mass_kg_s="kg/s", edge_inward_energy_w="W"),
        residuals=metrics,
        powers_w=dict(
            external_enthalpy=external_power,
            external_heat=heat,
            ambient_removal=ambient,
            hydraulic_gain=hydraulic["powers_w"]["prescribed_gain"],
            gravity=hydraulic["powers_w"]["gravity"],
            dissipation_to_fluid=float(deposited.sum()),
        ),
        maximum_scaled_condition=condition,
        warmup_steps=0,
        commands_changed=False,
        driver_sha256={p: sha(p) for p in DRIVERS},
        budget_sha256=sha(BUDGET),
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
        report = solve_thermal_graph(spec)
        name, code = "solution.json", 0
    except (ThermalSolveError, HydraulicSolveError) as error:
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
        driver_sha256={p: sha(p) for p in DRIVERS},
        budget_sha256=sha(BUDGET),
        files={p.name: sha(p) for p in sorted(args.output.iterdir())},
        exit_code=code,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
