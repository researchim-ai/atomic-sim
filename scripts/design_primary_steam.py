"""Design a consistent primary/steam operating point without a warm-up transient.

Steam p/x/rotor speed and signed primary flow are targets. Feed, electrical
load, sink temperature, primary motor/heater and all wall/fluid temperatures
are chosen explicitly. Arbitrary fixed boundary commands are not solved.
The resulting explicit schema-1 input is consumed by the existing SDK/CLI.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from physics.power.primary_steam import COUPLING_FIELDS, PASSPORT, controls
from physics.power.steam_cycle import balanced_steam_cycle
from scripts.verify_turbine import require_finite, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.primary_steam_experiment import prepare
from sdk.pump_experiment import prepare as prepare_pump

FIELDS = {
    "schema_version",
    "engine",
    "name",
    "primary",
    "steam_model",
    "coupling",
    "controls",
    "duration_s",
    "step_s",
    "solver",
    "timeline",
}


def design_steady_primary_steam(spec, *, primary_mass_flow_kg_s, valve_opening=None):
    """Return explicit given-state input and dimensional design evidence.

    Initial primary temperature guesses are replaced, never integrated toward
    equilibrium. Existing timelines are rejected because chosen boundaries
    would otherwise silently change the meaning of their absolute commands.
    """
    if (
        not isinstance(spec, dict)
        or set(spec) != FIELDS
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["engine"] != PASSPORT.model_id
    ):
        raise NetworkError("Exact primary/steam experiment schema 1 required for design")
    if not isinstance(spec["timeline"], list) or spec["timeline"]:
        raise NetworkError("Steady design requires an empty command timeline")
    require_finite(spec)
    original = controls(spec["controls"])
    flow = number(primary_mass_flow_kg_s, "primary_mass_flow_kg_s")
    opening = (
        original["steam_cycle"]["valve_opening"]
        if valve_opening is None
        else number(valve_opening, "valve_opening")
    )
    coupling = spec["coupling"]
    if not isinstance(coupling, dict) or set(coupling) != COUPLING_FIELDS:
        raise NetworkError("Exact primary/steam coupling required for design")
    link = number(coupling["ua_w_k"], "ua_w_k", positive=True)
    result = deepcopy(spec)
    steam = balanced_steam_cycle(result["steam_model"], valve_opening=opening)
    steam_controls = steam.boundary_controls
    target_heat = steam_controls["heater_power_w"]
    steam_controls["heater_power_w"] = 0
    result["steam_model"] = steam.configuration
    primary = result["primary"]
    if not isinstance(primary, dict) or set(primary) != {"loop", "pump", "initial_omega_rad_s"}:
        raise NetworkError("Exact primary loop/pump parameters required for design")
    topology = primary["loop"]["model"]["topology"]
    nodes = topology["nodes"]
    if (
        not isinstance(nodes, list)
        or not 1 <= len(nodes) <= 128
        or any(not isinstance(n, dict) or "id" not in n for n in nodes)
    ):
        raise NetworkError("Bounded named primary topology required for design")
    # Validate topology/material/machine parameters with a detached zero-power
    # probe. The requested operating point need not have a valid warm-start guess.
    probe = deepcopy(primary)
    probe["loop"]["initialization"] = dict(
        mode="given",
        mass_flow_kg_s=0,
        fluid_temperature_k={node["id"]: 500 for node in nodes},
        wall_temperature_k={node["id"]: 500 for node in nodes},
    )
    probe["initial_omega_rad_s"] = 0
    _, machine = prepare_pump(
        dict(
            schema_version=1,
            engine="displacement-pump-circulation-loop",
            name="steady_design_parameter_probe",
            **probe,
            motor_torque_nm=0,
            duration_s=1,
            step_s=1,
            solver=dict(method="Radau", rtol=1e-8, max_step_s=0.1, atol_scale=0.1),
            timeline=[],
        )
    )
    loop = machine._loop
    pump = machine.pump
    n = loop.n
    for key in ("steam_generator_segment", "heater_segment"):
        if not isinstance(coupling[key], str) or coupling[key] not in loop.names:
            raise NetworkError("Unknown design heat-exchange/heater segment")
    exchange = loop.names.index(coupling["steam_generator_segment"])
    heater = loop.names.index(coupling["heater_segment"])
    head = loop.resistance_total * flow * abs(flow)
    omega = (flow / loop.rho + pump["leakage_m3_s_pa"] * head) / pump["displacement_m3_rad"]
    torque = pump["displacement_m3_rad"] * head + pump["drag_nm_s_rad"] * omega
    leakage = pump["leakage_m3_s_pa"] * head**2
    anchor = steam.configuration["hot_wall"]["initial_temperature_k"] + target_heat / link
    matrix = np.zeros((2 * n + 1, 2 * n + 1))
    constants = np.zeros(2 * n + 1)
    advective = loop.cp * abs(flow)
    for i in range(n):
        donor = (i - 1 if flow >= 0 else i + 1) % n
        matrix[i, i] -= advective + loop.ua[i]
        matrix[i, donor] += advective
        matrix[i, n + i] += loop.ua[i]
        constants[i] = loop.resistance[i] * abs(flow) ** 3 / loop.rho
        if i == exchange:
            matrix[i, i] -= link
            constants[i] -= target_heat
        matrix[n + i, i] += loop.ua[i]
        matrix[n + i, n + i] -= loop.ua[i] + loop.ambient_ua[i]
        constants[n + i] = loop.heater[i] + loop.ambient_ua[i] * (
            loop.ambient_temperature_k - anchor
        )
        if i == machine._index:
            constants[n + i] += leakage
        if i == heater:
            matrix[n + i, 2 * n] = 1
    matrix[-1, exchange] = 1
    columns = np.r_[np.ones(2 * n), max(abs(target_heat), 1.0)]
    rows = np.maximum(np.max(abs(matrix * columns), axis=1), 1.0)
    scaled = matrix * columns / rows[:, None]
    condition = float(np.linalg.cond(scaled))
    if not np.isfinite(condition) or condition > 1e12:
        raise NetworkError("Joint thermal steady system is unanchored or ill-conditioned")
    try:
        solution = np.linalg.solve(scaled, -constants / rows) * columns
    except np.linalg.LinAlgError as exc:
        raise NetworkError("Joint thermal steady solve failed") from exc
    if not np.isfinite(solution).all() or solution[-1] < 0:
        raise NetworkError("Requested steady point requires an unmodelled primary cooler")
    residual = matrix @ solution + constants
    if max(abs(residual[:-1])) > 1e-6 or abs(residual[-1]) > 1e-10:
        raise NetworkError("Joint dimensional thermal residual exceeds design budget")
    result["primary"]["initial_omega_rad_s"] = float(omega)
    result["primary"]["loop"]["initialization"] = dict(
        mode="given",
        mass_flow_kg_s=flow,
        fluid_temperature_k=dict(zip(loop.names, map(float, anchor + solution[:n]))),
        wall_temperature_k=dict(zip(loop.names, map(float, anchor + solution[n : 2 * n]))),
    )
    result["controls"] = dict(
        primary_motor_torque_nm=float(torque),
        primary_wall_heater_power_w=float(solution[-1]),
        steam_cycle=steam_controls,
    )
    normalized, model = prepare(result)
    rates = model.derivatives(0, model._state)
    offset = model._offset
    cfg = model._steam.configuration
    stock_powers = [
        *(loop.capacity * rates[1 : 1 + n]),
        *(loop.wall_capacity * rates[1 + n : 1 + 2 * n]),
        rates[offset + 1],
        rates[offset + 3],
        cfg["hot_wall"]["heat_capacity_j_k"] * rates[offset + 4],
        cfg["cold_wall"]["heat_capacity_j_k"] * rates[offset + 5],
        rates[offset + 6],
    ]
    force = float(loop.inertia_total * rates[0])
    shaft = float(pump["inertia_kg_m2"] * rates[machine._size])
    mass = max(abs(rates[offset]), abs(rates[offset + 2]))
    if (
        max(abs(float(v)) for v in stock_powers) > 1e-5
        or abs(force) > 1e-6
        or abs(shaft) > 1e-8
        or mass > 1e-9
    ):
        raise NetworkError("Reconstructed coupled operating point fails steady residual budgets")
    report = dict(
        schema_version=1,
        mode="steady_boundary_design",
        warmup_steps=0,
        targets=dict(
            primary_mass_flow_kg_s=flow,
            valve_opening=opening,
            steam_pressures_pa={
                name: cfg[name]["initial_pressure_pa"] for name in ("steam_generator", "condenser")
            },
            steam_qualities={
                name: cfg[name]["initial_quality"] for name in ("steam_generator", "condenser")
            },
            rotor_speed_rad_s=cfg["turbine"]["initial_omega_rad_s"],
        ),
        chosen_commands=deepcopy(normalized["controls"]),
        selected_heat_transfer_w=float(target_heat),
        thermal_scaled_condition=condition,
        thermal_linear_residual_w=float(max(abs(residual[:-1]))),
        temperature_constraint_residual_k=float(abs(residual[-1])),
        primary_momentum_residual_pa=force,
        primary_shaft_residual_nm=shaft,
        maximum_stock_power_residual_w=float(max(abs(float(v)) for v in stock_powers)),
        maximum_inventory_mass_rate_kg_s=float(mass),
        domain_margins=model.observation()["domain_margins"],
        validated=False,
        plant_referenced=False,
    )
    return normalized, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--primary-flow-kg-s", type=float, required=True)
    parser.add_argument("--valve-opening", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.scenario.read_text())
    normalized, report = design_steady_primary_steam(
        request, primary_mass_flow_kg_s=args.primary_flow_kg_s, valve_opening=args.valve_opening
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "request.json", request)
    write(args.output / "input.json", normalized)
    write(args.output / "design.json", report)
    write(
        args.output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=_source_fingerprint(),
            designer_sha256=sha(__file__),
            artifacts={
                name: sha(args.output / name)
                for name in ("request.json", "input.json", "design.json")
            },
        ),
    )
    print(
        json.dumps(
            dict(
                designed=True,
                warmup_steps=0,
                maximum_stock_power_residual_w=report["maximum_stock_power_residual_w"],
            )
        )
    )


if __name__ == "__main__":
    main()
