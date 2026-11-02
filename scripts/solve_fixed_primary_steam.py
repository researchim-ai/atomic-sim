"""Solve a primary/steam steady point with fixed boundary commands.

Separate vapor/liquid outlets make steady powers independent of bulk quality.
Total steam mass and boiler quality are therefore explicit inventory anchors.
One pressure is found from condenser balance; the hot balance independently
tests command compatibility. No controls or physical states are clipped.
"""

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from core.network import NetworkError, number
from core.status import DomainError
from physics.materials.iapws import saturation_properties
from physics.power.primary_steam import PASSPORT, controls
from physics.power.steam_turbine_train import expansion
from scripts.design_primary_steam import FIELDS
from scripts.primary_steam_references import system
from scripts.verify_turbine import require_finite, sha
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.primary_steam_experiment import prepare

BUDGET = Path("validation/ai05_fixed_boundary_steady_budget_v3.json")
DRIVERS = (
    "scripts/solve_fixed_primary_steam.py",
    "scripts/design_primary_steam.py",
    "scripts/primary_steam_references.py",
    "scripts/verify_turbine.py",
    "scripts/turbine_references.py",
)


class SteadySolveError(NetworkError):
    """A validated model cannot satisfy the requested admitted steady constraints."""

    def __init__(self, criterion, message, **details):
        super().__init__(message)
        self.report = dict(
            schema_version=1,
            accepted=False,
            criterion=criterion,
            message=message,
            details=details,
            warmup_steps=0,
            commands_changed=False,
            validated=False,
            plant_referenced=False,
        )


def _reject(criterion, message, **details):
    raise SteadySolveError(criterion, message, **details)


def inventory_anchors(spec):
    """Explicit helper; callers decide whether initial stocks are their targets."""
    total = 0.0
    for name in ("steam_generator", "condenser"):
        vessel = spec["steam_model"][name]
        pressure = number(vessel["initial_pressure_pa"], "initial_pressure_pa", positive=True)
        volume = number(vessel["volume_m3"], "volume_m3", positive=True)
        x = number(vessel["initial_quality"], "initial_quality", positive=True)
        if x >= 1:
            raise NetworkError("Two-phase inventory anchors require quality in (0,1)")
        liquid, vapor = saturation_properties(pressure)
        total += volume / ((1 - x) * liquid.v_m3_kg + x * vapor.v_m3_kg)
    return dict(
        total_steam_mass_kg=number(total, "total_steam_mass_kg", positive=True),
        steam_generator_quality=spec["steam_model"]["steam_generator"]["initial_quality"],
    )


def _primary(model, budget):
    machine, link = model._primary, model._coupling["ua_w_k"]
    loop, pump, c = machine._loop, machine.pump, model.boundary_controls
    displacement, drag, leak = (
        pump[k] for k in ("displacement_m3_rad", "drag_nm_s_rad", "leakage_m3_s_pa")
    )
    quadratic = (displacement + drag * leak / displacement) * loop.resistance_total
    linear = drag / (displacement * loop.rho)
    torque = c["primary_motor_torque_nm"]
    if quadratic == linear == 0:
        _reject("hydraulic_rank", "Fixed motor torque does not determine a unique primary flow")
    denominator = linear + math.sqrt(linear**2 + 4 * quadratic * abs(torque))
    flow = 0.0 if torque == 0 else math.copysign(2 * abs(torque) / denominator, torque)
    head = loop.resistance_total * flow * abs(flow)
    omega = (flow / loop.rho + leak * head) / displacement
    # Validate actual pressure faces at the solved hydraulic point.
    probe = machine._state.copy()
    probe[0], probe[machine._size] = flow, omega
    machine.derivatives(0, probe)
    n, index = loop.n, model._exchange_index
    a, b = np.zeros((2 * n, 2 * n)), np.zeros(2 * n)
    anchor = model._steam.configuration["hot_wall"]["initial_temperature_k"]
    for i in range(n):
        donor = (i - 1 if flow >= 0 else i + 1) % n
        advective = loop.cp * abs(flow)
        a[i, i] -= advective + loop.ua[i]
        a[i, donor] += advective
        a[i, n + i] += loop.ua[i]
        b[i] = loop.resistance[i] * abs(flow) ** 3 / loop.rho
        if i == index:
            a[i, i] -= link
        a[n + i, i] += loop.ua[i]
        a[n + i, n + i] -= loop.ua[i] + loop.ambient_ua[i]
        b[n + i] = loop.heater[i] + loop.ambient_ua[i] * (loop.ambient_temperature_k - anchor)
        if i == machine._index:
            b[n + i] += leak * head**2
        if i == model._heater_index:
            b[n + i] += c["primary_wall_heater_power_w"]
    rows = np.maximum(np.max(abs(a), axis=1), 1.0)
    scaled = a / rows[:, None]
    rank = int(np.linalg.matrix_rank(scaled))
    condition = float(np.linalg.cond(scaled))
    if (
        rank != 2 * n
        or not np.isfinite(condition)
        or condition > budget["maximum_thermal_condition"]
    ):
        _reject(
            "thermal_rank",
            "Primary thermal system is unanchored or ill-conditioned",
            rank=rank,
            unknowns=2 * n,
            scaled_condition=condition if np.isfinite(condition) else None,
        )
    delta = np.linalg.solve(scaled, -b / rows)
    slope = np.linalg.solve(scaled, np.r_[np.zeros(n), loop.ambient_ua] / rows)
    residual = float(max(abs(a @ delta + b)))
    if residual > budget["maximum_linear_residual_w"]:
        _reject(
            "thermal_residual",
            "Primary linear balance residual exceeds budget",
            residual_w=residual,
        )
    return dict(
        flow=flow,
        omega=omega,
        anchor=anchor,
        delta=delta,
        slope=slope,
        q0=float(link * delta[index]),
        qslope=float(link * slope[index]),
        rank=rank,
        condition=condition,
        linear_residual_w=residual,
    )


def solve_fixed_primary_steam(spec, *, total_steam_mass_kg, steam_generator_quality):
    if (
        not isinstance(spec, dict)
        or set(spec) != FIELDS
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["engine"] != PASSPORT.model_id
    ):
        raise NetworkError("Exact primary/steam schema 1 required")
    if not isinstance(spec["timeline"], list) or spec["timeline"]:
        raise NetworkError("Fixed-command steady solve requires an empty timeline")
    require_finite(spec)
    total = number(total_steam_mass_kg, "total_steam_mass_kg", positive=True)
    quality = number(steam_generator_quality, "steam_generator_quality", positive=True)
    fixed = controls(spec["controls"])
    budget = json.loads(BUDGET.read_text())
    probe = deepcopy(spec)
    if not isinstance(probe["primary"], dict) or set(probe["primary"]) != {
        "loop",
        "pump",
        "initial_omega_rad_s",
    }:
        raise NetworkError("Exact primary loop/pump structure required")
    try:
        nodes = probe["primary"]["loop"]["model"]["topology"]["nodes"]
    except (KeyError, TypeError) as exc:
        raise NetworkError("Explicit primary model/topology required") from exc
    if (
        not isinstance(nodes, list)
        or not 2 <= len(nodes) <= 32
        or any(not isinstance(n, dict) or not isinstance(n.get("id"), str) for n in nodes)
    ):
        raise NetworkError("Named primary topology with 2..32 segments required")
    try:
        reference_pressure = number(
            probe["primary"]["loop"]["model"]["reference"]["pressure_pa"],
            "reference pressure",
            positive=True,
        )
        minimum_temperature = number(
            probe["coupling"]["min_primary_temperature_k"],
            "min_primary_temperature_k",
            positive=True,
        )
        subcooling = number(
            probe["coupling"]["minimum_subcooling_k"], "minimum_subcooling_k", positive=True
        )
    except (KeyError, TypeError) as exc:
        raise NetworkError("Explicit primary pressure/temperature constraints required") from exc
    upper_temperature = saturation_properties(reference_pressure)[0].T_k - subcooling
    if minimum_temperature >= upper_temperature:
        _reject(
            "primary_temperature_domain",
            "No interior subcooled temperature exists at the primary pressure reference",
            lower_temperature_k=minimum_temperature,
            upper_temperature_k=upper_temperature,
        )
    probe_temperature = minimum_temperature + (upper_temperature - minimum_temperature) * 0.5
    probe["primary"]["loop"]["initialization"] = dict(
        mode="given",
        mass_flow_kg_s=0,
        fluid_temperature_k={n["id"]: probe_temperature for n in nodes},
        wall_temperature_k={n["id"]: probe_temperature for n in nodes},
    )
    probe["primary"]["initial_omega_rad_s"] = 0
    normalized, model = prepare(probe)
    cfg, c = normalized["steam_model"], fixed["steam_cycle"]
    hot, cold = cfg["steam_generator"], cfg["condenser"]
    if not hot["min_quality"] <= quality <= hot["max_quality"]:
        _reject(
            "inventory_anchor",
            "Boiler quality anchor exceeds declared phase margins",
            quality=quality,
        )
    if c["feed_mass_flow_kg_s"] <= 0 or c["valve_opening"] <= 0:
        _reject(
            "open_flow_required", "This steady formulation requires positive feed and valve opening"
        )
    primary = _primary(model, budget)
    gap = c["feed_mass_flow_kg_s"] / (cfg["steam_conductance_kg_s_pa"] * c["valve_opening"])
    lower = max(cold["min_pressure_pa"], hot["min_pressure_pa"] - gap)
    upper = min(cold["max_pressure_pa"], hot["max_pressure_pa"] - gap)
    if gap < cfg["min_pressure_difference_pa"] or lower >= upper:
        _reject(
            "pressure_compatibility",
            "Feed/valve commands leave no admitted interior pressure bracket",
            pressure_difference_pa=gap,
            condenser_interval_pa=[lower, upper],
        )
    uh, uc, ua = (
        cfg["hot_wall"]["fluid_ua_w_k"],
        cfg["cold_wall"]["fluid_ua_w_k"],
        cfg["cold_wall"]["ambient_ua_w_k"],
    )
    hot_denominator = uh - primary["qslope"]
    if hot_denominator <= 0 or uc + ua <= 0:
        _reject("wall_rank", "Steam-wall steady temperatures are not uniquely anchored")
    rotor_loss = cfg["turbine"]["bearing_drag_nm_s_rad"] + (
        c["generator_conductance_nm_s_rad"] if c["generator_connected"] else 0
    )
    if rotor_loss <= 0:
        _reject("rotor_rank", "A positive-speed steady rotor requires a dissipative load")

    evaluations = 0

    def point(pc):
        nonlocal evaluations
        evaluations += 1
        ph = float(pc + gap)
        hf, hg = saturation_properties(ph)
        cf, _ = saturation_properties(float(pc))
        turbine = expansion(
            dict(
                inlet_pressure_pa=ph,
                outlet_pressure_pa=float(pc),
                isentropic_efficiency=cfg["turbine"]["isentropic_efficiency"],
            )
        )
        th = (
            primary["anchor"]
            + (uh * (hf.T_k - primary["anchor"]) + primary["q0"] + c["heater_power_w"])
            / hot_denominator
        )
        tc = c["sink_temperature_k"] + uc / (uc + ua) * (cf.T_k - c["sink_temperature_k"])
        qh, qc = uh * (th - hf.T_k), uc * (cf.T_k - tc)
        steam = cfg["steam_conductance_kg_s_pa"] * c["valve_opening"] * (ph - pc)
        feed = c["feed_mass_flow_kg_s"]
        work = (ph - pc) / (cf.rho_kg_m3 * cfg["feed_efficiency"])
        return dict(
            ph=ph,
            pc=float(pc),
            th=float(th),
            tc=float(tc),
            hot_w=float(feed * (cf.h_j_kg + work) - steam * hg.h_j_kg + qh),
            cold_w=float(steam * turbine["outlet_enthalpy_j_kg"] - feed * cf.h_j_kg - qc),
            omega=math.sqrt(steam * turbine["work_j_kg"] / rotor_loss),
        )

    samples = np.linspace(lower, upper, budget["branch_samples"])
    try:
        profile = [point(float(pc)) for pc in samples]
    except (NetworkError, DomainError) as exc:
        _reject(
            "thermodynamic_domain",
            "Pressure bracket is outside supported turbine thermodynamics",
            error=str(exc),
        )
    slopes = np.diff([p["cold_w"] for p in profile]) / np.diff(samples)
    minimum = budget["minimum_cold_balance_slope_w_pa"]
    if not (np.all(slopes < -minimum) or np.all(slopes > minimum)):
        _reject(
            "ambiguous_branch",
            "Sampled condenser balance is not strictly monotone; branch selection is unsupported",
        )
    tolerance = budget["maximum_stock_power_residual_w"]
    endpoint = next((p for p in (profile[0], profile[-1]) if abs(p["cold_w"]) <= tolerance), None)
    if endpoint is None and np.sign(profile[0]["cold_w"]) == np.sign(profile[-1]["cold_w"]):
        _reject(
            "condenser_compatibility",
            "Fixed cooling/feed commands have no condenser balance in the admitted bracket",
            endpoint_residuals_w=[profile[0]["cold_w"], profile[-1]["cold_w"]],
        )
    if endpoint is not None:
        pc, iterations, calls = endpoint["pc"], 0, 1
        root_mode = "endpoint_within_residual_budget"
    else:
        try:
            pc, root = brentq(
                lambda p: point(p)["cold_w"],
                lower,
                upper,
                xtol=budget["pressure_root_xtol_pa"],
                rtol=budget["pressure_root_rtol"],
                maxiter=budget["maximum_root_iterations"],
                full_output=True,
            )
        except (ValueError, RuntimeError) as exc:
            _reject("pressure_solver", "Condenser pressure root did not converge", error=str(exc))
        iterations, calls, root_mode = int(root.iterations), int(root.function_calls), "brentq"
    solved = point(pc)
    if max(abs(solved["hot_w"]), abs(solved["cold_w"])) > budget["maximum_stock_power_residual_w"]:
        _reject(
            "hot_energy_compatibility",
            "Fixed commands do not satisfy both vessel energy balances",
            hot_inventory_residual_w=solved["hot_w"],
            cold_inventory_residual_w=solved["cold_w"],
            condenser_pressure_pa=pc,
            commands=fixed,
        )
    pressure_margins = {
        "steam_generator_low_pa": solved["ph"] - hot["min_pressure_pa"],
        "steam_generator_high_pa": hot["max_pressure_pa"] - solved["ph"],
        "condenser_low_pa": pc - cold["min_pressure_pa"],
        "condenser_high_pa": cold["max_pressure_pa"] - pc,
    }
    if min(pressure_margins.values()) < budget["minimum_numerical_pressure_margin_pa"]:
        _reject(
            "active_pressure_boundary",
            "Steady initializer requires a numerical interior margin; stationary exact-boundary events are not admitted",
            pressure_margins_pa=pressure_margins,
            required_numerical_margin_pa=budget["minimum_numerical_pressure_margin_pa"],
            pressure_root_mode=root_mode,
            pressure_root_iterations=iterations,
            algebraic_point_evaluations=evaluations,
        )
    turbine = cfg["turbine"]
    if not turbine["min_omega_rad_s"] <= solved["omega"] <= turbine["max_omega_rad_s"]:
        _reject(
            "rotor_band",
            "Steady generator speed exceeds the declared operating band",
            omega_rad_s=solved["omega"],
        )
    liquid, vapor = saturation_properties(solved["ph"])
    mh = hot["volume_m3"] / ((1 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg)
    mc = total - mh
    if mc <= 0:
        _reject(
            "inventory_anchor",
            "Total mass leaves no positive condenser inventory",
            total_mass_kg=total,
            boiler_mass_kg=mh,
        )
    liquid, vapor = saturation_properties(pc)
    xc = (cold["volume_m3"] / mc - liquid.v_m3_kg) / (vapor.v_m3_kg - liquid.v_m3_kg)
    if not cold["min_quality"] <= xc <= cold["max_quality"]:
        _reject(
            "inventory_anchor",
            "Mass/quality anchors require condenser quality outside its passport",
            condenser_quality=xc,
        )
    result = deepcopy(normalized)
    result["steam_model"]["steam_generator"].update(
        initial_pressure_pa=solved["ph"], initial_quality=quality
    )
    result["steam_model"]["condenser"].update(initial_pressure_pa=pc, initial_quality=float(xc))
    result["steam_model"]["hot_wall"]["initial_temperature_k"] = solved["th"]
    result["steam_model"]["cold_wall"]["initial_temperature_k"] = solved["tc"]
    result["steam_model"]["turbine"]["initial_omega_rad_s"] = solved["omega"]
    delta = primary["delta"] + primary["slope"] * (solved["th"] - primary["anchor"])
    loop = model._primary._loop
    result["primary"]["initial_omega_rad_s"] = primary["omega"]
    result["primary"]["loop"]["initialization"] = dict(
        mode="given",
        mass_flow_kg_s=primary["flow"],
        fluid_temperature_k=dict(zip(loop.names, map(float, solved["th"] + delta[: loop.n]))),
        wall_temperature_k=dict(zip(loop.names, map(float, solved["th"] + delta[loop.n :]))),
    )
    try:
        result, final = prepare(result)
    except (NetworkError, DomainError) as exc:
        _reject(
            "physical_domain", "Solved point exceeds the coupled physical passport", error=str(exc)
        )
    if result["controls"] != fixed:
        raise AssertionError("Fixed boundary commands were rewritten")
    residuals = _residuals(result, final)
    for key, value in residuals.items():
        if not np.isfinite(value) or value > budget[key]:
            _reject(
                "residual_verification",
                "Production or independent residual exceeds the fixed budget",
                residuals=residuals,
            )
    observation = final.observation()
    domain_ratios = {
        name: value["value"] / value["root_tolerance"]
        for name, value in observation["domain_margins"].items()
    }
    if min(domain_ratios.values()) < budget["minimum_domain_margin_root_tolerances"]:
        _reject(
            "active_state_boundary",
            "Steady initialization requires numerical interior margin at every passport boundary",
            insufficient_margins={
                name: value
                for name, value in domain_ratios.items()
                if value < budget["minimum_domain_margin_root_tolerances"]
            },
        )
    observed = observation["steam_cycle"]["vessels"]
    mass_error = abs(sum(observed[n]["mass_kg"] for n in observed) - total)
    quality_error = abs(observed["steam_generator"]["quality"] - quality)
    if (
        mass_error > budget["maximum_total_mass_error_kg"]
        or quality_error > budget["maximum_quality_anchor_error"]
    ):
        _reject(
            "inventory_verification",
            "Reconstructed stocks do not satisfy explicit inventory anchors",
            mass_error_kg=mass_error,
            quality_error=quality_error,
        )
    report = dict(
        schema_version=1,
        accepted=True,
        mode="steady_fixed_commands",
        warmup_steps=0,
        commands_changed=False,
        fixed_commands=fixed,
        inventory_anchors=dict(total_steam_mass_kg=total, steam_generator_quality=quality),
        thermal_rank=primary["rank"],
        thermal_unknowns=2 * loop.n,
        thermal_scaled_condition=primary["condition"],
        thermal_linear_residual_w=primary["linear_residual_w"],
        pressure_interval_pa=[lower, upper],
        numerical_pressure_margins_pa=pressure_margins,
        minimum_domain_margin_in_root_tolerances=min(domain_ratios.values()),
        parameter_probe_temperature_k=probe_temperature,
        branch_samples=len(profile),
        sampled_branch_monotone=True,
        global_uniqueness_proven=False,
        pressure_root_mode=root_mode,
        pressure_root_iterations=iterations,
        pressure_root_function_calls=calls,
        algebraic_point_evaluations=evaluations,
        hot_inventory_residual_w=solved["hot_w"],
        cold_inventory_residual_w=solved["cold_w"],
        total_mass_error_kg=mass_error,
        quality_anchor_error=quality_error,
        dimensional_residuals=residuals,
        domain_margins=final.observation()["domain_margins"],
        validated=False,
        plant_referenced=False,
    )
    return result, report


def _residuals(spec, model):
    rates = model.derivatives(0, model._state)
    p, k = model._primary, model._offset
    cfg = spec["steam_model"]
    powers = np.r_[
        p._loop.capacity * rates[1 : 1 + p._loop.n],
        p._loop.wall_capacity * rates[1 + p._loop.n : 1 + 2 * p._loop.n],
        rates[k + 1],
        rates[k + 3],
        cfg["hot_wall"]["heat_capacity_j_k"] * rates[k + 4],
        cfg["cold_wall"]["heat_capacity_j_k"] * rates[k + 5],
        rates[k + 6],
    ]
    initial, rhs, _, _, data, offset = system(spec)
    independent = rhs(0, initial, spec["controls"])
    n = data["n"]
    other = np.r_[
        data["capacity"] * independent[2 : 2 + n],
        data["wall_capacity"] * independent[2 + n : 2 + 2 * n],
        independent[offset + 1],
        independent[offset + 3],
        cfg["hot_wall"]["heat_capacity_j_k"] * independent[offset + 4],
        cfg["cold_wall"]["heat_capacity_j_k"] * independent[offset + 5],
        independent[offset + 6],
    ]
    return dict(
        maximum_stock_power_residual_w=float(max(max(abs(powers)), max(abs(other)))),
        maximum_primary_momentum_residual_pa=float(
            max(abs(p._loop.inertia_total * rates[0]), abs(sum(data["inertia"]) * independent[0]))
        ),
        maximum_primary_shaft_residual_nm=float(
            max(
                abs(p.pump["inertia_kg_m2"] * rates[p._size]),
                abs(data["pump"]["inertia_kg_m2"] * independent[1]),
            )
        ),
        maximum_steam_mass_rate_kg_s=float(
            max(
                abs(rates[k]),
                abs(rates[k + 2]),
                abs(independent[offset]),
                abs(independent[offset + 2]),
            )
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--total-steam-mass-kg", type=float, required=True)
    parser.add_argument("--steam-generator-quality", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    request = json.loads(args.scenario.read_text())
    try:
        normalized, report = solve_fixed_primary_steam(
            request,
            total_steam_mass_kg=args.total_steam_mass_kg,
            steam_generator_quality=args.steam_generator_quality,
        )
    except SteadySolveError as error:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "request.json", request)
        write(args.output / "diagnostics.json", error.report)
        names = ("request.json", "diagnostics.json")
        code = 2
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "request.json", request)
        write(args.output / "input.json", normalized)
        write(args.output / "steady.json", report)
        names = ("request.json", "input.json", "steady.json")
        code = 0
    write(
        args.output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=_source_fingerprint(),
            initializer_sha256=sha(__file__),
            driver_sha256={path: sha(path) for path in DRIVERS},
            budget_sha256=sha(BUDGET),
            artifacts={name: sha(args.output / name) for name in names},
        ),
    )
    print(json.dumps(dict(accepted=code == 0, output=str(args.output))))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
