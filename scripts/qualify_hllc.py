"""Case-scoped grid/time qualification with exact contact references and raw states.

Rejected/inconclusive cases stay in the matrix. No universal spatial, acoustic,
positivity or real pressure-port claim follows from this numerical qualification.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.status import DomainError, SolverError
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from physics.hydraulics.euler_initialization import steady_contact_config
from scripts.contact_wave_cases import configuration, errors
from sdk.experiment import _source_fingerprint


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=lambda v: v.item())
        stream.write("\n")


def run_case(folder, cfg, duration, budget, *, tight=False, fast=False, solver_policy=None):
    folder.mkdir()
    solver = dict(
        method="Radau",
        rtol=1e-11 if tight else 1e-9,
        max_step_s=(5e-5 if tight else 1e-4) if fast else (0.01 if tight else 0.02),
    )
    if cfg.numerical_flux == "hllc" and "solver_atol_scale_base" in budget:
        solver["atol_scale"] = budget[
            "solver_atol_scale_tight" if tight else "solver_atol_scale_base"
        ]
    if solver_policy is not None:
        if not isinstance(solver_policy, dict) or set(solver_policy) != set(solver):
            raise ValueError("Explicit solver policy requires method, rtol and max_step_s")
        solver = dict(solver_policy)
    write(folder / "input.json", dict(config=cfg.as_dict(), duration_s=duration, solver=solver))
    model = EulerLiquidChannel(cfg)
    model.enable_continuation(duration)
    before = model.derivatives
    started = time.monotonic()

    def guarded(t, y):
        if time.monotonic() - started > budget["case_wall_limit_s"]:
            raise SolverError("Qualification wall deadline exceeded")
        return before(t, y)

    model.derivatives = guarded
    rows = [
        dict(
            observation=model.observation(),
            state=model._state.tolist(),
            ports=model.port_observation(),
        )
    ]
    events = []
    status = "completed"
    failure = None
    try:
        for end in np.linspace(duration / 10, duration, 10):
            obs = model.advance(float(end) - model.time_s, **solver)
            rows.append(
                dict(observation=obs, state=model._state.tolist(), ports=model.port_observation())
            )
            events.extend(model.last_events)
    except (DomainError, SolverError) as exc:
        status = "failed"
        failure = dict(type=type(exc).__name__, message=str(exc))
    balances = dict(
        energy_abs_j=max(abs(r["observation"]["energy_residual_j"]) for r in rows),
        momentum_abs_kg_m_s=max(abs(r["observation"]["momentum_residual_kg_m_s"]) for r in rows),
        mass_relative=max(abs(r["observation"]["mass_residual_kg"]) for r in rows)
        / model.initial_mass_kg,
    )
    accepted = status == "completed" and all(v <= budget[k] for k, v in balances.items())
    snapshot = model.snapshot()
    restored = EulerLiquidChannel.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    restart = True
    if accepted:
        restored.advance(min(0.00001, duration / 10), rtol=1e-9)
        restart = (
            abs(restored.observation()["energy_residual_j"]) <= budget["energy_abs_j"]
            and abs(restored.observation()["momentum_residual_kg_m_s"])
            <= budget["momentum_abs_kg_m_s"]
        )
    write(folder / "trajectory.json", rows)
    write(folder / "checkpoint.json", snapshot)
    report = dict(
        status=status,
        accepted=accepted,
        balances=balances,
        events=events,
        wall_time_s=time.monotonic() - started,
        restart_passed=restart,
        error=failure,
    )
    write(folder / "report.json", report)
    return dict(model=model, rows=rows, report=report, path=str(folder))


def profile_difference(coarse, fine):
    a, b = coarse["model"], fine["model"]
    n = a.config.cells
    if b.config.cells != 2 * n:
        raise ValueError("Uniform doubling grids required")
    values = []
    for ra, rb in zip(coarse["rows"], fine["rows"]):
        if ra["observation"]["time_s"] != rb["observation"]["time_s"]:
            raise ValueError("Grid sample clocks differ")
        qa = np.array(ra["state"][: 3 * n]).reshape(n, 3)
        qb = np.array(rb["state"][: 6 * n]).reshape(n, 2, 3).sum(axis=1)
        sa = [a.cell(q) for q in qa]
        sb = [a.cell(q) for q in qb]
        ta = np.array([v[0].T_k for v in sa])
        tb = np.array([v[0].T_k for v in sb])
        pa = np.array([v[0].p_pa for v in sa])
        pb = np.array([v[0].p_pa for v in sb])
        ua = qa[:, 1] / qa[:, 0]
        ub = qb[:, 1] / qb[:, 0]
        values.append(
            dict(
                temperature_l1_k=float(np.mean(abs(ta - tb))),
                temperature_l2_k=float(np.sqrt(np.mean((ta - tb) ** 2))),
                temperature_linf_k=float(max(abs(ta - tb))),
                pressure_l1_pa=float(np.mean(abs(pa - pb))),
                velocity_l1_m_s=float(np.mean(abs(ua - ub))),
                port_mass_flow_abs_kg_s=max(
                    abs(ra["ports"][side]["mass_flow_kg_s"] - rb["ports"][side]["mass_flow_kg_s"])
                    for side in ("left", "right")
                ),
                port_energy_flow_abs_w=max(
                    abs(ra["ports"][side]["energy_flow_w"] - rb["ports"][side]["energy_flow_w"])
                    for side in ("left", "right")
                ),
            )
        )
    return {key: max(v[key] for v in values) for key in values[0]}


def exact_over_time(run, **parameters):
    probe = EulerLiquidChannel(run["model"].config)
    values = []
    for row in run["rows"]:
        probe._state = np.array(row["state"])
        values.append(errors(probe, row["observation"]["time_s"], **parameters))
    result = {}
    for key in values[0]:
        if key == "conserved_l1":
            result[key] = np.max([v[key] for v in values], axis=0).tolist()
        elif key == "min_temperature_k":
            result[key] = min(v[key] for v in values)
        elif key == "temperature_signed_integral_k_m":
            result[key] = max((v[key] for v in values), key=abs)
        else:
            result[key] = max(v[key] for v in values)
    return result


def time_difference(base, tight):
    differences = []
    for a, b in zip(base["rows"], tight["rows"]):
        qa = np.array(a["state"][: 3 * base["model"].config.cells]).reshape(-1, 3)
        qb = np.array(b["state"][: 3 * base["model"].config.cells]).reshape(-1, 3)
        differences.extend(
            abs(base["model"].cell(x)[0].T_k - base["model"].cell(y)[0].T_k) for x, y in zip(qa, qb)
        )
    return float(max(differences))


def observed_order(a, b):
    return float(np.log2(a / b)) if a > 0 and b > 0 else None


def build(output, budget):
    entries = []
    results = {}
    for shape in ("smooth", "step"):
        for velocity in (-1.0, 1.0):
            name = f"{shape}_" + ("negative" if velocity < 0 else "positive")
            runs = []
            exact = []
            for n in (16, 32, 64):
                run = run_case(
                    output / f"{name}_{n}",
                    configuration(n, velocity=velocity, shape=shape),
                    budget["duration_contact_s"],
                    budget,
                )
                runs.append(run)
                exact.append(exact_over_time(run, velocity=velocity, shape=shape))
            tight = run_case(
                output / f"{name}_64_tight",
                configuration(64, velocity=velocity, shape=shape),
                budget["duration_contact_s"],
                budget,
                tight=True,
            )
            td = time_difference(runs[-1], tight)
            spatial = profile_difference(runs[-2], runs[-1])["temperature_l1_k"]
            key = "temperature_l1_k"
            orders = [observed_order(a[key], b[key]) for a, b in zip(exact, exact[1:])]
            min_order = (
                budget["smooth_min_order"] if shape == "smooth" else budget["step_min_order"]
            )
            checks = dict(
                completed=all(
                    r["report"]["accepted"] and r["report"]["restart_passed"]
                    for r in runs + [tight]
                ),
                spatial=exact[-1]["temperature_linf_k"] <= budget["smooth_linf_k"]
                if shape == "smooth"
                else exact[-1][key] <= budget["step_l1_k"],
                order=all(p is not None and p >= min_order for p in orders),
                temporal=td <= budget["temperature_time_abs_k"]
                and td <= budget["time_spatial_ratio"] * spatial + budget["temperature_floor_k"],
                pressure=exact[-1]["pressure_error_pa"] <= budget["contact_pressure_abs_pa"],
                bounds=exact[-1]["min_temperature_k"] >= 500 - 0.01 - budget["overshoot_k"]
                and exact[-1]["max_temperature_k"] <= 500 + 0.01 + budget["overshoot_k"],
            )
            entry = dict(
                id=name,
                accepted=all(checks.values()),
                checks=checks,
                grids=[16, 32, 64],
                exact_errors=exact,
                orders=orders,
                time_linf_k=td,
                fine_spatial_difference_k=spatial,
                qualified_cells=[64] if all(checks.values()) else [],
                paths=[r["path"] for r in runs + [tight]],
            )
            entries.append(entry)
            results[name] = runs
            print(
                json.dumps(dict(case=name, accepted=entry["accepted"], time_linf_k=td)), flush=True
            )
            if shape == "step" and "step_refinement_grids" in budget:
                grids = budget["step_refinement_grids"]
                if grids != [32, 64, 128]:
                    raise ValueError(
                        "Refinement protocol requires 32/64/128, retaining 16-cell rejection"
                    )
                refined_name = name + "_refined"
                extra = run_case(
                    output / f"{refined_name}_128",
                    configuration(128, velocity=velocity, shape=shape),
                    budget["duration_contact_s"],
                    budget,
                )
                extra_tight = run_case(
                    output / f"{refined_name}_128_tight",
                    extra["model"].config,
                    budget["duration_contact_s"],
                    budget,
                    tight=True,
                )
                refined_errors = exact[1:] + [
                    exact_over_time(extra, velocity=velocity, shape=shape)
                ]
                refined_orders = [
                    observed_order(a[key], b[key])
                    for a, b in zip(refined_errors, refined_errors[1:])
                ]
                refined_td = time_difference(extra, extra_tight)
                refined_spatial = profile_difference(runs[-1], extra)[key]
                refined_checks = dict(
                    completed=all(
                        r["report"]["accepted"] and r["report"]["restart_passed"]
                        for r in runs[1:] + [extra, extra_tight]
                    ),
                    spatial=refined_errors[-1][key] <= budget["step_l1_k"],
                    order=all(p is not None and p >= min_order for p in refined_orders),
                    temporal=refined_td <= budget["temperature_time_abs_k"]
                    and refined_td
                    <= budget["time_spatial_ratio"] * refined_spatial
                    + budget["temperature_floor_k"],
                    pressure=refined_errors[-1]["pressure_error_pa"]
                    <= budget["contact_pressure_abs_pa"],
                    bounds=refined_errors[-1]["min_temperature_k"]
                    >= 500 - 0.01 - budget["overshoot_k"]
                    and refined_errors[-1]["max_temperature_k"]
                    <= 500 + 0.01 + budget["overshoot_k"],
                )
                entries.append(
                    dict(
                        id=refined_name,
                        accepted=all(refined_checks.values()),
                        checks=refined_checks,
                        grids=grids,
                        diagnostic_fourth_grid=16,
                        original_entry=name,
                        exact_errors=refined_errors,
                        orders=refined_orders,
                        time_linf_k=refined_td,
                        fine_spatial_difference_k=refined_spatial,
                        qualified_cells=[128] if all(refined_checks.values()) else [],
                        paths=[r["path"] for r in runs[1:] + [extra, extra_tight]],
                        scope="Refined 32/64/128 sequence; original 16/32/64 rejection retained",
                    )
                )
                print(
                    json.dumps(
                        dict(
                            case=refined_name,
                            accepted=all(refined_checks.values()),
                            orders=refined_orders,
                        )
                    ),
                    flush=True,
                )
                if "step_asymptotic_grids" in budget:
                    if budget["step_asymptotic_grids"] != [128, 256, 512]:
                        raise ValueError("Asymptotic protocol requires 128/256/512")
                    asym_name = name + "_asymptotic"
                    asym_runs = [extra]
                    asym_errors = [refined_errors[-1]]
                    for n in (256, 512):
                        run = run_case(
                            output / f"{asym_name}_{n}",
                            configuration(n, velocity=velocity, shape=shape),
                            budget["duration_contact_s"],
                            budget,
                        )
                        asym_runs.append(run)
                        asym_errors.append(exact_over_time(run, velocity=velocity, shape=shape))
                    asym_tight = run_case(
                        output / f"{asym_name}_512_tight",
                        asym_runs[-1]["model"].config,
                        budget["duration_contact_s"],
                        budget,
                        tight=True,
                    )
                    asym_td = time_difference(asym_runs[-1], asym_tight)
                    asym_spatial = profile_difference(asym_runs[-2], asym_runs[-1])[key]
                    asym_orders = [
                        observed_order(a[key], b[key]) for a, b in zip(asym_errors, asym_errors[1:])
                    ]
                    fine_error = asym_errors[-1]
                    asym_checks = dict(
                        completed=all(
                            r["report"]["accepted"] and r["report"]["restart_passed"]
                            for r in asym_runs + [asym_tight]
                        ),
                        spatial=fine_error[key] <= budget["step_l1_k"],
                        order=all(p is not None and p >= min_order for p in asym_orders),
                        temporal=asym_td <= budget["temperature_time_abs_k"]
                        and asym_td
                        <= budget["time_spatial_ratio"] * asym_spatial
                        + budget["temperature_floor_k"],
                        pressure=fine_error["pressure_error_pa"]
                        <= budget["contact_pressure_abs_pa"],
                        bounds=fine_error["min_temperature_k"] >= 499.99 - budget["overshoot_k"]
                        and fine_error["max_temperature_k"] <= 500.01 + budget["overshoot_k"],
                    )
                    entries.append(
                        dict(
                            id=asym_name,
                            accepted=all(asym_checks.values()),
                            checks=asym_checks,
                            grids=[128, 256, 512],
                            diagnostic_grids=[16, 32, 64],
                            original_entry=refined_name,
                            exact_errors=asym_errors,
                            orders=asym_orders,
                            time_linf_k=asym_td,
                            fine_spatial_difference_k=asym_spatial,
                            qualified_cells=[512] if all(asym_checks.values()) else [],
                            paths=[r["path"] for r in asym_runs + [asym_tight]],
                            scope="Asymptotic 128/256/512 sequence; both coarser rejected sequences retained",
                        )
                    )
                    print(
                        json.dumps(
                            dict(
                                case=asym_name,
                                accepted=all(asym_checks.values()),
                                orders=asym_orders,
                            )
                        ),
                        flush=True,
                    )
    baseline = run_case(
        output / "rusanov_smooth_32",
        configuration(32, flux="rusanov"),
        budget["duration_contact_s"],
        budget,
    )
    baseline_error = exact_over_time(baseline)["temperature_l1_k"]
    hllc_error = entries[1]["exact_errors"][1]["temperature_l1_k"]
    ratio = hllc_error / baseline_error
    entries.append(
        dict(
            id="contact_diffusion_comparison",
            accepted=baseline["report"]["accepted"]
            and ratio <= budget["diffusion_error_ratio_max"],
            hllc_to_rusanov_l1_error_ratio=ratio,
            hllc_l1_k=hllc_error,
            rusanov_l1_k=baseline_error,
        )
    )
    # Stationary discontinuity: no artificial estimate of convergence order.
    stationary = run_case(
        output / "stationary_contact_32", configuration(32, velocity=0, shape="step"), 0.3, budget
    )
    error = exact_over_time(stationary, velocity=0, shape="step")
    entries.append(
        dict(
            id="stationary_contact",
            accepted=stationary["report"]["accepted"]
            and error["temperature_linf_k"] <= budget["overshoot_k"],
            exact_errors=error,
            order=None,
            order_note="Stationary solution; absolute error criterion",
            qualified_cells=[32]
            if stationary["report"]["accepted"]
            and error["temperature_linf_k"] <= budget["overshoot_k"]
            else [],
            paths=[stationary["path"]],
        )
    )
    for name in ("heated_steady", "wall_pulse", "startup", "reversal"):
        runs = []
        for n in (8, 16, 32):
            cfg = EulerConfig(
                cells=n, numerical_flux="hllc", left_pressure_pa=7.02e6, initial_pressure_pa=7.01e6
            )
            if name in ("heated_steady", "wall_pulse"):
                cfg = steady_contact_config(replace(cfg, heat_w=10.0))
                if name == "wall_pulse":
                    cfg = replace(
                        cfg,
                        initial_wall_temperature_profile_k=tuple(
                            t + 2 for t in cfg.initial_wall_temperature_profile_k
                        ),
                    )
            else:
                cfg = replace(
                    cfg, total_ua_w_k=0, initial_velocity_m_s=-0.005 if name == "reversal" else 0.0
                )
            duration = 0.1 if name == "heated_steady" else 1.0 if name == "wall_pulse" else 0.005
            run = run_case(
                output / f"{name}_{n}", cfg, duration, budget, fast=name in ("startup", "reversal")
            )
            runs.append(run)
        tight = run_case(
            output / f"{name}_32_tight",
            runs[-1]["model"].config,
            duration,
            budget,
            tight=True,
            fast=name in ("startup", "reversal"),
        )
        differences = [profile_difference(a, b) for a, b in zip(runs, runs[1:])]
        td = time_difference(runs[-1], tight)
        fine = differences[-1]["temperature_l1_k"]
        order = (
            observed_order(differences[0]["temperature_l1_k"], fine)
            if fine > budget["temperature_floor_k"]
            else None
        )
        checks = dict(
            completed=all(
                r["report"]["accepted"] and r["report"]["restart_passed"] for r in runs + [tight]
            ),
            temperature=fine <= budget["self_temperature_l1_k"],
            pressure=differences[-1]["pressure_l1_pa"] <= budget["self_pressure_l1_pa"],
            velocity=differences[-1]["velocity_l1_m_s"] <= budget["self_velocity_l1_m_s"],
            order=order is None or order >= budget["self_min_order"],
            temporal=td <= budget["temperature_time_abs_k"]
            and td <= budget["time_spatial_ratio"] * fine + budget["temperature_floor_k"],
        )
        events = None
        if name == "reversal":
            events = []
            for r in runs + [tight]:
                found = [e["t"] for e in r["report"]["events"] if e["direction"] == 1]
                events.append(found[0] if found else None)
            checks["event"] = (
                all(t is not None for t in events)
                and abs(events[-1] - events[-2]) <= budget["event_time_error_s"]
                and abs(events[-2] - events[-3]) <= budget["event_time_error_s"]
            )
        entries.append(
            dict(
                id=name,
                accepted=all(checks.values()),
                checks=checks,
                grids=[8, 16, 32],
                profile_differences=differences,
                temperature_order=order,
                time_linf_k=td,
                event_times_s=events,
                qualified_cells=[32] if all(checks.values()) else [],
                paths=[r["path"] for r in runs + [tight]],
                reference="self-convergence for declared exterior-state boundary problem, not experimental truth",
            )
        )
        print(json.dumps(dict(case=name, accepted=all(checks.values()), checks=checks)), flush=True)
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--budget", type=Path, default=Path("validation/ai04_qualification_budget_v1.json")
    )
    args = p.parse_args()
    args.output.mkdir()
    budget = json.loads(args.budget.read_text())
    source = _source_fingerprint()
    write(args.output / "budget.json", budget)
    entries = build(args.output, budget)
    if source != _source_fingerprint():
        raise RuntimeError("Source changed during qualification")
    matrix = dict(
        schema_version=1,
        entries=entries,
        all_cases_accepted=all(e["accepted"] for e in entries),
        source_sha256=source,
        budget_sha256=hashlib.sha256(args.budget.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validated=False,
        plant_referenced=False,
        acoustic_qualified=False,
        exact_pressure_ports_qualified=False,
        scope="Listed finite-amplitude numerical cases, prescribed exterior states; no generic whole-plant or nonlinear shock qualification",
    )
    write(args.output / "qualification.json", matrix)
    hashes = {
        str(f.relative_to(args.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in args.output.rglob("*")
        if f.is_file()
    }
    write(args.output / "manifest.json", dict(sha256=hashes))
    print(json.dumps(dict(all_cases_accepted=matrix["all_cases_accepted"], cases=len(entries))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
