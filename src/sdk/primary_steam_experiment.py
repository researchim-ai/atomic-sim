"""Reproducible monolithic primary-loop/steam-cycle jobs with physical stops."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from physics.power.primary_steam import PASSPORT, PrimarySteamCycle, controls
from physics.power.steam_cycle import PASSPORT as STEAM_PASSPORT
from sdk.boundary_job import run_boundary_job
from sdk.pump_experiment import prepare as prepare_pump
from sdk.steam_cycle_experiment import prepare as prepare_steam


def prepare(spec):
    fields = {
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
    if (
        not isinstance(spec, dict)
        or set(spec) != fields
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["engine"] != PASSPORT.model_id
    ):
        raise NetworkError("Exact primary/steam experiment schema 1 required")
    boundary = controls(spec["controls"])
    normalized_steam, steam = prepare_steam(
        dict(
            schema_version=1,
            engine=STEAM_PASSPORT.model_id,
            name=spec["name"],
            model=spec["steam_model"],
            controls=boundary["steam_cycle"],
            duration_s=spec["duration_s"],
            step_s=spec["step_s"],
            solver=spec["solver"],
            timeline=[],
        )
    )
    primary = spec["primary"]
    if not isinstance(primary, dict) or set(primary) != {"loop", "pump", "initial_omega_rad_s"}:
        raise NetworkError("Exact initial primary loop/pump definition required")
    normalized_primary, pump = prepare_pump(
        dict(
            schema_version=1,
            engine="displacement-pump-circulation-loop",
            name=spec["name"],
            **primary,
            motor_torque_nm=boundary["primary_motor_torque_nm"],
            duration_s=normalized_steam["duration_s"],
            step_s=normalized_steam["step_s"],
            solver=normalized_steam["solver"],
            timeline=[],
        )
    )
    model = PrimarySteamCycle(pump.snapshot(), steam.snapshot(), spec["coupling"], boundary)
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Primary/steam command resource limit exceeded")
    timeline, previous = [], -1.0
    for event in spec["timeline"]:
        if not isinstance(event, dict) or set(event) != {"t", "controls"}:
            raise NetworkError("Exact primary/steam control event required")
        t = number(event["t"], "command time")
        if not previous < t <= normalized_steam["duration_s"] or t < 0:
            raise NetworkError("Primary/steam commands must increase inside horizon")
        timeline.append(dict(t=t, controls=controls(event["controls"])))
        previous = t
    result = deepcopy(spec)
    result.update(
        primary={key: normalized_primary[key] for key in primary},
        steam_model=normalized_steam["model"],
        coupling=model.configuration["coupling"],
        controls=model.boundary_controls,
        duration_s=normalized_steam["duration_s"],
        step_s=normalized_steam["step_s"],
        solver=normalized_steam["solver"],
        timeline=timeline,
    )
    return result, model


def restore(spec, payload):
    normalized, fresh = prepare(spec)
    model = PrimarySteamCycle.from_snapshot(payload)
    if (
        model.configuration != fresh.configuration
        or model._initial_controls != fresh._initial_controls
        or model.time_s > normalized["duration_s"]
    ):
        raise NetworkError("Primary/steam restart differs from initial configuration/horizon")
    actual = [dict(t=e["t"], controls=e["after"]) for e in model.events if e["kind"] == "controls"]
    expected = [e for e in normalized["timeline"] if e["t"] <= model.time_s]
    if actual != expected[: len(actual)] or len(actual) < len(
        [e for e in expected if e["t"] < model.time_s]
    ):
        raise NetworkError("Primary/steam restart differs from executed command prefix")
    return normalized, model


def run_primary_steam_experiment(spec, output, *, checkpoint=None, wall_limit_s=None):
    return run_boundary_job(
        spec,
        output,
        prepare=prepare,
        restore=restore,
        model_class=PrimarySteamCycle,
        passport=PASSPORT,
        initial_artifacts=lambda model: {"initialization.json": model.initialization_report()},
        checkpoint=checkpoint,
        wall_limit_s=wall_limit_s,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wall-limit-s", type=float)
    args = parser.parse_args(argv)
    report = run_primary_steam_experiment(
        json.loads(args.scenario.read_text()),
        args.output,
        checkpoint=json.loads(args.resume.read_text()) if args.resume else None,
        wall_limit_s=args.wall_limit_s,
    )
    print(
        json.dumps(
            dict(
                status=report["status"],
                completed_time_s=report["completed_time_s"],
                numerical_acceptance=report["numerical_acceptance"],
            )
        )
    )
    return 0 if report["status"] == "completed" else 2 if report["status"] == "domain_exit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
