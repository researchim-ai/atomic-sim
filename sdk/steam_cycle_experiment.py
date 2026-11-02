"""Reproducible dynamic two-inventory steam-cycle experiments."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from physics.power.steam_cycle import PASSPORT, SteamPowerCycle, controls
from sdk.boundary_job import run_boundary_job


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
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
        or not isinstance(spec["name"], str)
        or not spec["name"].strip()
    ):
        raise NetworkError("Exact named steam-cycle experiment schema 1 required")
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 10000:
        raise NetworkError("Steam-cycle output resource limits exceeded")
    solver = spec["solver"]
    if (
        not isinstance(solver, dict)
        or set(solver) != {"method", "rtol", "max_step_s", "atol_scale"}
        or solver["method"] not in ("Radau", "BDF")
    ):
        raise NetworkError("Exact steam-cycle solver policy required")
    solver = {
        key: (value if key == "method" else number(value, key, positive=True))
        for key, value in solver.items()
    }
    if not 1e-12 <= solver["rtol"] <= 1e-2 or not 1e-6 <= solver["atol_scale"] <= 1:
        raise NetworkError("Unsupported steam-cycle solver tolerance")
    model = SteamPowerCycle(spec["model"], spec["controls"])
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Steam-cycle command resource limit exceeded")
    timeline, previous = [], -1.0
    for event in spec["timeline"]:
        if not isinstance(event, dict) or set(event) != {"t", "controls"}:
            raise NetworkError("Exact steam-cycle control command required")
        t = number(event["t"], "command time")
        if not previous < t <= duration or t < 0:
            raise NetworkError("Steam-cycle commands must increase inside horizon")
        timeline.append(dict(t=t, controls=controls(event["controls"])))
        previous = t
    result = deepcopy(spec)
    result.update(
        model=model.configuration,
        controls=model.boundary_controls,
        duration_s=duration,
        step_s=step,
        solver=solver,
        timeline=timeline,
    )
    return result, model


def restore(spec, payload):
    normalized, fresh = prepare(spec)
    model = SteamPowerCycle.from_snapshot(payload)
    if (
        model.configuration != fresh.configuration
        or model._initial_controls != fresh._initial_controls
        or model.time_s > normalized["duration_s"]
    ):
        raise NetworkError("Steam-cycle restart differs from scenario configuration/horizon")
    actual = [dict(t=e["t"], controls=e["after"]) for e in model.events if e["kind"] == "controls"]
    expected = [e for e in normalized["timeline"] if e["t"] <= model.time_s]
    if actual != expected[: len(actual)] or len(actual) < len(
        [e for e in expected if e["t"] < model.time_s]
    ):
        raise NetworkError("Steam-cycle restart differs from executed control prefix")
    return normalized, model


def run_steam_cycle_experiment(spec, output, *, checkpoint=None, wall_limit_s=None):
    return run_boundary_job(
        spec,
        output,
        prepare=prepare,
        restore=restore,
        model_class=SteamPowerCycle,
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
    report = run_steam_cycle_experiment(
        json.loads(args.scenario.read_text()),
        args.output,
        checkpoint=json.loads(args.resume.read_text()) if args.resume else None,
        wall_limit_s=args.wall_limit_s,
    )
    print(
        json.dumps(
            dict(status=report["status"], numerical_acceptance=report["numerical_acceptance"])
        )
    )
    return 0 if report["status"] == "completed" else 2 if report["status"] == "domain_exit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
