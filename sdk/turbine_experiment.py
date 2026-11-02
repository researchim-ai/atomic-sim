"""Steam turbine/generator jobs with explicit physical-boundary termination."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from core.network import NetworkError, number
from physics.power.steam_turbine_train import PASSPORT, SteamTurbineTrain, controls
from sdk.boundary_job import run_boundary_job


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
        "controls",
        "initial_omega_rad_s",
        "duration_s",
        "step_s",
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
        raise NetworkError("Exact named turbine experiment schema 1 required")
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 10000:
        raise NetworkError("Turbine output resource limits exceeded")
    model = SteamTurbineTrain(spec["model"], spec["controls"], spec["initial_omega_rad_s"])
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Turbine command resource limit exceeded")
    commands, previous = [], -1.0
    for event in spec["timeline"]:
        if not isinstance(event, dict) or set(event) != {"t", "controls"}:
            raise NetworkError("Exact turbine command required")
        stamp = number(event["t"], "command time")
        if not previous < stamp <= duration or stamp < 0:
            raise NetworkError("Turbine commands require increasing in-range times")
        commands.append(dict(t=stamp, controls=controls(event["controls"])))
        previous = stamp
    normalized = deepcopy(spec)
    normalized.update(
        model=model.configuration,
        controls=model.boundary_controls,
        initial_omega_rad_s=model._omega0,
        duration_s=duration,
        step_s=step,
        timeline=commands,
    )
    return normalized, model


def restore(spec, payload):
    normalized, fresh = prepare(spec)
    model = SteamTurbineTrain.from_snapshot(payload)
    if (
        model.configuration != fresh.configuration
        or model._omega0 != fresh._omega0
        or model._initial_controls != fresh._initial_controls
        or model.time_s > normalized["duration_s"]
    ):
        raise NetworkError("Turbine restart differs from scenario configuration/horizon")
    actual = [dict(t=e["t"], controls=e["after"]) for e in model.events if e["kind"] == "controls"]
    expected = [e for e in normalized["timeline"] if e["t"] <= model.time_s]
    if actual != expected[: len(actual)] or len(actual) < len(
        [e for e in expected if e["t"] < model.time_s]
    ):
        raise NetworkError("Turbine restart differs from executed control prefix")
    return normalized, model


def run_turbine_experiment(spec, output, *, checkpoint=None, wall_limit_s=None):
    return run_boundary_job(
        spec,
        output,
        prepare=prepare,
        restore=restore,
        model_class=SteamTurbineTrain,
        passport=PASSPORT,
        initial_artifacts=lambda model: {"thermodynamics.json": model.thermodynamics},
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
    report = run_turbine_experiment(
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
