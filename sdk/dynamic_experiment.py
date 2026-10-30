"""Reproducible single-phase pressure-driven channel experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from sdk.channel_experiment import _run_channel_experiment
from sdk.experiment import _number


def validate_dynamic_experiment(spec):
    allowed = {
        "schema_version",
        "engine",
        "name",
        "duration_s",
        "step_s",
        "hydraulics",
        "solver",
        "budgets",
        "timeline",
    }
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Unknown pressure-driven experiment fields")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("Expected schema_version 1")
    if spec.get("engine") != "pressure-driven-wall-channel":
        raise ValueError("Expected pressure-driven-wall-channel engine")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("Experiment name required")
    if not isinstance(spec.get("timeline", []), list) or spec.get("timeline", []):
        raise ValueError("Timeline commands are not supported by this passport")
    raw = spec.get("hydraulics")
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(DynamicConfig)}:
        raise ValueError("Invalid hydraulics configuration")
    try:
        config = DynamicConfig(**raw)
    except TypeError as exc:
        raise ValueError("Incomplete hydraulics configuration") from exc
    solver, budgets = spec.get("solver", {}), spec.get("budgets", {})
    if not isinstance(solver, dict) or set(solver) - {"method", "rtol", "max_step_s"}:
        raise ValueError("Unknown solver fields")
    if solver.get("method", "Radau") not in ("BDF", "Radau"):
        raise ValueError("Expected BDF or Radau")
    if not isinstance(budgets, dict) or set(budgets) - {"mass_relative", "energy_relative"}:
        raise ValueError("Unknown balance budgets")
    return dict(
        schema_version=1,
        engine=spec["engine"],
        name=spec["name"],
        duration_s=_number(spec.get("duration_s"), "duration_s", 1e-9, 86400.0),
        step_s=_number(spec.get("step_s", 1.0), "step_s", 1e-6, 5.0),
        timeline=[],
        hydraulics=config.as_dict(),
        solver=dict(
            method=solver.get("method", "Radau"),
            rtol=_number(solver.get("rtol", 1e-7), "rtol", 1e-12, 1e-2),
            max_step_s=_number(solver.get("max_step_s", 0.1), "max_step_s", 1e-6, 5.0),
        ),
        budgets={
            k: _number(budgets.get(k, v), k, 0.0, 1.0)
            for k, v in [("mass_relative", 1e-6), ("energy_relative", 1e-5)]
        },
    )


def run_dynamic_experiment(spec, output, **options):
    normalized = validate_dynamic_experiment(spec)
    channel = DynamicWallChannel(DynamicConfig(**normalized["hydraulics"]))
    return _run_channel_experiment(normalized, channel, output, **options)


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_dynamic_experiment(
            yaml.safe_load(args.scenario.read_text()),
            args.output,
            wall_limit_s=args.wall_limit_s,
            cancel_file=args.cancel_file,
        )
    except (ValueError, DomainError, OSError, yaml.YAMLError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps({k: report[k] for k in ("status", "completed_time_s", "numerical_acceptance")})
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
