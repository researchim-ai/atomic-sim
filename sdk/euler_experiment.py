"""Reproducible conservative Euler liquid-channel experiments."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml

from core.status import DomainError
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from sdk.channel_experiment import _run_channel_experiment
from sdk.experiment import _number


def validate_euler_experiment(spec):
    if not isinstance(spec, dict) or set(spec) - {
        "schema_version",
        "engine",
        "name",
        "duration_s",
        "step_s",
        "channel",
        "solver",
        "budgets",
        "timeline",
    }:
        raise ValueError("Unknown Euler experiment fields")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        raise ValueError("Euler input schema must be 1")
    if spec.get("engine") not in (
        "conservative-euler-liquid-channel",
        "contact-euler-liquid-channel",
    ):
        raise ValueError("Explicit conservative Euler engine required")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("Name required")
    if spec.get("timeline", []) != []:
        raise ValueError("Euler timeline not supported")
    raw = spec.get("channel")
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(EulerConfig)}:
        raise ValueError("Unknown Euler channel fields")
    config = EulerConfig(**raw)
    if (spec["engine"] == "contact-euler-liquid-channel") != (config.numerical_flux == "hllc"):
        raise ValueError("Engine must explicitly match the Euler flux")
    solver = spec.get("solver", {})
    if not isinstance(solver, dict) or set(solver) - {
        "method",
        "rtol",
        "max_step_s",
        "jacobian_mode",
        "atol_scale",
    }:
        raise ValueError("Unknown Euler solver fields")
    if solver.get("method", "Radau") not in ("BDF", "Radau") or solver.get(
        "jacobian_mode", "analytic"
    ) not in ("analytic", "finite_difference"):
        raise ValueError("Unsupported Euler solver policy")
    budgets = spec.get("budgets", {})
    defaults = {
        "mass_relative": 1e-6,
        "energy_relative": 1e-5,
        "energy_absolute_j": 1e-3,
        "momentum_absolute_kg_m_s": 1e-7,
    }
    if not isinstance(budgets, dict) or set(budgets) - set(defaults):
        raise ValueError("Unknown Euler budgets")
    result = dict(
        schema_version=1,
        engine=spec["engine"],
        name=spec["name"],
        timeline=[],
        duration_s=_number(spec.get("duration_s"), "duration_s", 1e-9, 86400.0),
        step_s=_number(spec.get("step_s", 0.1), "step_s", 1e-6, 5.0),
        channel=config.as_dict(),
        solver=dict(
            method=solver.get("method", "Radau"),
            jacobian_mode=solver.get("jacobian_mode", "analytic"),
            rtol=_number(solver.get("rtol", 1e-7), "rtol", 1e-12, 1e-2),
            max_step_s=_number(solver.get("max_step_s", 0.01), "max_step_s", 1e-8, 5.0),
        ),
        budgets={
            k: _number(budgets.get(k, v), k, 0, 1e12 if "absolute" in k else 1)
            for k, v in defaults.items()
        },
    )
    if "atol_scale" in solver:
        result["solver"]["atol_scale"] = _number(solver["atol_scale"], "atol_scale", 1e-6, 1.0)
    return result


def run_euler_experiment(spec, output, **options):
    normalized = validate_euler_experiment(spec)
    return _run_channel_experiment(
        normalized, EulerLiquidChannel(EulerConfig(**normalized["channel"])), output, **options
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--wall-limit-s", type=float)
    p.add_argument("--cancel-file", type=Path)
    args = p.parse_args(argv)
    try:
        spec = (
            json.loads(args.scenario.read_text())
            if args.scenario.suffix.lower() == ".json"
            else yaml.safe_load(args.scenario.read_text())
        )
        report = run_euler_experiment(
            spec, args.output, wall_limit_s=args.wall_limit_s, cancel_file=args.cancel_file
        )
    except (ValueError, DomainError, OSError, yaml.YAMLError) as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {key: report[key] for key in ("status", "completed_time_s", "numerical_acceptance")}
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
