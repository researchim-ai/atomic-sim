"""Auditable hybrid check-valve pump and storage experiments."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.network import NetworkError, number
from physics.hydraulics.check_pump_storage import (
    CONFIG_FIELDS,
    CONTROL_FIELDS,
    PASSPORT,
    CheckPumpStorage,
)
from physics.hydraulics.pump_storage_pair import PASSPORT as HYDRAULIC_PASSPORT
from sdk.experiment import _source_fingerprint
from sdk.network_experiment import write
from sdk.pump_storage_experiment import prepare as prepare_hydraulics


def prepare(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "model",
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
        or not isinstance(spec["model"], dict)
        or set(spec["model"]) != CONFIG_FIELDS
    ):
        raise NetworkError("Exact coupled pump-storage schema 1 required")
    probe = deepcopy(spec)
    probe.update(
        engine=HYDRAULIC_PASSPORT.model_id,
        model=spec["model"]["plant_configuration"],
        timeline=[],
    )
    normalized, _ = prepare_hydraulics(probe)
    model = CheckPumpStorage(**spec["model"])
    if not isinstance(spec["timeline"], list) or len(spec["timeline"]) > 10000:
        raise NetworkError("Control timeline list required")
    commands = []
    previous = -1.0
    for c in spec["timeline"]:
        if (
            not isinstance(c, dict)
            or set(c) != {"t", "controls"}
            or not isinstance(c["controls"], dict)
            or set(c["controls"]) != CONTROL_FIELDS
        ):
            raise NetworkError("Complete atomic control vector required")
        stamp = number(c["t"], "command time")
        values = {k: number(v, k) for k, v in c["controls"].items()}
        if (
            not previous < stamp <= normalized["duration_s"]
            or stamp < 0
            or values["resistance_pa_s2_kg2"] < 0
        ):
            raise NetworkError("Invalid pump-storage control time/resistance")
        commands.append(dict(t=stamp, controls=values))
        previous = stamp
    result = deepcopy(spec)
    result.update(
        model=model.configuration,
        duration_s=normalized["duration_s"],
        step_s=normalized["step_s"],
        solver=normalized["solver"],
        timeline=commands,
    )
    return result, model


def run_check_storage_experiment(spec, output, *, wall_limit_s=None):
    normalized, model = prepare(spec)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    output = Path(output)
    if output.exists():
        raise NetworkError("Pump-storage output must be fresh")
    output.mkdir(parents=True)
    write(output / "input.json", normalized)
    initial = model.observation()
    rate = model.derivatives(0, model._state)
    write(
        output / "initialization.json",
        dict(
            mode="explicit-inventories-flow-and-shaft",
            observation=initial,
            initial_momentum_rate_pa=float(model._plant._hydraulic._inertia * rate[8]),
            initial_shaft_net_torque_nm=float(
                model.configuration["plant_configuration"]["pump"]["inertia_kg_m2"] * rate[9]
            ),
        ),
    )
    source = _source_fingerprint()
    started = time.monotonic()
    original = model.derivatives

    def guarded(t, y, mode=None):
        if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
            raise NetworkError("Pump-storage wall deadline exceeded")
        return original(t, y, mode)

    model.derivatives = guarded
    rows = [dict(phase="initial", **initial)]
    index = 0
    status = "completed"
    error = None
    horizon, step = normalized["duration_s"], normalized["step_s"]
    samples = {min(horizon, k * step) for k in range(1, math.ceil(horizon / step) + 1)}
    grid = sorted(samples | {c["t"] for c in normalized["timeline"] if c["t"] > 0})

    def commands():
        nonlocal index
        while (
            index < len(normalized["timeline"])
            and normalized["timeline"][index]["t"] == model.time_s
        ):
            o = model.set_controls(normalized["timeline"][index]["controls"])
            rows.append(dict(phase="controls_changed", **o))
            index += 1

    try:
        commands()
        for target in grid:
            o = model.advance(target - model.time_s, **normalized["solver"])
            rows.append(dict(phase="sample" if target in samples else "control_boundary", **o))
            commands()
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        status = "failed"
        error = dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status = "failed"
        error = dict(type="SourceChanged", message="Pump-storage runtime changed during experiment")
    report = dict(
        schema_version=1,
        name=normalized["name"],
        status=status,
        numerical_acceptance=status == "completed",
        requested_duration_s=horizon,
        completed_time_s=model.time_s,
        accepted_commands=index,
        energy_absolute_j=max(abs(x["energy_residual_j"]) for x in rows),
        shaft_energy_absolute_j=max(abs(x["shaft_energy_residual_j"]) for x in rows),
        local_energy_absolute_w=max(
            abs(v) for x in rows for v in x["local_energy_residual_w"].values()
        ),
        link_energy_absolute_w=max(abs(x["link_energy_residual_w"]) for x in rows),
        events=deepcopy(model.events),
        error=error,
        wall_time_s=time.monotonic() - started,
        validated=False,
        plant_referenced=False,
        scope="Hybrid ideal check valve, pump/shaft and gas storage; no flap impact/water hammer/steam pressurizer/plant qualification",
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        dict(
            schema_version=1,
            source_sha256=source,
            configuration_sha256=model.configuration_sha256(),
            passport=asdict(PASSPORT),
            versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy")},
            artifacts={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
                if p.is_file()
            },
        ),
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wall-limit-s", type=float)
    args = p.parse_args(argv)
    r = run_check_storage_experiment(
        json.loads(args.scenario.read_text()), args.output, wall_limit_s=args.wall_limit_s
    )
    print(json.dumps(dict(status=r["status"], numerical_acceptance=r["numerical_acceptance"])))
    return 0 if r["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
