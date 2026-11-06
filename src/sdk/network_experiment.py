"""Reproducible experiments for the compiled constant-coefficient thermal network."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from components.linear_thermal_network import LinearThermalNetwork
from core.network import NetworkError, number
from sdk.experiment import _source_fingerprint


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def validate_network_experiment(spec):
    fields = {
        "schema_version",
        "engine",
        "name",
        "topology",
        "initial_temperature_k",
        "duration_s",
        "step_s",
        "energy_tolerance_j",
    }
    if (
        not isinstance(spec, dict)
        or set(spec) != fields
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise NetworkError("Exact network experiment schema 1 required")
    if (
        spec["engine"] != LinearThermalNetwork.model_id
        or not isinstance(spec["name"], str)
        or not spec["name"].strip()
    ):
        raise NetworkError("Explicit linear thermal engine and name required")
    duration = number(spec["duration_s"], "duration_s", positive=True)
    step = number(spec["step_s"], "step_s", positive=True)
    if duration > 86400 or duration / step > 100000:
        raise NetworkError("Experiment resource limits exceeded")
    model = LinearThermalNetwork(
        spec["topology"],
        spec["initial_temperature_k"],
        energy_tolerance_j=spec["energy_tolerance_j"],
    )
    return dict(
        schema_version=1,
        engine=model.model_id,
        name=spec["name"],
        topology=model.compiled.specification(),
        initial_temperature_k=model.temperature_k,
        duration_s=duration,
        step_s=step,
        energy_tolerance_j=model.energy_tolerance_j,
    )


def run_network_experiment(spec, output, *, wall_limit_s=None):
    normalized = validate_network_experiment(spec)
    if wall_limit_s is not None:
        wall_limit_s = number(wall_limit_s, "wall_limit_s", positive=True)
    model = LinearThermalNetwork(
        normalized["topology"],
        normalized["initial_temperature_k"],
        energy_tolerance_j=normalized["energy_tolerance_j"],
    )
    output = Path(output)
    output.mkdir(parents=True)
    write(output / "input.json", normalized)
    write(output / "assembly.json", model.compiled.describe())
    started = time.monotonic()
    source = _source_fingerprint()
    rows = [model.observation()]
    status = "completed"
    error = None
    steps = 0
    try:
        while model.time_s < normalized["duration_s"]:
            if wall_limit_s is not None and time.monotonic() - started > wall_limit_s:
                raise NetworkError("Network experiment wall deadline exceeded")
            end = min(normalized["duration_s"], (steps + 1) * normalized["step_s"])
            if end <= model.time_s:
                raise NetworkError("Output clock cannot advance at requested precision")
            rows.append(model.advance(end - model.time_s))
            steps += 1
    except (NetworkError, ValueError, RuntimeError, ArithmeticError) as exc:
        status = "failed"
        error = dict(type=type(exc).__name__, message=str(exc))
    if source != _source_fingerprint():
        status = "failed"
        error = dict(type="SourceChanged", message="Runtime changed during experiment")
    max_energy = max(abs(r["energy_residual_j"]) for r in rows)
    report = dict(
        schema_version=1,
        name=normalized["name"],
        status=status,
        requested_duration_s=normalized["duration_s"],
        completed_time_s=model.time_s,
        accepted_output_steps=steps,
        energy_absolute_j=max_energy,
        numerical_acceptance=status == "completed" and max_energy <= model.energy_tolerance_j,
        wall_time_s=time.monotonic() - started,
        error=error,
        validated=False,
        plant_referenced=False,
        scope="Closed constant-coefficient thermal network; no hydraulic/equipment/whole-plant qualification",
    )
    write(output / "trajectory.json", rows)
    write(output / "checkpoint.json", model.snapshot())
    write(output / "report.json", report)
    manifest = dict(
        schema_version=1,
        source_sha256=source,
        passport=dict(model_id=model.model_id, version=model.model_version, status="implemented"),
        topology_sha256=model.compiled.topology_sha256,
        assembly_sha256=model.compiled.assembly_sha256,
        versions={k: importlib.metadata.version(k) for k in ("numpy", "scipy")},
        python=platform.python_version(),
        artifacts={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    )
    write(output / "manifest.json", manifest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float)
    args = parser.parse_args(argv)
    spec = json.loads(args.scenario.read_text(encoding="utf-8"))
    report = run_network_experiment(spec, args.output, wall_limit_s=args.wall_limit_s)
    print(
        json.dumps(
            dict(status=report["status"], numerical_acceptance=report["numerical_acceptance"])
        )
    )
    return 0 if report["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
