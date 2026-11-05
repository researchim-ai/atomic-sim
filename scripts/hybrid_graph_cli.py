"""Run and replay declarative hybrid graph programs with detached native checkpoints."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from atomic_model_workspace.common import atomic_json, local_file, read
from core.network import NetworkError
from scripts.hybrid_thermal_graph import HybridThermalGraph
from scripts.hybrid_thermal_graph import signature as model_signature
from scripts.solve_hydraulic_graph import _keys, sha


def signature():
    return dict(model=model_signature(), cli_sha256=sha(__file__))


def execute(spec):
    _keys(spec, ("schema_version", "model", "max_macro_step_s", "program", "initial_checkpoint"))
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise NetworkError("Hybrid program schema mismatch")
    program = spec["program"]
    if not isinstance(program, list) or not 1 <= len(program) <= 32:
        raise NetworkError("Hybrid program requires 1..32 calls")
    captured = signature()
    model = HybridThermalGraph(spec["model"], max_macro_step_s=spec["max_macro_step_s"])
    if spec["initial_checkpoint"] is not None:
        model.restore(spec["initial_checkpoint"])
    observations = []
    size = 0
    for call in program:
        _keys(call, ("duration_s", "command"))
        command = call["command"]
        if command is None:
            model.advance(call["duration_s"])
        else:
            _keys(command, ("mode", "expected_revision", "controller_state", "priority"))
            model.step(
                command["mode"],
                call["duration_s"],
                expected_revision=command["expected_revision"],
                controller_state=command["controller_state"],
                priority=command["priority"],
            )
        observation = model.snapshot()
        size += len(json.dumps(observation, allow_nan=False).encode())
        if size > 16 * 1024**2:
            raise NetworkError("Hybrid report observation budget exceeded")
        observations.append(observation)
    if signature() != captured:
        raise NetworkError("Hybrid program source drift")
    return dict(
        schema_version=1,
        source=captured,
        observations=observations,
        final_checkpoint=model.checkpoint(),
        validated=False,
        plant_referenced=False,
    )


def run(spec, folder):
    folder = Path(folder)
    result = execute(spec)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(folder / "input.json", spec)
        atomic_json(folder / "result.json", result)
        if any(
            (folder / name).stat().st_size > 32 * 1024**2 for name in ("input.json", "result.json")
        ):
            raise NetworkError("Hybrid result file budget exceeded")
        if signature() != result["source"]:
            raise NetworkError("Hybrid result publication source drift")
        atomic_json(
            folder / "manifest.json",
            dict(
                schema_version=1,
                source=signature(),
                artifacts={name: sha(folder / name) for name in ["input.json", "result.json"]},
            ),
        )
    except BaseException:
        shutil.rmtree(folder)
        raise
    return result


def replay(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    _keys(manifest, ("schema_version", "source", "artifacts"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source"] != signature()
        or set(manifest["artifacts"]) != {"input.json", "result.json"}
    ):
        raise NetworkError("Hybrid result source/schema/inventory mismatch")
    if {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()
    } != {"input.json", "result.json", "manifest.json"}:
        raise NetworkError("Unexpected hybrid result files")
    for name, value in manifest["artifacts"].items():
        path = local_file(folder, name)
        if path.stat().st_size > 32 * 1024**2 or sha(path) != value:
            raise NetworkError("Hybrid result hash/size mismatch")
    result = execute(read(local_file(folder, "input.json")))
    if result != read(local_file(folder, "result.json")):
        raise NetworkError("Hybrid result full replay differs")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("run")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("replay")
    a.add_argument("folder", type=Path)
    a = sub.add_parser("continue")
    a.add_argument("folder", type=Path)
    a.add_argument("--program", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != root:
        # Legacy graph resources are relative to their distribution root. Run a
        # fixed child module instead of changing the parent process working dir.
        forwarded = [a.command]
        if a.command == "run":
            forwarded += [str(a.input.resolve()), "--output", str(a.output.resolve())]
        else:
            forwarded += [str(a.folder.resolve())]
            if a.command == "continue":
                forwarded += [
                    "--program",
                    str(a.program.resolve()),
                    "--output",
                    str(a.output.resolve()),
                ]
        return subprocess.run(
            [sys.executable, "-m", "scripts.hybrid_graph_cli", *forwarded], cwd=root
        ).returncode
    try:
        if a.command == "run":
            result = run(read(a.input), a.output)
        elif a.command == "replay":
            result = replay(a.folder)
        else:
            previous = replay(a.folder)
            spec = read(local_file(a.folder, "input.json"))
            spec["initial_checkpoint"] = previous["final_checkpoint"]
            spec["program"] = read(a.program)
            result = run(spec, a.output)
    except (ValueError, OSError, KeyError, TypeError) as e:
        p.error(str(e))
    final = result["observations"][-1]
    print(
        dict(
            time_s=final["time_s"],
            mode=final["mode"],
            revision=final["revision"],
            events=len(final["history"]),
            validated=False,
            plant_referenced=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
