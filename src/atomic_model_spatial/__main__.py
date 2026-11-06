"""Numerical qualification of synthetic-data 1D multigroup diffusion."""

import argparse
import json
from pathlib import Path

from atomic_model_hem_throughflow.__main__ import publish
from atomic_model_workspace.common import digest

from .workflow import audit, qualify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("qualify").add_argument("--output", type=Path, required=True)
    sub.add_parser("audit").add_argument("input", type=Path)
    run = sub.add_parser("run")
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    example = sub.add_parser("example")
    example.add_argument("name", choices=["slab_eigen_v1", "slab_transient_v1"])
    example.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "example":
            from importlib.resources import files

            if args.output.exists():
                raise FileExistsError(args.output)
            data = files("atomic_model_spatial").joinpath("data", args.name + ".json").read_text()
            # Atomic publication uses the same JSON value without adding an envelope.
            publish(args.output, json.loads(data))
            print(str(args.output))
            return 0
        if args.command in ("qualify", "run"):
            if args.output.exists():
                raise FileExistsError(args.output)
            if args.command == "qualify":
                result = qualify()
            else:
                from .experiment import execute

                request = json.loads(args.input.read_text())
                saved = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
                if saved is not None and "result" in saved.get("payload", {}):
                    if (
                        set(saved) != {"payload", "sha256"}
                        or digest(saved["payload"]) != saved["sha256"]
                    ):
                        raise ValueError("Invalid saved run envelope")
                    saved = saved["payload"]["result"]["checkpoint"]
                result = execute(request, saved)
            publish(args.output, dict(payload=result, sha256=digest(result)))
        else:
            result = audit(args.input)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(dict(status=result.get("status", "completed"), validated=False, plant_referenced=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
