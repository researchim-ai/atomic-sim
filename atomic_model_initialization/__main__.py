"""Certified graph initialization, audit and numerical qualification."""

import argparse
import json
from pathlib import Path

from atomic_model_hem_throughflow.__main__ import publish
from atomic_model_workspace.common import digest

from .qualification import qualify
from .steady import audit_certificate, identity, initialize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("initialize")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--flow-guess", type=float, default=0.13)
    a.add_argument("--temperature-guess", type=float, default=310.0)
    sub.add_parser("audit").add_argument("input", type=Path)
    sub.add_parser("qualify").add_argument("--output", type=Path, required=True)
    sub.add_parser("audit-qualification").add_argument("input", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command in ("initialize", "qualify"):
            if args.output.exists():
                raise FileExistsError(args.output)
            if args.command == "initialize":
                saved = initialize(
                    json.loads(args.input.read_text()),
                    flow_guess_kg_s=args.flow_guess,
                    temperature_guess_k=args.temperature_guess,
                )
            else:
                result = qualify()
                saved = dict(payload=result, sha256=digest(result))
            publish(args.output, saved)
        elif args.command == "audit":
            audit_certificate(json.loads(args.input.read_text()))
        else:
            saved = json.loads(args.input.read_text())
            if (
                set(saved) != {"payload", "sha256"}
                or digest(saved["payload"]) != saved["sha256"]
                or saved["payload"]["source"] != identity()
            ):
                raise ValueError("Qualification identity mismatch")
            fresh = qualify()
            if digest(fresh) != saved["sha256"]:
                raise ValueError("Qualification replay differs")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    print(dict(status="numerically_verified", validated=False, plant_referenced=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
