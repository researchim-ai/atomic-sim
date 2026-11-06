"""Offline primary HIPER admission census, exact replay and reimport verification."""

import argparse
from pathlib import Path

from atomic_model_workspace.common import atomic_json, digest, read

from .audit import assess_target, audit, evaluate, verify_reimport


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("screen").add_argument("--output", type=Path, required=True)
    for name in ("audit", "reimport", "target"):
        sub.add_parser(name).add_argument("input", type=Path)
    a = p.parse_args(argv)
    try:
        if a.command == "screen":
            if a.output.exists():
                raise FileExistsError(a.output)
            result = evaluate()
            atomic_json(a.output, dict(payload=result, sha256=digest(result)))
        elif a.command == "audit":
            result = audit(a.input)
        elif a.command == "reimport":
            result = verify_reimport(a.input)
        else:
            result = assess_target(read(a.input))
        print(
            {
                k: v
                for k, v in result.items()
                if k not in ("source", "assessment", "ranges", "per_run", "acoustic_screen")
            }
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
