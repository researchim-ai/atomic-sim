"""Reproducible scoped experimental comparison, with explicit screening failures."""

import argparse
from pathlib import Path

from atomic_model_validation import pipe


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("run-pipe")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("verify")
    a.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    try:
        r = pipe.run(a.output) if a.command == "run-pipe" else pipe.verify(a.folder)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(
        dict(
            samples=r["samples"],
            passed=r["passed"],
            max_abs_relative_error=r["max_abs_relative_error"],
            validated=r["validated"],
            plant_referenced=r["plant_referenced"],
        )
    )
    return 0 if r["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
