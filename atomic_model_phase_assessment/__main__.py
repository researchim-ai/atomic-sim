"""Import and assess primary transient pool boiling without mixing clock bases."""

import argparse
import json
from pathlib import Path

from atomic_model_hem_throughflow.__main__ import publish
from atomic_model_workspace.common import digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("import-pool")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    sub.add_parser("qualify-vessels").add_argument("--output", type=Path, required=True)
    sub.add_parser("audit-vessels").add_argument("input", type=Path)
    sub.add_parser("screen").add_argument("--output", type=Path, required=True)
    sub.add_parser("audit").add_argument("input", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "import-pool":
            from .importer import import_primary

            result = import_primary(args.input, args.output)
            print(
                dict(
                    native_counts=result["native_counts"], retained_counts=result["retained_counts"]
                )
            )
            return 0
        if args.command in ("qualify-vessels", "audit-vessels"):
            from .vessel import qualify

            if args.command == "qualify-vessels":
                if args.output.exists():
                    raise FileExistsError(args.output)
                result = qualify()
                publish(args.output, dict(payload=result, sha256=digest(result)))
            else:
                saved = json.loads(args.input.read_text())
                if (
                    set(saved) != {"payload", "sha256"}
                    or digest(saved["payload"]) != saved["sha256"]
                ):
                    raise ValueError("Invalid vessel qualification envelope")
                result = qualify()
                if digest(result) != saved["sha256"]:
                    raise ValueError("Vessel qualification replay differs")
            print(
                dict(
                    cases=len(result["cases"]),
                    trajectories=sum(len(c["runs"]) for c in result["cases"]),
                    validated=False,
                )
            )
            return 0
        from .assessment import evaluate

        if args.command == "screen":
            if args.output.exists():
                raise FileExistsError(args.output)
            result = evaluate()
            publish(args.output, dict(payload=result, sha256=digest(result)))
        else:
            saved = json.loads(args.input.read_text())
            if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
                raise ValueError("Invalid phase-data envelope")
            result = evaluate()
            if digest(result) != saved["sha256"]:
                raise ValueError("Phase-data assessment changed")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    print(
        dict(
            case=result["primary_case"],
            admitted=result["production_forced_boiling_admitted_rows"],
            validated=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
