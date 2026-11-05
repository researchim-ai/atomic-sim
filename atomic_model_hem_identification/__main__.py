"""Source-bound conditional identifiability; rejected calibration is preserved."""

import argparse
from pathlib import Path

from atomic_model_hem_throughflow.__main__ import publish
from atomic_model_workspace.common import digest

from . import qualification, workflow


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--require-identifiable", action="store_true")
    a.add_argument("--require-calibratable", action="store_true")
    sub.add_parser("audit").add_argument("input", type=Path)
    sub.add_parser("qualify").add_argument("--output", type=Path, required=True)
    sub.add_parser("audit-qualification").add_argument("input", type=Path)
    args = p.parse_args(argv)
    try:
        if args.command in ("analyse", "qualify"):
            if args.output.exists():
                raise FileExistsError(args.output)
            result = workflow.evaluate() if args.command == "analyse" else qualification.qualify()
            publish(args.output, dict(payload=result, sha256=digest(result)))
        else:
            result = (
                workflow.audit(args.input)
                if args.command == "audit"
                else qualification.audit(args.input)
            )
        if args.command in ("qualify", "audit-qualification"):
            print(dict(primary_states_checked=result["primary_states_checked"], validated=False))
            return 0
        print(
            dict(
                calibration_status=result["calibration_status"],
                development_rows=result["development"]["observations"],
                evaluation_rows=result["evaluation"]["observations"],
                parameter_update=None,
                validated=False,
            )
        )
        if (
            args.command == "analyse"
            and args.require_calibratable
            and result["calibration_status"] != "admitted"
        ):
            return 3
        if (
            args.command == "analyse"
            and args.require_identifiable
            and not result["development"]["conditional_identifiable_at_declared_resolution"]
        ):
            return 3
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
