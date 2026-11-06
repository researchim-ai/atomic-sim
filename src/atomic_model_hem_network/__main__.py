"""Finite-plenum HEM graph run/replay/qualification CLI."""

import argparse
from pathlib import Path

from atomic_model_hem_network import workflow
from atomic_model_workspace.common import atomic_json, read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["demo", "qualify"]:
        a = sub.add_parser(name)
        a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("run")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("replay")
    a.add_argument("folder", type=Path)
    a = sub.add_parser("audit")
    a.add_argument("campaign", type=Path)
    a = p.parse_args(argv)
    try:
        if a.command in ["qualify", "audit"]:
            result = workflow.qualify() if a.command == "qualify" else workflow.audit(a.campaign)
            if a.command == "qualify":
                if a.output.exists():
                    raise FileExistsError(a.output)
                atomic_json(a.output, result)
            print(
                {
                    key: result[key]
                    for key in [
                        "case_count",
                        "temporal_trajectories",
                        "semidiscrete_reference_ivps",
                        "validated",
                    ]
                }
            )
            return 0
        if a.command == "run":
            result = workflow.run(read(a.input), a.output)
        elif a.command == "demo":
            result = workflow.demo(a.output)
        else:
            result = workflow.replay(a.folder)
    except (ValueError, OSError, TypeError, KeyError) as e:
        p.error(str(e))
    final = result["observations"][-1]
    print(
        dict(
            time_s=final["time_s"],
            phases=[c["phase"] for c in final["cells"]],
            ledger=final["global_ledger"],
            validated=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
