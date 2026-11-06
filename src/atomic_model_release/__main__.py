"""Executable release runbook and honest AI use-case entrypoints."""

import argparse
from pathlib import Path

from atomic_model_release import lanes
from atomic_model_workspace.common import canonical


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("golden")
    sub.add_parser("use-cases")
    a = sub.add_parser("doctor")
    a.add_argument("--folder", type=Path)
    a = sub.add_parser("control-demo")
    a.add_argument("--output", required=True, type=Path)
    for name in ("diagnostic-demo", "data-demo"):
        demo = sub.add_parser(name)
        demo.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("migrate-project")
    a.add_argument("project", type=Path)
    a.add_argument("--output", required=True, type=Path)
    for name in ("diagnose", "dataset-summary"):
        a = sub.add_parser(name)
        a.add_argument("folder", type=Path)
    a = sub.add_parser("preflight")
    a.add_argument("lane", choices=lanes.LANES)
    a = p.parse_args(argv)
    try:
        if a.command not in ("doctor", "preflight", "use-cases"):
            from atomic_model_release import checks
        if a.command == "use-cases":
            from atomic_model_workspace.common import read

            result = read(Path(__file__).parent / "use_cases.json")
        elif a.command == "diagnostic-demo":
            result = checks.diagnostic_demo(a.output)
        elif a.command == "data-demo":
            result = checks.data_demo(a.output)
        elif a.command == "golden":
            result = checks.golden()
        elif a.command == "doctor":
            from atomic_model_release.environment import doctor

            result = doctor(a.folder)
        elif a.command == "control-demo":
            result = checks.control_demo(a.output)
        elif a.command == "migrate-project":
            result = checks.migrate_project(a.project, a.output)
        elif a.command == "diagnose":
            result = checks.diagnose(a.folder)
        elif a.command == "dataset-summary":
            result = checks.dataset_summary(a.folder)
        else:
            result = lanes.preflight(a.lane)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
