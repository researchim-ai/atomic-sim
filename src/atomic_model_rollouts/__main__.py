"""Durable bounded EOS rollout batches with explicit pause/cancel/CAS resume."""

import argparse
from pathlib import Path

from atomic_model_rollouts import reports, store
from atomic_model_rollouts.controller import Controller, recover, request
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("create")
    a.add_argument("project", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--config", type=Path)
    for name in (
        "inspect",
        "run",
        "resume",
        "recover",
        "pause",
        "cancel",
        "report",
        "export",
        "verify",
    ):
        a = sub.add_parser(name)
        a.add_argument("folder", type=Path)
        if name in ("resume", "recover"):
            a.add_argument("--expected-sha256", required=True)
        if name in ("run", "resume"):
            a.add_argument("--wall-s", type=float)
        if name == "export":
            a.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.command == "create":
            result = store.create(a.project, a.output, read(a.config) if a.config else None)
        elif a.command == "inspect":
            result = store.load(a.folder)
        elif a.command in ("run", "resume"):
            result = Controller(a.folder).run(
                resume=a.command == "resume",
                expected_sha256=getattr(a, "expected_sha256", None),
                wall_s=a.wall_s,
            )
        elif a.command == "recover":
            result = recover(a.folder, expected_sha256=a.expected_sha256)
        elif a.command in ("pause", "cancel"):
            result = request(a.folder, a.command)
        elif a.command == "report":
            result = reports.report(a.folder)
        elif a.command == "export":
            result = reports.export(a.folder, a.output)
        else:
            result = reports.verify(a.folder)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        p.error(str(exc))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
