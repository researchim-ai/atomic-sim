"""Create, inspect, preview, atomically edit, restore and export EOS projects."""

import argparse
from pathlib import Path

from atomic_model_projects import store
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create")
    p.add_argument("project", type=Path)
    p.add_argument("--output", required=True, type=Path)
    for name in ("inspect", "preview", "edit", "restore", "export"):
        p = sub.add_parser(name)
        p.add_argument("store", type=Path)
        if name in ("preview", "edit"):
            p.add_argument("request", type=Path)
        if name == "export":
            p.add_argument("--output", required=True, type=Path)
        if name == "restore":
            p.add_argument("--expected-revision", type=int, required=True)
            p.add_argument("--expected-project-sha256", required=True)
            p.add_argument("--target-revision", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = store.create(args.project, args.output)
        elif args.command == "inspect":
            result = store.inspect(args.store)
        elif args.command == "preview":
            result = store.preview(args.store, read(args.request))
        elif args.command == "edit":
            result = store.edit(args.store, read(args.request))
        elif args.command == "restore":
            result = store.restore(
                args.store,
                expected_revision=args.expected_revision,
                expected_project_sha256=args.expected_project_sha256,
                target_revision=args.target_revision,
            )
        else:
            result = store.export(args.store, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
