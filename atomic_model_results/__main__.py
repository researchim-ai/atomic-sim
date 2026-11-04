"""Portable native-replayed study result packages and fair quality comparisons."""

import argparse
from pathlib import Path

from atomic_model_results import archive, bundle
from atomic_model_workspace.common import canonical


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("export")
    a.add_argument("study", type=Path)
    a.add_argument("evaluations", nargs="+")
    a.add_argument("--output", type=Path, required=True)
    for name in ["verify", "replay", "pair", "pack", "unpack"]:
        a = sub.add_parser(name)
        a.add_argument("bundle", type=Path)
        if name in ("replay", "pack", "unpack"):
            a.add_argument("--output", type=Path, required=True)
        if name == "pair":
            a.add_argument("left")
            a.add_argument("right")
            a.add_argument("channel")
    args = p.parse_args(argv)
    try:
        if args.command == "export":
            result = bundle.export(args.study, args.evaluations, args.output)
        elif args.command in ("pack", "unpack"):
            result = getattr(archive, args.command)(args.bundle, args.output)
        elif args.command == "verify":
            result = bundle.verify(args.bundle)
        elif args.command == "replay":
            result = bundle.replay(args.bundle, args.output)
        else:
            result = bundle.pair(args.bundle, args.left, args.right, args.channel)
    except (ValueError, OSError, KeyError, TypeError, archive.zipfile.BadZipFile) as exc:
        p.error(str(exc))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
