"""Source-bound active-chart HEM demo, replay and numerical qualification."""

import argparse
from pathlib import Path

from atomic_model_hem_qualified.cases import programs
from atomic_model_hem_qualified.model import signature
from atomic_model_hem_qualified.qualification import execute, qualify
from atomic_model_workspace.common import atomic_json, digest, read


def demo():
    name, request, calls = programs()[0]
    observations, checkpoints = execute(request, calls)
    return dict(
        schema_version=1,
        source=signature(),
        name=name,
        request=request,
        calls=calls,
        observations=observations,
        checkpoint_sha256=checkpoints,
        validated=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "qualify"):
        sub.add_parser(name).add_argument("--output", type=Path, required=True)
    for name in ("replay", "audit"):
        sub.add_parser(name).add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ("demo", "qualify"):
            if args.output.exists():
                raise FileExistsError(args.output)
            result = demo() if args.command == "demo" else qualify()
            atomic_json(args.output, dict(payload=result, sha256=digest(result)))
        else:
            record = read(args.input)
            result = record["payload"]
            if record["sha256"] != digest(result) or result["source"] != signature():
                raise ValueError("Evidence digest/source mismatch")
            if args.command == "audit":
                replay = qualify()
            else:
                observations, checkpoints = execute(result["request"], result["calls"])
                replay = dict(result, observations=observations, checkpoint_sha256=checkpoints)
            if digest(replay) != digest(result):
                raise ValueError("Evidence replay mismatch")
        print(dict(command=args.command, cases=len(result.get("cases", [])), validated=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
