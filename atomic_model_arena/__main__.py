"""Open EOS benchmark: frozen physics splits, train-only selection and audit."""

import argparse
import sys
from pathlib import Path

from atomic_model_arena import protocol, runner
from atomic_model_workspace.common import canonical, read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("validate")
    a.add_argument("protocol", type=Path)
    a = sub.add_parser("run")
    a.add_argument("protocol", type=Path)
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--commands", type=Path)
    a.add_argument("--cpu", type=int)
    a = sub.add_parser("verify")
    a.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    try:
        if a.command == "validate":
            value = protocol.normalize(read(a.protocol))
            result = dict(
                schema_version=1,
                protocol_sha256=protocol.digest(value),
                cases=len(value["cases"]),
                agents=len(value["agents"]),
            )
        elif a.command == "run":
            value = read(a.protocol)
            commands = (
                read(a.commands)
                if a.commands
                else {
                    id: [sys.executable, "-m", "atomic_model_agents.baseline"]
                    for id in value["agents"]
                }
            )
            result = runner.run(value, commands, a.output, cpu=a.cpu)
        else:
            result = runner.verify(a.folder)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
