"""Run from repository root: python -m examples.terminal_eos_chambers."""

import json

from scripts.terminal_chamber_references import request
from scripts.terminal_eos_chambers import TerminalEOSChambers


def main():
    assembly = TerminalEOSChambers(request(bias=1000, heat=500, ambient=True))
    records = [assembly.snapshot()]
    for _ in range(4):
        records.append(assembly.advance(0.05))
    print(
        json.dumps(dict(model="finite same-area terminal volumes", observations=records), indent=2)
    )


if __name__ == "__main__":
    main()
