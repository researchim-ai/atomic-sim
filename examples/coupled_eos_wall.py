"""Run from repository root: python -m examples.coupled_eos_wall."""

import json

from scripts.coupled_eos_wall import CoupledEOSWall
from scripts.eos_wall_references import request


def main():
    assembly = CoupledEOSWall(request(gravity=True, ambient=True))
    assembly.advance(0.2)
    print(json.dumps(assembly.snapshot(), indent=2))


if __name__ == "__main__":
    main()
