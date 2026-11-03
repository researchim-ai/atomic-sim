"""Run from repository root: python -m examples.controlled_eos_assembly."""

import json

from scripts.controlled_assembly_references import request
from scripts.controlled_eos_assembly import ControlledEOSAssembly


def main():
    assembly = ControlledEOSAssembly(request(ambient=True))
    before = assembly.advance(0.05)
    powers = before["active_heat_power_w"].copy()
    powers["primary.chamber.right"] = 1000.0
    powers["secondary.channel.c2"] = -200.0
    powers["wall.w0"] = 500.0
    after = assembly.step(
        powers, 0.05, expected_revision=before["revision"], command_id="online.heat"
    )
    print(json.dumps(dict(before=before, after=after), indent=2))


if __name__ == "__main__":
    main()
