"""Run from repository root: python -m examples.open_eos_channel."""

import json

from scripts.open_channel_references import request
from scripts.open_eos_channel import OpenEOSChannel


def main():
    channel = OpenEOSChannel(request(velocity=2, bias=-10000), max_step_s=0.0005)
    records = [channel.snapshot()["state"]]
    for _ in range(4):
        records.append(channel.advance(0.05)["state"])
    print(json.dumps(dict(contract=channel.system.contract(), observations=records), indent=2))


if __name__ == "__main__":
    main()
