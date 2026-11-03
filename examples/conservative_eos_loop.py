"""Run from the repository root: python -m examples.conservative_eos_loop."""

import json

from scripts.conservative_eos_loop import ConservativeEOSLoop
from scripts.conservative_loop_references import request


def main():
    engine = ConservativeEOSLoop(request(height=10, gravity=9.81, heat=500, drag=0.1))
    engine.advance(0.2)
    print(json.dumps(engine.snapshot(), indent=2))


if __name__ == "__main__":
    main()
