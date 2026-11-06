"""Run one bounded, licensed, real OpenMC replica and retain all evidence."""

import argparse
import json

from atomic_model_multiphysics.nuclear import specification
from atomic_model_multiphysics.transport import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    result = run(specification(seed=args.seed), args.directory)
    print(json.dumps(result))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
