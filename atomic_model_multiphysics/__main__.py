"""Execute bounded numerical qualification; transport results remain separate."""

import argparse
import json

from atomic_model_training.spec import atomic_json

from .provenance import seal
from .qualification import qualify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openmc-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = qualify(args.openmc_root)
    atomic_json(args.output, seal(result))
    print(
        json.dumps(
            dict(
                numerical_accepted=result["numerical_accepted"],
                openmc_pipeline_executed=result["openmc_pipeline_executed"],
                external_gates=result["external_gates"],
            )
        )
    )
    return 0 if result["numerical_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
