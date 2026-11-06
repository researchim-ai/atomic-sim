"""Evaluate supplied uncertainty budget; never infer missing covariance."""

import argparse
import json

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json, read_json

from .coverage import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(read_json(args.input))
    atomic_json(args.output, seal(result))
    print(
        json.dumps(
            dict(
                heldout_coverage=result["heldout_coverage"],
                target_met=result["target_met"],
                validated=False,
            )
        )
    )
    return 0 if result["target_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
