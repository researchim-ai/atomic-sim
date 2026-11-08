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
    parser.add_argument(
        "--acceptance-metric",
        choices=("pooled", "family-balanced"),
        default="pooled",
        help="Coverage used for exit status; recorded in the sealed report (default: pooled)",
    )
    args = parser.parse_args()
    result = evaluate(read_json(args.input))
    metric = (
        "heldout_coverage" if args.acceptance_metric == "pooled" else "family_balanced_coverage"
    )
    accepted = result[metric] >= result["target_coverage"]
    result["acceptance"] = dict(
        metric=metric,
        value=result[metric],
        target=result["target_coverage"],
        target_met=accepted,
        experimental_validation_promoted=False,
    )
    atomic_json(args.output, seal(result))
    print(
        json.dumps(
            dict(
                heldout_coverage=result["heldout_coverage"],
                target_met=result["target_met"],
                heldout_families=result["heldout_families"],
                family_balanced_coverage=result["family_balanced_coverage"],
                family_balanced_target_met=result["family_balanced_target_met"],
                acceptance=result["acceptance"],
                validated=False,
            )
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
