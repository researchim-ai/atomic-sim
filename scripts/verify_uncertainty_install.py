"""Exercise installed UQ CLI criteria, including deliberate non-acceptance."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_uncertainty.coverage import evaluate, identity
from atomic_model_workspace.common import digest


def verify(specification):
    expected = evaluate(specification)
    source = identity()
    results = []
    with tempfile.TemporaryDirectory(prefix="atomic-uq-criteria-") as directory:
        directory = Path(directory)
        atomic_json(directory / "input.json", specification)
        for metric, field in (
            ("pooled", "heldout_coverage"),
            ("family-balanced", "family_balanced_coverage"),
        ):
            output = directory / f"{metric}.json"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "atomic_model_uncertainty",
                    "input.json",
                    "--output",
                    str(output),
                    "--acceptance-metric",
                    metric,
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )
            accepted = expected[field] >= specification["target_coverage"]
            if process.returncode != (0 if accepted else 2):
                raise ValueError(f"UQ CLI exit differs for {metric}: {process.stderr}")
            report = unseal(read_json(output))
            acceptance = report.pop("acceptance")
            if report != expected or acceptance != dict(
                metric=field,
                value=expected[field],
                target=specification["target_coverage"],
                target_met=accepted,
                experimental_validation_promoted=False,
            ):
                raise ValueError("Installed UQ CLI report differs from Python evaluation")
            if json.loads(process.stdout)["acceptance"] != acceptance:
                raise ValueError("UQ CLI summary differs from sealed report")
            results.append(dict(metric=metric, exit_code=process.returncode, acceptance=acceptance))
        if read_json(directory / "input.json") != specification or identity() != source:
            raise ValueError("UQ input or implementation changed during verification")
    return dict(
        source=source,
        input_sha256=digest(specification),
        criteria=results,
        python_cli_equal=True,
        validated=False,
        plant_referenced=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    atomic_json(args.output, seal(verify(read_json(args.input))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
