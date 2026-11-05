"""Outside-checkout numerical qualification, negative benchmark and restart checks."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("receipt", type=Path)
    a = parser.parse_args()
    import atomic_model_hem_throughflow.model as model

    if not Path(model.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Model not imported from installed runtime")
    steps = []
    commands = [
        (
            [
                "-m",
                "atomic_model_hem_throughflow",
                "audit-qualification",
                str(a.inputs / "qualification.json"),
            ],
            0,
        ),
        (
            [
                "-m",
                "atomic_model_hem_throughflow",
                "audit-benchmark",
                str(a.inputs / "benchmark.json"),
            ],
            0,
        ),
        (["-m", "atomic_model_hem_throughflow", "replay", str(a.inputs / "demo.json")], 0),
        (
            [
                "-m",
                "atomic_model_hem_throughflow",
                "benchmark",
                "--output",
                str(a.inputs / "installed_benchmark.json"),
                "--require-pass",
            ],
            3,
        ),
        (["-m", "pytest", "-q", str(a.inputs / "test_hem_throughflow.py")], 0),
    ]
    for argv, expected in commands:
        result = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        if result.returncode != expected:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            raise ValueError(f"Installed command returned{result.returncode}, expected{expected}")
        steps.append(
            dict(
                argv=argv,
                returncode=result.returncode,
                expected_returncode=expected,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
    if json.loads((a.inputs / "benchmark.json").read_text()) != json.loads(
        (a.inputs / "installed_benchmark.json").read_text()
    ):
        raise ValueError("Installed conditional benchmark changed")
    receipt = dict(
        schema_version=1,
        source=model.signature(),
        module_path=model.__file__,
        python=sys.version,
        steps=steps,
        scope="Same host with shared dependencies, outside checkout; no independent OS/hardware claim",
        numerical_verification_repeated=True,
        experimental_screening_status="failed",
        validated=False,
        plant_referenced=False,
    )
    with a.receipt.open("x") as f:
        json.dump(receipt, f, indent=2)


if __name__ == "__main__":
    main()
