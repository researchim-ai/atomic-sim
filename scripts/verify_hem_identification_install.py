"""Outside-checkout identifiability qualification and denied-calibration replay."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("receipt", type=Path)
    a = p.parse_args()
    import atomic_model_hem_identification.workflow as workflow

    if not Path(workflow.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Identification module not imported from installed environment")
    commands = [
        (
            [
                "-m",
                "atomic_model_hem_identification",
                "audit-qualification",
                str(a.inputs / "qualification.json"),
            ],
            0,
        ),
        (["-m", "atomic_model_hem_identification", "audit", str(a.inputs / "assessment.json")], 0),
        (
            [
                "-m",
                "atomic_model_hem_identification",
                "analyse",
                "--output",
                str(a.inputs / "installed_assessment.json"),
                "--require-calibratable",
                "--require-identifiable",
            ],
            3,
        ),
        (["-m", "pytest", "-q", str(a.inputs / "test_hem_identification.py")], 0),
    ]
    steps = []
    for argv, expected in commands:
        r = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        if r.returncode != expected:
            print(r.stdout, file=sys.stderr)
            print(r.stderr, file=sys.stderr)
            raise ValueError("Installed verification command failed")
        steps.append(
            dict(
                argv=argv,
                returncode=r.returncode,
                expected_returncode=expected,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        )
    if json.loads((a.inputs / "assessment.json").read_text()) != json.loads(
        (a.inputs / "installed_assessment.json").read_text()
    ):
        raise ValueError("Installed identification analysis differs")
    result = dict(
        schema_version=1,
        source=workflow.identity(),
        module_path=workflow.__file__,
        steps=steps,
        scope="Same host/shared dependency environment, outside checkout; no independent OS/hardware matrix",
        calibration_status="rejected",
        parameter_update=None,
        validated=False,
        plant_referenced=False,
    )
    with a.receipt.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
