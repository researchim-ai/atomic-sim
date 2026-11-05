"""Run the source-bound HIPER screen/audit and tests outside the checkout."""

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
    import atomic_model_hem_assessment.audit as audit

    if not Path(audit.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Assessment was not imported from installed environment")
    steps = []
    commands = [
        ["-m", "atomic_model_hem_assessment", "audit", str(a.inputs / "screen.json")],
        [
            "-m",
            "atomic_model_hem_assessment",
            "screen",
            "--output",
            str(a.inputs / "installed_screen.json"),
        ],
        ["-m", "pytest", "-q", str(a.inputs / "test_hem_assessment.py")],
    ]
    if (a.inputs / "primary.zip").is_file():
        commands.append(
            ["-m", "atomic_model_hem_assessment", "reimport", str(a.inputs / "primary.zip")]
        )
    for argv in commands:
        run = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        if run.returncode:
            print(run.stdout, file=sys.stderr)
            print(run.stderr, file=sys.stderr)
            run.check_returncode()
        steps.append(
            dict(argv=argv, stdout=run.stdout, stderr=run.stderr, returncode=run.returncode)
        )
    if json.loads((a.inputs / "screen.json").read_text()) != json.loads(
        (a.inputs / "installed_screen.json").read_text()
    ):
        raise ValueError("Installed screen differs from checkout")
    result = dict(
        schema_version=1,
        source=audit.identity(),
        module_path=audit.__file__,
        python=sys.version,
        steps=steps,
        outside_checkout=True,
        screen_exact=True,
        limitation="Same machine, shared dependencies; no independent OS/hardware matrix",
        dynamic_hem_validated=False,
        plant_referenced=False,
    )
    with a.receipt.open("x") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
