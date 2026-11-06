"""Cold installed-wheel numerical qualification, examples, restart and tests."""

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
    import atomic_model_spatial.workflow as workflow

    if not Path(workflow.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Spatial module not imported from installed environment")
    commands = [
        ["-m", "atomic_model_spatial", "audit", str(a.inputs / "qualification.json")],
        [
            "-m",
            "atomic_model_spatial",
            "qualify",
            "--output",
            str(a.inputs / "installed_qualification.json"),
        ],
        [
            "-m",
            "atomic_model_spatial",
            "example",
            "slab_eigen_v1",
            "--output",
            str(a.inputs / "eigen_input.json"),
        ],
        [
            "-m",
            "atomic_model_spatial",
            "run",
            str(a.inputs / "eigen_input.json"),
            "--output",
            str(a.inputs / "eigen_output.json"),
        ],
        [
            "-m",
            "atomic_model_spatial",
            "example",
            "slab_transient_v1",
            "--output",
            str(a.inputs / "transient_input.json"),
        ],
        [
            "-m",
            "atomic_model_spatial",
            "run",
            str(a.inputs / "transient_input.json"),
            "--output",
            str(a.inputs / "transient_output.json"),
        ],
        ["-m", "pytest", "-q", str(a.inputs / "test_spatial_diffusion.py")],
    ]
    steps = []
    for argv in commands:
        result = subprocess.run(
            [sys.executable, *argv], capture_output=True, text=True, timeout=180
        )
        steps.append(
            dict(
                argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
            )
        )
        if result.returncode:
            raise ValueError(f"Installed verification failed: {steps[-1]}")
    if json.loads((a.inputs / "installed_qualification.json").read_text()) != json.loads(
        (a.inputs / "qualification.json").read_text()
    ):
        raise ValueError("Installed qualification differs")
    # Exercise the actual installed console script as well as python -m.
    console = subprocess.run(
        [str(Path(sys.executable).parent / "atomic-spatial"), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if console.returncode or "qualify" not in console.stdout:
        raise ValueError("Installed console entrypoint failed")
    receipt = dict(
        schema_version=1,
        source=workflow.identity(),
        module_path=workflow.__file__,
        steps=steps,
        console_entrypoint_returncode=console.returncode,
        scope="Outside checkout, same host and shared dependency environment; not an OS/hardware matrix",
        validated=False,
        plant_referenced=False,
    )
    with a.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
