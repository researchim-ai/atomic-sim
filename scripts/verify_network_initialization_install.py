"""Verify installed initialization with explicit staged legacy workspace inputs."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("receipt", type=Path)
    args = p.parse_args()
    import atomic_model_initialization.steady as steady

    root = Path(steady.__file__).resolve().parent.parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Initialization not imported from installed environment")
    # Existing graph family reads these paths relative to its working directory.
    # Stage exact resources FROM THE WHEEL, never from a checkout or a new budget.
    budgets = [
        "ai05_inertial_graph_budget_v1.json",
        "ai05_thermal_inertial_budget_v1.json",
        "ai05_buoyant_graph_budget_v1.json",
    ]
    Path("validation").mkdir(exist_ok=False)
    for name in budgets:
        shutil.copyfile(root / "validation" / name, Path("validation") / name)
    Path("scenarios").mkdir(exist_ok=False)
    from scripts.qualify_system_horizons import CASES

    for name in {file for _, file, _ in CASES}:
        shutil.copyfile(root / "scenarios" / name, Path("scenarios") / name)
    commands = [
        [
            "-m",
            "atomic_model_initialization",
            "audit-qualification",
            str(args.inputs / "qualification.json"),
        ],
        [
            "-m",
            "pytest",
            "-q",
            str(args.inputs / "test_network_initialization.py"),
            str(args.inputs / "test_system_horizons.py"),
        ],
    ]
    steps = []
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command], text=True, capture_output=True, timeout=180
        )
        steps.append(
            dict(
                argv=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            raise ValueError(str(steps[-1]))
    receipt = dict(
        schema_version=1,
        source=steady.identity(),
        steps=steps,
        scope="Installed wheel outside checkout, same host/shared dependencies; three legacy graph budgets and six scenario inputs explicitly staged from installed resources",
        module_path=steady.__file__,
        validated=False,
        plant_referenced=False,
    )
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
