"""Outside-checkout qualification driver; run with an installed wheel's Python.

Arguments: evidence directory (campaign.json/test_hem_qualified.py and optional
old_graph/old_mixture checkpoints), receipt path. No downloads or mutation of
installed source; local shared dependencies are not an independent OS matrix.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    import atomic_model_hem_qualified.model as model

    if not Path(model.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Module is not from installed environment")
    steps = []

    def run(argv):
        result = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            result.check_returncode()
        steps.append(
            dict(
                argv=argv, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
            )
        )

    run(["-m", "atomic_model_hem_qualified", "audit", str(args.inputs / "campaign.json")])
    run(
        [
            "-m",
            "atomic_model_hem_qualified",
            "demo",
            "--output",
            str(args.inputs / "installed_demo.json"),
        ]
    )
    run(["-m", "atomic_model_hem_qualified", "replay", str(args.inputs / "installed_demo.json")])
    run(["-m", "pytest", "-q", str(args.inputs / "test_hem_qualified.py")])
    for family, folder in [("network", "old_graph"), ("mixture", "old_mixture")]:
        if (args.inputs / folder).exists():
            run(["-m", "atomic_model_hem_" + family, "replay", str(args.inputs / folder)])
    receipt = dict(
        schema_version=1,
        python=sys.version,
        executable=sys.executable,
        module_path=model.__file__,
        source=model.signature(),
        steps=steps,
        scope="Same host, shared dependency environment, outside checkout; no independent OS/hardware claim",
        validated=False,
        plant_referenced=False,
    )
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
