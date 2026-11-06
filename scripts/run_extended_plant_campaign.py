"""Execute the whole predeclared campaign, preserving every failed result."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atomic_model_plant_extension.session import identity

PROGRAMS = [
    ("nominal_radau", "nominal", ["--max-step", "2"]),
    (
        "nominal_bdf",
        "nominal",
        ["--max-step", "2", "--method", "BDF", "--rtol", "1e-7", "--atol", "1e-6"],
    ),
    ("scram", "scram", ["--output-step", "2"]),
    ("coastdown", "coastdown", ["--output-step", "2"]),
    ("load_rejection", "load_rejection", ["--output-step", "2"]),
    ("feed", "feed_disturbance", ["--output-step", "2"]),
    ("sink", "sink_disturbance", ["--output-step", "2"]),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--cases", nargs="+", choices=[r[0] for r in PROGRAMS])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = identity()
    (a.output / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def run(program):
        label, case, options = program
        command = [
            sys.executable,
            "-m",
            "atomic_model_plant_extension",
            "run",
            case,
            "--output",
            str(a.output / label),
            *options,
        ]
        with (a.output / (label + ".log")).open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(label, result.returncode, flush=True)
        return dict(label=label, returncode=result.returncode)

    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        selected = [r for r in PROGRAMS if a.cases is None or r[0] in a.cases]
        results = list(executor.map(run, selected))
    if source != identity():
        raise ValueError("Campaign source changed during execution")
    (a.output / "execution.json").write_text(
        json.dumps(dict(source=source, runs=results), indent=2) + "\n"
    )
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
