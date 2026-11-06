"""Execute complete controlled-plant programs, retaining failed full requests."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atomic_model_controlled_plant.session import identity

PROGRAMS = [
    ("load_rejection", "load_rejection_equilibrium", []),
    ("coastdown", "coastdown", []),
    (
        "coastdown_peer",
        "coastdown",
        ["--method", "Radau", "--max-step", "1", "--rtol", "1e-7", "--atol", "1e-6"],
    ),
    ("natural", "natural", []),
    ("natural_perturbation", "natural_perturbation", ["--method", "BDF"]),
    ("nominal_radau", "nominal", []),
    (
        "nominal_bdf",
        "nominal",
        ["--method", "BDF", "--max-step", "1", "--rtol", "1e-7", "--atol", "1e-6"],
    ),
    ("scram", "scram", []),
    ("feed", "feed_disturbance", []),
    ("sink", "sink_disturbance", []),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", choices=[1, 2, 3], type=int, default=3)
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
            "atomic_model_controlled_plant",
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

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(run, PROGRAMS))
    if identity() != source:
        raise ValueError("Controlled campaign source changed")
    (a.output / "execution.json").write_text(
        json.dumps(dict(source=source, runs=results), indent=2) + "\n"
    )
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
