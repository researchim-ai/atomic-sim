"""Offline trusted-wheel runbook verification outside checkout; dependencies are shared."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    wheel = a.wheel.resolve()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    required = {
        "atomic_model_release/golden/manifest.json",
        "atomic_model_release/golden/benchmark.json",
        "atomic_model_release/use_cases.json",
        "atomic_model_release/py.typed",
        "atomic_model_release/golden/oracle.json",
        "atomic_model_release/golden/actuated_oracle.json",
        "atomic_model_release/golden/project/project.json",
        "atomic_model_agents/jobs.pyi",
        "atomic_model_agents/py.typed",
        "atomic_model_contracts/py.typed",
    }
    with zipfile.ZipFile(wheel) as archive:
        if not required <= set(archive.namelist()):
            raise ValueError("Wheel is missing release resources or public type declarations")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    phases = []
    with tempfile.TemporaryDirectory(prefix="atomic-release-install-") as temporary:
        root = Path(temporary)
        work = root / "work"
        work.mkdir()
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / "venv")
        python = root / "venv/bin/python"

        def run(name, command, expected_exit=0):
            with (output / (name + ".log")).open("w") as stream:
                done = subprocess.run(
                    list(map(str, command)),
                    cwd=work,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            phases.append(
                dict(
                    name=name,
                    returncode=done.returncode,
                    expected_exit=expected_exit,
                    passed=done.returncode == expected_exit,
                )
            )
            if done.returncode != expected_exit:
                raise ValueError("Installed release phase failed: " + name)

        run(
            "install",
            [python, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", wheel],
        )
        run(
            "module_location",
            [
                python,
                "-c",
                "import atomic_model_release,pathlib,sys; p=pathlib.Path(atomic_model_release.__file__).resolve(); assert pathlib.Path(sys.prefix).resolve() in p.parents; print(p)",
            ],
        )
        executable = root / "venv/bin/atomic-model-release"
        (work / "typed_positive.py").write_text(
            'from pathlib import Path\nfrom atomic_model_agents import jobs\nfrom atomic_model_contracts import Scenario,AgentDescriptor\nfrom atomic_model_release.api_contract import start_checked_job\ndef call(s:Scenario,a:AgentDescriptor)->None:\n jobs.create(s,a,Path("job"),["python"])\n start_checked_job(s,a,Path("other"),["python"])\n'
        )
        (work / "typed_negative.py").write_text(
            'from atomic_model_agents import jobs\njobs.create("bad-scenario","bad-descriptor","job",["python"])\n'
        )
        run(
            "installed_types_positive",
            [python, "-m", "mypy", "--strict", "--follow-imports=silent", "typed_positive.py"],
        )
        run(
            "installed_types_negative",
            [python, "-m", "mypy", "--strict", "--follow-imports=silent", "typed_negative.py"],
            expected_exit=1,
        )
        if "[arg-type]" not in (output / "installed_types_negative.log").read_text():
            raise ValueError("Installed type gate did not reject actual argument types")
        run("doctor", [executable, "doctor"])
        run("golden", [executable, "golden"])
        run("runbook", [executable, "control-demo", "--output", "case"])
        run("diagnosis", [executable, "diagnose", "case/job"])
        run("use_cases", [executable, "use-cases"])
        run("diagnostic_demo", [executable, "diagnostic-demo", "--output", "diagnostic_case"])
        run("data_demo", [executable, "data-demo", "--output", "data_case"])
        shutil.copytree(work / "diagnostic_case", output / "diagnostic_case")
        shutil.copytree(work / "data_case", output / "data_case")

        shutil.copytree(work / "case", output / "case")
    proof = dict(
        schema_version=1,
        phases=phases,
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        dependency_scope="offline --no-deps wheel in fresh venv outside checkout; system scientific dependencies inherited, not hermetic",
        runtime_matrix_executed=False,
        external_runtimes_executed=False,
        validated=False,
        plant_referenced=False,
    )
    (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
