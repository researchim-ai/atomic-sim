"""Local library verification: tests, wheel/sdist, isolated install and replay.

Requires the project's development tools in the invoking Python environment.
Scientific dependencies are reused offline; a full dependency resolver or a
separate physical machine is not part of this verification contract.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_TESTS = [
    "tests/test_controlled_agent_tools.py",
    "tests/test_agent_tools.py",
    "tests/test_agent_tools_cli.py",
    "tests/test_pwr_agent.py",
    "tests/test_external_acceptance.py",
    "tests/test_uncertainty_coverage.py",
    "tests/test_physics_verification.py",
    "tests/test_pwr_horizon_profile.py",
    "tests/test_pwr_refinement.py",
    "tests/test_pwr_windowed_reference.py",
    "tests/test_phase_release.py",
    "tests/test_hem_closure_audit.py",
    "tests/test_chf_screening.py",
    "tests/test_multiphysics.py",
    "tests/test_research_project.py",
    "tests/test_training_campaign.py",
    "tests/test_spatial_diffusion.py",
    "tests/test_release_engineering.py",
]


def selection(suite):
    if suite == "library":
        return list(LIBRARY_TESTS)
    if suite == "portable":
        return [
            "tests",
            "-m",
            "not external_integration and not long_running and not hardware_specific",
        ]
    if suite == "all":
        return ["tests"]
    raise ValueError("Unknown local test suite")


def check_wheel(path, *, source_root=None, resource_roots=None):
    root = Path(source_root) if source_root is not None else ROOT / "src"
    expected = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.relative_to(root).parts[0].endswith(".egg-info")
    }
    if not expected:
        raise ValueError("Empty source inventory")
    if resource_roots is None:
        resource_roots = (
            {name: ROOT / name for name in ("scripts", "data", "scenarios", "validation")}
            if source_root is None
            else {}
        )
    counts = {"src": len(expected)}
    for name, directory in resource_roots.items():
        if name not in {"scripts", "data", "scenarios", "validation"}:
            raise ValueError("Unknown root-mapped package")
        directory = Path(directory)
        resources = {
            name + "/" + p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        if not resources or not (directory / "__init__.py").is_file():
            raise ValueError("Missing root-mapped package: " + name)
        if set(resources) & set(expected):
            raise ValueError("Overlapping source and resource inventory")
        expected.update(resources)
        counts[name] = len(resources)
    source_names = {name.split("/")[0] for name in expected}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate archive entries")
        if any(
            "__pycache__" in name.split("/") or name.endswith((".pyc", ".pyo")) for name in names
        ):
            raise ValueError("Wheel contains compiled Python cache")
        extras = {
            name.split("/")[0]
            for name in names
            if name.split("/")[0] not in source_names
            and not name.split("/")[0].endswith(".dist-info")
        }
        if extras:
            raise ValueError("Wheel contains unexpected top-level modules: " + str(sorted(extras)))
        actual = {
            name for name in names if name.split("/")[0] in source_names and not name.endswith("/")
        }
        if actual != set(expected):
            raise ValueError("Wheel contains missing or stale source modules/resources")
        if any(archive.read(name) != raw for name, raw in expected.items()):
            raise ValueError("Wheel source bytes differ")
    return dict(
        verified=True,
        files=len(expected),
        package_counts=counts,
        scope="src and declared root-mapped packages; exact bytes; no caches or duplicates",
    )


def verify(directory, *, suite="library", phase_timeout_s=1800):
    from atomic_agent_tools.server import provenance as agent_provenance
    from atomic_model_acceptance.replay import compare, make_request, source_identity, verify_packet
    from atomic_model_multiphysics.provenance import seal, unseal
    from atomic_model_training.spec import atomic_json, read_json
    from atomic_model_uncertainty.coverage import identity as uncertainty_identity

    chosen = selection(suite)
    verifier_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "tests").is_dir():
        raise ValueError("Run local verification from the source checkout with tests")
    if type(phase_timeout_s) is not int or not 1 <= phase_timeout_s <= 14400:
        raise ValueError("Phase timeout domain1..14400 seconds")
    output = Path(directory).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Verification directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    phases = []

    def run(name, command, *, cwd=ROOT):
        started = time.monotonic()
        log = output / (name + ".log")
        print(name, flush=True)
        try:
            with log.open("xb") as stream:
                result = subprocess.run(
                    list(map(str, command)),
                    cwd=cwd,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=phase_timeout_s,
                )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
        phases.append(
            dict(
                name=name,
                exit_code=code,
                passed=code == 0,
                wall_s=time.monotonic() - started,
                log=log.name,
            )
        )
        atomic_json(output / "progress.json", seal(dict(phases=phases, completed=False)))
        if code:
            raise ValueError(f"Local verification failed: {name}; see {log}")

    run(
        "source-tests",
        [
            sys.executable,
            "-m",
            "pytest",
            *chosen,
            "-q",
            "-p",
            "atomic_model_release.pytest_gate",
            "--atomic-require-executed",
            "--atomic-receipt",
            output / "tests.json",
        ],
    )
    tests = read_json(output / "tests.json")
    if tests["exit_code"] or tests["counts"]["passed"] == 0:
        raise ValueError("Local tests did not execute successfully")
    run(
        "build-source",
        [sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir", output / "build"],
    )
    sdists = list((output / "build").glob("*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError("Expected one source distribution")
    # Build from a clean sdist: removed Python modules must not survive in build/lib.
    with tempfile.TemporaryDirectory(prefix="atomic-clean-wheel-") as clean:
        with tarfile.open(sdists[0]) as archive:
            archive.extractall(clean, filter="data")
        roots = [p for p in Path(clean).iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "pyproject.toml").is_file():
            raise ValueError("Unexpected source distribution layout")
        run(
            "build-wheel",
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                output / "build",
            ],
            cwd=roots[0],
        )
    wheels = list((output / "build").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one wheel")
    wheel_inventory = check_wheel(wheels[0])
    request = make_request(read_json(ROOT / "examples/research/project.json"), output / "request")
    verify_packet(output / "request")
    expected_source = source_identity()
    expected_uncertainty = uncertainty_identity()
    expected_agent = agent_provenance()
    with tempfile.TemporaryDirectory(prefix="atomic-local-verification-") as temporary:
        temporary = Path(temporary)
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(temporary / "venv")
        python = temporary / "venv/bin/python"
        if os.name == "nt":
            python = temporary / "venv/Scripts/python.exe"
        work = temporary / "work"
        work.mkdir()
        for name in ("source.json", "project.zip", "request.json"):
            shutil.copyfile(output / "request" / name, work / name)
        run(
            "install",
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                wheels[0],
            ],
            cwd=work,
        )
        run(
            "installed-replay",
            [
                python,
                "-m",
                "scripts.verify_research_install",
                "source.json",
                "project.zip",
                "replay",
                "installed.json",
            ],
            cwd=work,
        )
        run(
            "installed-uncertainty",
            [
                python,
                "-m",
                "atomic_model_uncertainty",
                ROOT / "examples/research/uncertainty.json",
                "--output",
                "uq.json",
            ],
            cwd=work,
        )
        run(
            "installed-uncertainty-criteria",
            [
                python,
                "-m",
                "scripts.verify_uncertainty_install",
                ROOT / "examples/research/uncertainty_family_imbalance.json",
                "uq-criteria.json",
            ],
            cwd=work,
        )
        criteria = unseal(read_json(work / "uq-criteria.json"))
        if (
            criteria["source"] != expected_uncertainty
            or not criteria["python_cli_equal"]
            or [r["exit_code"] for r in criteria["criteria"]] != [0, 2]
        ):
            raise ValueError("Installed UQ criteria verification differs")
        run(
            "installed-agent-interfaces",
            [python, "-m", "scripts.verify_pwr_agent_install", "agent.json"],
            cwd=work,
        )
        agent = read_json(work / "agent.json")
        if agent["provenance"] != expected_agent or not all(
            agent[k]
            for k in (
                "python_tool_equal",
                "host_restart_equal",
                "jsonl_restart_equal",
                "gym_lifecycle_passed",
            )
        ):
            raise ValueError("Installed agent source/interface verification differs")
        installed = unseal(read_json(work / "installed.json"))
        uq = unseal(read_json(work / "uq.json"))
        compared = compare(seal(request), installed["project_run"])
        if (
            not compared["numerical_replay_accepted"]
            or uq["source"] != expected_uncertainty
            or not uq["target_met"]
        ):
            raise ValueError("Installed replay or uncertainty source/criteria differ")
        for name in ("installed.json", "uq.json", "uq-criteria.json", "agent.json"):
            shutil.copyfile(work / name, output / name)
        atomic_json(output / "comparison.json", seal(compared))
    if (
        expected_agent != agent_provenance()
        or expected_source != source_identity()
        or expected_uncertainty != uncertainty_identity()
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != verifier_sha256
    ):
        raise ValueError("Source files changed during local verification")
    report = dict(
        schema_version=1,
        verifier_sha256=verifier_sha256,
        source=expected_source,
        uncertainty_source=expected_uncertainty,
        agent_source=expected_agent,
        suite=suite,
        tests=tests,
        phases=phases,
        completed=True,
        numerical_replay_accepted=True,
        installed_resource_count=installed["resource_count"],
        build_artifacts=[p.name for p in wheels + sdists],
        wheel_inventory=wheel_inventory,
        verification_scope="local source, distribution and installed-library replay",
        ci_required=False,
        remote_execution_required=False,
        network_required=False,
        dependency_scope="offline wheel install in fresh venv, reusing installed scientific dependencies",
        validated=False,
        plant_referenced=False,
    )
    atomic_json(output / "report.json", seal(report))
    print(json.dumps(dict(completed=True, passed=tests["counts"]["passed"], output=str(output))))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        help="New result directory; default creates a unique artifacts/local-verification-<id>",
    )
    parser.add_argument("--suite", choices=["library", "portable", "all"], default="library")
    parser.add_argument("--phase-timeout-s", type=int, default=1800)
    args = parser.parse_args(argv)
    output = args.output or str(Path("artifacts") / ("local-verification-" + uuid.uuid4().hex[:10]))
    try:
        verify(output, suite=args.suite, phase_timeout_s=args.phase_timeout_s)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
