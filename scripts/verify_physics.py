"""Local numerical physics profile; never experimental or plant qualification."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest

ROOT = Path(__file__).resolve().parents[1]
LANES = {
    "eos-phase": {
        "scope": "IF97 reference states, inverse EOS, phase endpoints, heating/cooling and phase restart",
        "tests": ["tests/test_hem_equilibrium.py", "tests/test_hem_qualified.py"],
    },
    "balances": {
        "scope": "Shared face fluxes, mass/momentum/energy inventories, wall exchange and rollback",
        "tests": ["tests/test_hem_flow.py", "tests/test_finite_hem_system.py"],
    },
    "convergence": {
        "scope": "Acoustic mesh refinement, network temporal reference and convergence-gate negative controls",
        "tests": [
            "tests/test_hem_flow.py::test_spatial_acoustic_cell_average_error_decreases",
            "tests/test_hem_network_reference.py::test_model_pulse_temporal_reference",
            "tests/test_transient_convergence.py",
            "tests/test_phase_temporal_refinement.py",
        ],
    },
    "restart": {
        "scope": "Exact restart during network pulse/wall exchange and spatial source history",
        "tests": [
            "tests/test_hem_network_reference.py::test_model_restart_during_pulse_and_wall_exchange",
            "tests/test_spatial_diffusion.py::test_time_dependent_source_and_restart_full_history",
        ],
    },
    "controlled-pwr": {
        "scope": "Short actual reduced-PWR SCRAM/residual heat/coastdown, boundary disturbances, replay and horizon-gate negative controls",
        "tests": [
            "tests/test_controlled_plant.py",
            "tests/test_controlled_physics_regression.py",
            "tests/test_system_horizons.py",
        ],
    },
    "neutronics": {
        "scope": "1D diffusion analytic states, local kinetics exponential and neutron inventory balance",
        "tests": [
            "tests/test_spatial_diffusion.py::test_one_group_reflecting_analytic_source_and_k",
            "tests/test_spatial_diffusion.py::test_k_continuum_and_absorber_monotonicity",
            "tests/test_spatial_diffusion.py::test_critical_state_local_precursors_and_balance",
            "tests/test_spatial_diffusion.py::test_independent_homogeneous_reflecting_local_kinetics_exponential",
            "tests/test_spatial_diffusion.py::test_current_divergence_boundary_signs_and_local_balance",
        ],
    },
}


EXTENDED_LANES = {
    "phase-mesh": {
        "scope": "Fixed-geometry phase transitions on nested meshes with separate time-step diagnostics; finite 32-cell reference",
        "tests": ["tests/test_phase_mesh_refinement.py"],
    }
}


def source_manifest():
    paths = {ROOT / "pyproject.toml", Path(__file__).resolve()}
    paths.update(
        p
        for p in (ROOT / "src").rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.relative_to(ROOT / "src").parts[0].endswith(".egg-info")
    )
    # Tests may call helpers from scripts or load root-mapped resource packages.
    for name in ("scripts", "data", "scenarios", "validation"):
        paths.update(
            p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    paths.update(
        ROOT / node.split("::")[0]
        for lane in (LANES | EXTENDED_LANES).values()
        for node in lane["tests"]
    )
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def receipt_passed(receipt, code):
    if not isinstance(receipt, dict) or type(code) is not int or code != 0:
        return False
    counts = receipt.get("counts")
    if not isinstance(counts, dict) or any(
        type(counts.get(key)) is not int or counts[key] < 0
        for key in ("collected", "passed", "failed", "errors", "skipped")
    ):
        return False
    return (
        type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("gate_failures") == []
        and counts["collected"] == counts["passed"] > 0
        and counts["failed"] == counts["errors"] == counts["skipped"] == 0
    )


def verify(output, *, timeout_s=600, include_phase_mesh=False):
    if type(timeout_s) is not int or not 1 <= timeout_s <= 14400:
        raise ValueError("Timeout domain 1..14400 seconds")
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output must not be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    before = source_manifest()
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    results = []
    chosen_lanes = LANES | EXTENDED_LANES if include_phase_mesh else LANES
    for name, lane in chosen_lanes.items():
        print(name, flush=True)
        receipt_path = output / f"{name}.json"
        started = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "pytest",
            *lane["tests"],
            "-q",
            "-p",
            "atomic_model_release.pytest_gate",
            "--atomic-require-executed",
            "--atomic-no-skips",
            "--atomic-receipt",
            str(receipt_path),
            "--junitxml",
            str(output / f"{name}.xml"),
        ]
        with (output / f"{name}.log").open("xb") as stream:
            try:
                code = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
        error = None
        try:
            receipt = read_json(receipt_path)
        except (OSError, ValueError) as exc:
            receipt = None
            error = str(exc)
        results.append(
            dict(
                name=name,
                scope=lane["scope"],
                tests=lane["tests"],
                exit_code=code,
                passed=receipt_passed(receipt, code),
                receipt=receipt,
                receipt_error=error,
                wall_s=time.monotonic() - started,
                log=f"{name}.log",
            )
        )
        atomic_json(output / "progress.json", seal(dict(completed=False, lanes=results)))
    after = source_manifest()
    unchanged = before == after
    report = dict(
        schema_version=1,
        profile="local-numerical-physics-v3",
        include_phase_mesh=include_phase_mesh,
        completed=True,
        passed=unchanged and all(r["passed"] for r in results),
        source_unchanged=unchanged,
        source_sha256=digest(before),
        source_manifest=before,
        lanes=results,
        environment=dict(
            python=platform.python_version(),
            system=platform.system(),
            machine=platform.machine(),
            **{n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pytest")},
        ),
        limitations=[
            "Selected numerical regressions, not exhaustive qualification",
            "Some tests repeat across lanes; counts must not be summed as unique tests",
            "Convergence gate tests include manufactured traces; not all are solver convergence experiments",
            "No experimental validation or whole-plant reference implied",
        ],
        ci_required=False,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(output / "report.json", seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--phase-timeout-s", type=int, default=600)
    parser.add_argument(
        "--include-phase-mesh",
        action="store_true",
        help="Also run the slower phase-change spatial refinement study",
    )
    args = parser.parse_args(argv)
    output = args.output or str(Path("artifacts") / ("physics-" + uuid.uuid4().hex[:10]))
    try:
        report = verify(
            output, timeout_s=args.phase_timeout_s, include_phase_mesh=args.include_phase_mesh
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"passed={report['passed']} output={output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
