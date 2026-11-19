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
    "tests/test_phase_branching.py",
    "tests/test_phase_cycles.py",
    "tests/test_external_benchmark.py",
    "tests/test_if97_optimization.py",
    "tests/test_modelica_benchmark.py",
    "tests/test_phase_boundary_benchmark.py",
    "tests/test_phase_wall_reference.py",
    "tests/test_msl_coefficient_audit.py",
    "tests/test_phase_event.py",
    "tests/test_msl_saturation.py",
    "tests/test_phase_event_readmission.py",
    "tests/test_modelica_network.py",
    "tests/test_contact_flux.py",
    "tests/test_modelica_hllc.py",
    "tests/test_liquid_riemann.py",
    "tests/test_local_hllc.py",
    "tests/test_riemann_profile.py",
    "tests/test_modelica_local_hllc.py",
    "tests/test_hllc_long.py",
    "tests/test_hllc_speed_cache.py",
    "tests/test_hllc_liquid_inverse.py",
    "tests/test_hllc_direct_rhs.py",
    "tests/test_hllc_branching.py",
    "tests/test_hllc_transactions.py",
    "tests/test_heated_hllc.py",
    "tests/test_modelica_heated_benchmark.py",
    "tests/test_modeler_comparison.py",
    "tests/test_agent_decisions.py",
    "tests/test_thermal_agent.py",
    "tests/test_thermal_energy.py",
    "tests/test_thermal_faults.py",
    "tests/test_thermal_observer.py",
    "tests/test_thermal_recovery.py",
    "tests/test_thermal_recovery_audit.py",
    "tests/test_adaptive_thermal_recovery.py",
    "tests/test_adaptive_recovery_audit.py",
    "tests/test_thermal_drift.py",
    "tests/test_beavrs_boron.py",
    "tests/test_beavrs_boron_sensitivity.py",
    "tests/test_beavrs_temperature.py",
    "tests/test_neutronic_response.py",
    "tests/test_station_r0.py",
    "tests/test_station_assembly.py",
    "tests/test_station_foundation.py",
    "tests/test_physical_pwr_train.py",
    "tests/test_station_actuators.py",
    "tests/test_station_cooling.py",
    "tests/test_station_feed.py",
    "tests/test_station_power.py",
    "tests/test_station_core.py",
    "tests/test_station_primary.py",
    "tests/test_station_film.py",
    "tests/test_station_steam.py",
    "tests/test_station_stages.py",
    "tests/test_station_bypass.py",
    "tests/test_station_signals.py",
    "tests/test_station_control.py",
    "tests/test_station_feed_control.py",
    "tests/test_station_faults.py",
    "tests/test_station_campaign.py",
    "tests/test_station_env.py",
    "tests/test_station_evaluation.py",
    "tests/test_station_tasks.py",
    "tests/test_station_energy_eval.py",
    "tests/test_station_heat_accounting.py",
    "tests/test_station_event_heat.py",
    "tests/test_station_refined_heat.py",
    "tests/test_station_piecewise_heat.py",
    "tests/test_station_jump_heat.py",
    "tests/test_station_hybrid_heat.py",
    "tests/test_station_hybrid_event_groups.py",
    "tests/test_station_condensate_reopening.py",
    "tests/test_station_feed_reopening.py",
    "tests/test_station_relief_heat.py",
    "tests/test_station_hybrid_schedule.py",
    "tests/test_station_cycle_catalog.py",
    "tests/test_station_hybrid_actions.py",
    "tests/test_station_cost_matrix.py",
    "tests/test_station_water_cache.py",
    "tests/test_station_nominal_requalification.py",
    "tests/test_station_examples.py",
    "tests/test_station_capabilities.py",
    "tests/test_station_controlled_recovery.py",
    "tests/test_spatial_response.py",
    "tests/test_station_evidence.py",
    "tests/test_station_precision.py",
    "tests/test_station_view.py",
    "tests/test_station_delivery.py",
    "tests/test_station_history.py",
    "tests/test_local_test_batches.py",
    "tests/test_hem_network.py",
    "tests/test_hem_network_reference.py",
    "tests/test_hem_qualified.py",
    "tests/test_hem_mixture.py",
    "tests/test_station_readiness.py",
    "tests/test_station_hourly_protocol.py",
    "tests/test_station_inventory.py",
    "tests/test_station_inventory_deadband.py",
    "tests/test_station_hot.py",
    "tests/test_station_equilibrium.py",
    "tests/test_station_load.py",
    "tests/test_station_load_envelope.py",
    "tests/test_station_zero_demand.py",
    "tests/test_station_shutdown.py",
    "tests/test_station_gnielinski.py",
    "tests/test_station_solver_budget.py",
    "tests/test_station_process.py",
    "tests/test_station_low_flow_qualification.py",
    "tests/test_station_mixed_convection.py",
    "tests/test_station_mixed_qualification.py",
    "tests/test_station_pipe.py",
    "tests/test_station_pipe_qualification.py",
    "tests/test_station_turbulence.py",
    "tests/test_station_turbulence_qualification.py",
    "tests/test_station_heated.py",
    "tests/test_station_heated_qualification.py",
    "tests/test_station_quadrature.py",
    "tests/test_station_quadrature_qualification.py",
    "tests/test_station_spectral.py",
    "tests/test_station_spectral_qualification.py",
    "tests/test_station_initialization.py",
    "tests/test_station_pressure.py",
    "tests/test_station_regular.py",
    "tests/test_station_regular_qualification.py",
    "tests/test_station_stokes_pressure.py",
    "tests/test_station_stokes_pressure_qualification.py",
    "tests/test_station_wall.py",
    "tests/test_station_wall_qualification.py",
    "tests/test_station_advection.py",
    "tests/test_station_advection_qualification.py",
    "tests/test_station_nonlinear.py",
    "tests/test_station_nonlinear_qualification.py",
    "tests/test_station_resolution.py",
    "tests/test_station_resolution_qualification.py",
    "tests/test_station_adaptive.py",
    "tests/test_station_adaptive_qualification.py",
    "tests/test_station_refined.py",
    "tests/test_station_refined_qualification.py",
    "tests/test_station_thermal.py",
    "tests/test_station_thermal_qualification.py",
    "tests/test_station_thermal_resolved.py",
    "tests/test_station_thermal_resolved_qualification.py",
    "tests/test_station_thermal_path.py",
    "tests/test_station_thermal_path_qualification.py",
    "tests/test_station_thermal_cases.py",
    "tests/test_station_thermal_cases_qualification.py",
    "tests/test_station_thermal_case_paths_qualification.py",
    "tests/test_station_thermal_stationary_reference.py",
    "tests/test_station_thermal_radial_catalog.py",
    "tests/test_station_thermal_transient_reference.py",
    "tests/test_station_thermal_radial_paths.py",
    "tests/test_station_thermal_radial_resume.py",
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


def source_manifest(root=ROOT):
    """Freeze all shipped code/resources and packaging rules, not generated caches."""
    root=Path(root)
    result={}
    for name in ('src','scripts','data','scenarios','validation','requirements','tests','examples','configs'):
        for p in sorted((root/name).rglob('*')):
            if (p.is_file() and '__pycache__' not in p.parts and
                not any(part.endswith('.egg-info') for part in p.relative_to(root).parts)):
                result[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('pyproject.toml','MANIFEST.in','requirements.txt'):
        p=root/name
        if p.is_file():result[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    if not result:raise ValueError('Empty source manifest')
    return result


def evidence_manifest(root=ROOT):
    root=Path(root)
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'docs/benchmarks').glob('*.json')) if p.is_file()}


def evidence_unchanged(expected,root=ROOT):
    # New receipts may be published later; every validation input stays fixed.
    root=Path(root)
    return all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==value
        for name,value in expected.items())


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


def verify(directory, *, suite="library", phase_timeout_s=1800, station_nominal=None, test_receipts=None):
    from atomic_agent_tools.server import provenance as agent_provenance
    from atomic_model_acceptance.replay import compare, make_request, source_identity, verify_packet
    from atomic_model_multiphysics.provenance import seal, unseal
    from atomic_model_training.spec import atomic_json, read_json
    from atomic_model_uncertainty.coverage import identity as uncertainty_identity

    chosen = selection(suite)
    expected_tree = source_manifest()
    expected_evidence = evidence_manifest()
    if station_nominal is not None:
        from scripts.verify_station_install import accept_nominal
        accept_nominal(read_json(station_nominal))
    verifier_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "tests").is_dir():
        raise ValueError("Run local verification from the source checkout with tests")
    if type(phase_timeout_s) is not int or not 1 <= phase_timeout_s <= 14400:
        raise ValueError("Phase timeout domain1..14400 seconds")
    output = Path(directory).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Verification directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    station_input = None
    if station_nominal is not None:
        station_input = output / "station-nominal-input.json"
        shutil.copyfile(station_nominal, station_input)
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

    if test_receipts is None:
        run(
            "source-tests",
            [sys.executable,"-m","pytest",*chosen,"-q","-p",
                "atomic_model_release.pytest_gate","--atomic-require-executed",
                "--atomic-receipt",output / "tests.json"],
        )
    else:
        from scripts.verify_local_tests import accept_batches
        test_receipts=Path(test_receipts).absolute()
        run("source-tests-batches",[sys.executable,"-m","scripts.verify_local_tests",
            "--output",test_receipts,"--suite",suite,"--accept"])
        tests=accept_batches(test_receipts,suite=suite)
        if tests!=unseal(read_json(test_receipts/'summary.json')):
            raise ValueError('Source test batch summary changed')
        atomic_json(output/'tests.json',tests)
    tests = read_json(output / "tests.json")
    if tests["exit_code"] or tests["counts"]["passed"] == 0:
        raise ValueError("Local tests did not execute successfully")
    station_source = None
    if station_input is not None:
        run("source-station", [sys.executable, "-m", "scripts.verify_station_install",
            "--output", output / "station-source.json", "--nominal-receipt", station_input])
        station_source = unseal(read_json(output / "station-source.json"))["scientific"]
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
        if station_input is not None:
            run("installed-station", [python, "-I", "-m", "scripts.verify_station_install",
                "--output", output / "station-installed.json", "--nominal-receipt", station_input,
                "--require-installed"], cwd=work)
            station_installed = unseal(read_json(output / "station-installed.json"))["scientific"]
            if station_source != station_installed or not station_installed["passed"]:
                raise ValueError("Installed native station results differ from source")
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
        or source_manifest() != expected_tree
        or not evidence_unchanged(expected_evidence)
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
        source_files_sha256=expected_tree,
        evidence_inputs_sha256=expected_evidence,
        station_delivery=dict(accepted=station_source is not None,
            scientific_exact=station_source is not None,
            nominal=station_source['nominal'] if station_source is not None else None,
            new_hour_executed=False),
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
    parser.add_argument("--station-nominal", type=Path,
        help="Explicit prior complete nominal receipt; exact source-bound reuse, no new hour")
    parser.add_argument("--test-receipts",type=Path,
        help="Completed source-bound pytest batches from verify_local_tests")
    args = parser.parse_args(argv)
    output = args.output or str(Path("artifacts") / ("local-verification-" + uuid.uuid4().hex[:10]))
    try:
        options=dict(suite=args.suite,phase_timeout_s=args.phase_timeout_s)
        if args.station_nominal is not None:
            options['station_nominal']=args.station_nominal
        if args.test_receipts is not None:
            options['test_receipts']=args.test_receipts
        verify(output, **options)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
