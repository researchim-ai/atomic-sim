"""Disjoint CI lanes and fail-closed preflights, not physical qualification receipts."""

import importlib.util
import os
import shutil
from pathlib import Path

LANES = ("portable", "long", "openmc", "mpi", "hardware")


def classify(path, name):
    file = Path(str(path)).name
    if name == "test_openmc_particle_run_reports_k_and_mgxs":
        return "openmc"
    if file == "test_batch.py" and name == "test_performance":
        return "hardware"
    if file in {
        "test_simulator.py",
        "test_integration.py",
        "test_transient_convergence.py",
        "test_muscl_total_initialization.py",
    }:
        return "long"
    return "portable"


def preflight(lane):
    if lane not in LANES:
        raise ValueError("Unknown release lane")
    missing = []
    if lane == "openmc":
        if shutil.which("openmc") is None:
            missing.append("openmc executable")
        raw = os.environ.get("OPENMC_CROSS_SECTIONS")
        if not raw or not Path(raw).is_file():
            missing.append("OPENMC_CROSS_SECTIONS file")
        if not raw or not Path(str(raw) + ".license.json").is_file():
            missing.append("cross-section license sidecar")
    if lane == "mpi":
        if shutil.which("mpiexec") is None:
            missing.append("mpiexec executable")
        if importlib.util.find_spec("mpi4py") is None:
            missing.append("mpi4py import")
    if lane == "hardware" and not os.environ.get("ATOMIC_HARDWARE_PROFILE"):
        missing.append("ATOMIC_HARDWARE_PROFILE identity")
    if missing:
        raise ValueError("Lane prerequisites missing: " + ", ".join(missing))
    return dict(
        schema_version=1, lane=lane, prerequisites_present=True, executed=False, qualified=False
    )
