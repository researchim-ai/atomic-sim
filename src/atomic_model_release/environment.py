"""Dependency-free environment/recovery inspection; never changes an existing lock."""

import hashlib
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path


def doctor(folder=None):
    packages = {}
    for name in ("atomic-sim", "numpy", "scipy", "gymnasium", "torch", "pyarrow", "pytest", "mypy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = None
    if folder is not None:
        path = Path(folder) / ".lock"
        if path.exists():
            with path.open("r") as stream:
                text = stream.read(32).strip()
            pid = int(text) if text.isdecimal() else None
            lock = dict(
                pid=pid,
                pid_present=(Path("/proc") / str(pid)).exists() if pid else None,
                removed=False,
                advice="PID presence alone cannot prove ownership or exclude PID reuse; no automatic stale-lock removal",
            )
    golden = Path(__file__).parent / "golden"
    resources = {
        p.relative_to(golden).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(golden.rglob("*.json"))
    }
    return dict(
        schema_version=1,
        release_tools_version="1.0.0",
        python=sys.version.split()[0],
        platform=sys.platform,
        packages=packages,
        resources=resources,
        linux_agent_process_supported=sys.platform == "linux" and hasattr(os, "sched_setaffinity"),
        openmc_found=shutil.which("openmc") is not None,
        mpi_launcher_found=shutil.which("mpiexec") is not None,
        external_runtime_qualified=False,
        lock=lock,
        dependency_scope="Observed versions only; not a lock or supported-version matrix",
    )
