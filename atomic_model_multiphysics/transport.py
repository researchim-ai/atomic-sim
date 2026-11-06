"""Bounded local transport process with retained failure evidence and no shell."""

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from atomic_model_training.spec import atomic_json, exact, integer, read_json

from .provenance import identity, seal, sha, unseal


def admit(spec):
    exact(
        spec,
        {
            "seed",
            "temperature_k",
            "density_kg_m3",
            "nuclides",
            "width_m",
            "energy_bounds_ev_ascending",
            "particles",
            "batches",
            "inactive",
        },
    )
    integer(spec["seed"], 1, 2**31 - 1)
    integer(spec["particles"], 100, 100000)
    integer(spec["batches"], 10, 500)
    integer(spec["inactive"], 1, spec["batches"] - 2)
    if spec["batches"] * spec["particles"] > 2000000:
        raise ValueError("Transport particle budget")
    for key, low, high in [
        ("temperature_k", 294, 900),
        ("density_kg_m3", 100, 12000),
        ("width_m", 0.01, 2.0),
    ]:
        value = spec[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Transport parameter out of domain: " + key)
    exact(spec["nuclides"], {"U235", "U238", "O16", "H1"})
    if (
        any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in spec["nuclides"].values()
        )
        or abs(sum(spec["nuclides"].values()) - 1) > 1e-12
    ):
        raise ValueError("Nuclide normalization")
    if spec["energy_bounds_ev_ascending"] != [1e-5, 0.625, 2e7]:
        raise ValueError("Only declared two-group partition supported")
    return json.loads(json.dumps(spec, allow_nan=False))


def run(spec, directory, *, deadline_s=300.0, cancelled=None):
    spec = admit(spec)
    if (
        type(deadline_s) not in (int, float)
        or not math.isfinite(deadline_s)
        or not 0 < deadline_s <= 600
    ):
        raise ValueError("Transport deadline domain")
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("Symlink transport directory")
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    atomic_json(request, spec)
    cancelled = cancelled or threading.Event()
    start = time.monotonic()
    with (directory / "worker.log").open("xb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "atomic_model_multiphysics.openmc_worker",
                str(directory),
                str(request),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        )
        status = "failed"
        try:
            while process.poll() is None:
                if cancelled.is_set():
                    status = "cancelled"
                    break
                if time.monotonic() - start > deadline_s:
                    status = "deadline"
                    break
                time.sleep(0.02)
            else:
                status = "completed" if process.returncode == 0 else "failed"
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
    receipt = dict(
        status=status,
        returncode=process.returncode,
        wall_s=time.monotonic() - start,
        source=identity(),
        worker_sha256=sha(Path(__file__).with_name("openmc_worker.py")),
        log_sha256=sha(directory / "worker.log"),
        validated=False,
        plant_referenced=False,
    )
    if status == "completed":
        result = unseal(read_json(directory / "mgxs.json"))
        if result["specification"] != spec:
            raise ValueError("Transport request/output mismatch")
        receipt["result_sha256"] = sha(directory / "mgxs.json")
        for name, expected in result["retained"].items():
            if Path(name).name != name or sha(directory / name) != expected:
                raise ValueError("Transport retained artifact differs")
    atomic_json(directory / "execution.json", seal(receipt))
    return receipt
