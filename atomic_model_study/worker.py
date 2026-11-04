"""Private trusted evaluation process; no executable/model plug-ins from input."""

import argparse
import ctypes
import os
import signal
from pathlib import Path


def main():
    from atomic_model_workspace.common import read

    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("id")
    p.add_argument("attempt", type=int)
    a = p.parse_args()
    if len(a.id) != 32 or any(c not in "0123456789abcdef" for c in a.id) or not 0 <= a.attempt < 32:
        raise ValueError("Invalid private worker arguments")
    raw = read(a.folder / "evaluations" / a.id / "input.json")["payload"]
    if os.name != "posix" or not Path("/proc/self/status").exists():
        raise ValueError("Evaluation supervision requires Linux")
    # Parent-death termination prevents orphan computation when the UI/controller dies.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl parent-death signal")
    owner = read(a.folder / "evaluations" / a.id / "attempts" / str(a.attempt) / "owner.json")
    if os.getppid() != owner["parent_pid"]:
        raise SystemExit(3)
    os.sched_setaffinity(0, raw["cpus"])
    from atomic_model_study.evaluate import run

    run(a.folder, a.id, a.attempt)


if __name__ == "__main__":
    main()
