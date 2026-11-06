"""Private worker: owner thread stays alive; no native work before durable GO."""

import ctypes
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl")
    if os.getppid() != request["parent_pid"]:
        raise SystemExit(3)
    os.sched_setaffinity(0, [request["cpu"]])
    go = request_path.parent / "GO.json"
    started = time.monotonic()
    while not go.exists():
        if time.monotonic() - started > 10 or os.getppid() != request["parent_pid"]:
            raise SystemExit(4)
        time.sleep(0.01)
    grant = json.loads(go.read_text())
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if grant != dict(pid=os.getpid(), input_sha256=hashlib.sha256(canonical).hexdigest()):
        raise ValueError("Rollout GO identity mismatch")
    from atomic_model_rollouts import store
    from atomic_model_workspace.jobs import load_job, resume_job, run_job

    if request["source"] != store.source_identity():
        raise ValueError("Rollout worker source changed")
    if load_job(request["job"])["status"] == "cancelled":
        resume_job(request["job"])
    run_job(
        request["job"], max_decisions=request["decision_slice"], deadline_s=request["job_wall_s"]
    )


if __name__ == "__main__":
    main()
