"""Local trusted worker supervisor with pinned inputs and bounded attempts.

Wall/RSS limits are process supervision, not a sandbox for untrusted policies.
"""

import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from atomic_model_study import evaluate, store
from atomic_model_workspace.common import Lock, atomic_json, digest, read


class Manager:
    def __init__(self, folder, *, rss_limit_bytes=1024**3):
        if sys.platform != "linux":
            raise ValueError("Study worker supervision requires Linux")
        if type(rss_limit_bytes) is not int or not 32 * 1024**2 <= rss_limit_bytes <= 16 * 1024**3:
            raise ValueError("RSS limit must be 32 MiB..16 GiB")
        self.folder = Path(folder).resolve()
        store.load(self.folder)
        self.rss_limit = rss_limit_bytes
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.closed = False
        self.thread = None

    def _write(self, id, state):
        atomic_json(evaluate.evaluation_path(self.folder, id) / "status.json", state)

    def start(self, binding_id, *, expected_revision, expected_checkpoint_sha256):
        with self.lock, Lock(self.folder):
            self._available()
            value, _, _ = store.load(self.folder)
            # Evaluation does not consume a model revision budget.
            if (
                type(expected_revision) is not int
                or expected_revision != value["payload"]["revision"]
                or expected_checkpoint_sha256 != value["sha256"]
            ):
                raise ValueError("Stale study revision/content identity")
            if binding_id not in value["payload"]["references"]:
                raise ValueError("Unknown reference binding")
            id = secrets.token_hex(16)
            path = evaluate.evaluation_path(self.folder, id)
            path.mkdir()
            (path / "attempts").mkdir()
            p = dict(
                schema_version=1,
                study_checkpoint_sha256=value["sha256"],
                study=value["payload"],
                binding_id=binding_id,
                source=store.source_identity(),
                cpus=[min(os.sched_getaffinity(0))],
                wall_limit_s=value["payload"]["project"]["limits"]["job_timeout_s"],
                rss_limit_bytes=self.rss_limit,
                parent_pid=os.getpid(),
            )
            atomic_json(path / "input.json", dict(payload=p, sha256=digest(p)))
            self._launch(id, p, 0)
            return self.status(id)

    def _available(self):
        if self.closed or self.active is not None:
            raise ValueError("Supervisor closed or already running an evaluation")
        # A second supervisor may inspect the store, but must not launch while
        # another worker owns its active marker.
        if (self.folder / "EVALUATION.lock").exists():
            raise ValueError("Another evaluation owns the study worker lock")

    def _launch(self, id, p, attempt):
        path = evaluate.evaluation_path(self.folder, id)
        work = path / "attempts" / str(attempt)
        work.mkdir()
        atomic_json(work / "owner.json", dict(parent_pid=os.getpid()))
        marker = self.folder / "EVALUATION.lock"
        with marker.open("x") as out:
            out.write(id)
        env = dict(os.environ)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        state = dict(
            schema_version=1,
            evaluation_id=id,
            study_revision=p["study"]["revision"],
            study_checkpoint_sha256=p["study_checkpoint_sha256"],
            binding_id=p["binding_id"],
            status="running",
            attempt=attempt,
            report=None,
            comparison_passed=None,
        )
        ready = threading.Event()
        failures = []

        def supervise():
            # Linux PDEATHSIG follows the spawning thread. Keep it alive for
            # the entire worker lifetime, even after an HTTP request returns.
            try:
                with (work / "worker.log").open("wb") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "atomic_model_study.worker",
                            str(self.folder),
                            id,
                            str(attempt),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=True,
                    )
                self.process, self.active = process, id
                state["pid"] = process.pid
                self._write(id, state)
            except BaseException:
                marker.unlink(missing_ok=True)
                failures.append(sys.exc_info()[1])
            ready.set()
            if not failures:
                self._monitor(id, p, state)

        self.thread = threading.Thread(target=supervise, daemon=True)
        self.thread.start()
        ready.wait()
        if failures:
            raise failures[0]

    def _monitor(self, id, p, state):
        process = self.process
        started = time.monotonic()
        reason = None
        peak = 0
        while process.poll() is None:
            with self.lock:
                if self.closed or state.get("cancel_requested"):
                    reason = "cancelled"
                if time.monotonic() - started > p["wall_limit_s"]:
                    reason = reason or "timed_out"
                try:
                    lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    rss = next(
                        (int(s.split()[1]) * 1024 for s in lines if s.startswith("VmRSS:")), 0
                    )
                    peak = max(peak, rss)
                    if rss > p["rss_limit_bytes"]:
                        reason = reason or "rss_limit"
                except (OSError, ValueError):
                    pass  # Process exit races with /proc reads.
                if reason:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            time.sleep(0.025)
        process.wait()
        with self.lock:
            state.update(
                status=reason or "failed",
                exit_code=process.returncode,
                peak_rss_bytes=peak,
                wall_s=time.monotonic() - started,
            )
            path = evaluate.evaluation_path(self.folder, id)
            if reason is None and process.returncode == 0:
                try:
                    result = read(path / "attempts" / str(state["attempt"]) / "result.json")
                    if (
                        result["study_checkpoint_sha256"] != p["study_checkpoint_sha256"]
                        or result["binding_id"] != p["binding_id"]
                        or result["attempt"] != state["attempt"]
                    ):
                        raise ValueError("Worker result identity mismatch")
                    state.update(result)
                    state["status"] = (
                        "completed"
                        if result["report"]
                        else result["job_status"]
                        if result["job_status"] in ("rejected", "failed")
                        else "paused"
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    state.update(status="failed", error=str(exc)[:512])
            # Only this joined worker can leave this native job lock behind.
            joblock = path / "job/.lock"
            if joblock.exists():
                try:
                    if joblock.read_text().strip() == str(process.pid):
                        joblock.unlink()
                except OSError:
                    pass
            self._write(id, state)
            marker = self.folder / "EVALUATION.lock"
            if marker.exists() and marker.read_text() == id:
                marker.unlink()
            self.active = None

    def status(self, id):
        with self.lock:
            state = read(evaluate.evaluation_path(self.folder, id) / "status.json")
            current, _, _ = store.load(self.folder)
            return dict(state, stale=current["sha256"] != state["study_checkpoint_sha256"])

    def cancel(self, id):
        with self.lock:
            if self.active != id or self.process.poll() is not None:
                raise ValueError("Evaluation is not owned/running")
            # The monitor shares this state by reference through its arguments.
            self.closed = True
        self.thread.join()
        with self.lock:
            self.closed = False
        return self.status(id)

    def resume(self, id):
        with self.lock, Lock(self.folder):
            self._available()
            state = self.status(id)
            if state["stale"] or state["status"] not in (
                "paused",
                "cancelled",
                "timed_out",
                "rss_limit",
                "failed",
                "interrupted",
            ):
                raise ValueError("Only interrupted current-revision evaluations can resume")
            if state["attempt"] >= 31:
                raise ValueError("Evaluation attempt limit")
            p, _ = evaluate.load_input(self.folder, id)
            self._launch(id, p, state["attempt"] + 1)
            return self.status(id)

    def recover(self, id):
        """Explicit recovery only after proving the recorded worker is dead.

        A reused/live PID is conservatively refused; no unknown process is killed.
        """
        with self.lock, Lock(self.folder):
            if self.active is not None or self.closed:
                raise ValueError("Cannot recover an owned active supervisor")
            evaluate.load_input(self.folder, id)
            path = evaluate.evaluation_path(self.folder, id)
            state = read(path / "status.json")
            marker = self.folder / "EVALUATION.lock"
            if state["status"] != "running" or not marker.exists() or marker.read_text() != id:
                raise ValueError("No matching interrupted worker ownership")
            pid = state.get("pid")
            if type(pid) is not int or pid <= 0:
                raise ValueError("Unknown worker identity")
            proc = Path(f"/proc/{pid}/stat")
            try:
                # A zombie has exited and cannot write the accepted checkpoint.
                if proc.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    raise ValueError("Recorded PID still exists; recovery refused")
            except FileNotFoundError:
                pass
            joblock = path / "job/.lock"
            if joblock.exists():
                if joblock.read_text().strip() != str(pid):
                    raise ValueError("Native job lock belongs to an unknown writer")
                joblock.unlink()
            state.update(status="interrupted", reason="supervisor lost; dead worker verified")
            self._write(id, state)
            marker.unlink()
            return self.status(id)

    def wait(self, id):
        thread = self.thread
        if thread is not None:
            thread.join()
        return self.status(id)

    def close(self):
        with self.lock:
            self.closed = True
            thread = self.thread
        if thread is not None:
            thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
