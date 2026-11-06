"""Isolated sync/async campaigns with bounded IPC and transactional progress."""

import concurrent.futures
import json
import math
import os
import platform
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from atomic_model_workspace.common import digest

from .episode import identity
from .spec import admit_campaign, atomic_json, episode_seed, integer, read_json

MAX_MESSAGE = 2 * 1024 * 1024
MAX_EPISODE_OUTPUT = 64 * 1024 * 1024


def envelope(value):
    return dict(payload=value, sha256=digest(value))


def unpack(path):
    value = read_json(path, limit=MAX_EPISODE_OUTPUT)
    if set(value) != {"payload", "sha256"} or digest(value["payload"]) != value["sha256"]:
        raise ValueError("Corrupt campaign artifact")
    return value["payload"]


def stop(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
    process.stdin.close()
    process.stdout.close()


def rss_mb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return 0
    return 0


def cpu_seconds(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except FileNotFoundError:
        return 0.0


class Campaign:
    """A local research runner, not an OS sandbox for arbitrary Python policies.

    Only this package's own worker executable runs. Policies are admitted JSON
    command lists. The caller owns the output directory; concurrent writers are
    refused using an exclusive lock. Cancellation preserves accepted progress.
    """

    def __init__(
        self, cases, directory, *, workers=1, memory_mb=512, deadline_s=60.0, cpu_s=30, root_seed=0
    ):
        if platform.system() != "Linux":
            raise ValueError("Resource-qualified runner requires Linux")
        self.cases = admit_campaign(cases)
        self.workers = integer(workers, 1, 8)
        self.memory_mb = integer(memory_mb, 128, 8192)
        self.cpu_s = integer(cpu_s, 1, 300)
        self.root_seed = integer(root_seed, 0, 2**32 - 1)
        if (
            type(deadline_s) not in (int, float)
            or not math.isfinite(deadline_s)
            or not 0 < deadline_s <= 3600
        ):
            raise ValueError("Deadline domain (0,3600] seconds per episode")
        self.deadline_s = float(deadline_s)
        self.directory = Path(directory).absolute()
        if any(p.is_symlink() for p in (self.directory, *self.directory.parents)):
            raise ValueError("Symlink output path is not admitted")
        self.cancelled = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()

    def cancel(self):
        self.cancelled.set()

    def start(self, *, resume=False):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self.run, resume=resume)
        future.add_done_callback(lambda _: pool.shutdown(wait=False))
        return future

    def _start_worker(self):
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONHASHSEED="0",
        )
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "atomic_model_training.worker",
                str(self.memory_mb),
                str(self.cpu_s),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
            bufsize=0,
        )

    def _episode(self, process, case, saved):
        started = time.monotonic()
        peak = 0.0
        initial_cpu = cpu_seconds(process.pid)
        request = json.dumps(dict(case=case, saved=saved), allow_nan=False).encode() + b"\n"
        if len(request) > MAX_MESSAGE:
            raise ValueError("Worker request exceeds budget")
        os.set_blocking(process.stdin.fileno(), False)
        offset = 0
        with selectors.DefaultSelector() as writer:
            writer.register(process.stdin, selectors.EVENT_WRITE)
            while offset < len(request):
                if self.cancelled.is_set() or time.monotonic() - started >= self.deadline_s:
                    return dict(
                        id=case["id"],
                        status="cancelled" if self.cancelled.is_set() else "deadline",
                        progress=saved,
                    )
                if writer.select(timeout=0.02):
                    try:
                        offset += os.write(process.stdin.fileno(), request[offset:])
                    except BlockingIOError:
                        pass
        buffer = b""
        total = 0
        progress = saved
        status, receipt, error = "worker_failed", None, None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                peak = max(peak, rss_mb(process.pid))
                if self.cancelled.is_set():
                    status = "cancelled"
                    break
                if time.monotonic() - started >= self.deadline_s:
                    status = "deadline"
                    break
                if cpu_seconds(process.pid) - initial_cpu > self.cpu_s:
                    status = "cpu_limit"
                    break
                if peak > self.memory_mb:
                    status = "memory_limit"
                    break
                if not selector.select(timeout=0.02):
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                total += len(chunk)
                buffer += chunk
                if total > MAX_EPISODE_OUTPUT:
                    status = "output_limit"
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > MAX_MESSAGE:
                        raise ValueError("Worker message exceeds budget")
                    message = json.loads(line)
                    if message["kind"] == "progress":
                        progress = message["receipt"]
                        self._validate_receipt(case, progress)
                        atomic_json(
                            self.directory / (case["id"] + ".progress.json"), envelope(progress)
                        )
                    elif message["kind"] == "result":
                        receipt = message["receipt"]
                        self._validate_receipt(case, receipt, completed=True)
                        status = "completed"
                        return dict(
                            id=case["id"],
                            status=status,
                            receipt=receipt,
                            seed=episode_seed(self.root_seed, case),
                            peak_rss_mb=peak,
                            wall_s=time.monotonic() - started,
                        )
                    elif message["kind"] == "failure":
                        status, error = "episode_failed", message["error"]
                        break
                    else:
                        raise ValueError("Unknown worker frame")
                else:
                    if len(buffer) > MAX_MESSAGE:
                        raise ValueError("Worker unterminated message exceeds budget")
                    continue
                break
        return dict(
            id=case["id"],
            status=status,
            error=error,
            progress=progress,
            seed=episode_seed(self.root_seed, case),
            peak_rss_mb=peak,
            wall_s=time.monotonic() - started,
        )

    def _validate_receipt(self, case, receipt, completed=False):
        if receipt["case_sha256"] != digest(case) or receipt["source_sha256"] != self.source:
            raise ValueError("Worker source or case differs")
        cp = receipt["checkpoint"]
        if digest(cp["payload"]) != cp["sha256"]:
            raise ValueError("Worker checkpoint corrupt")
        if completed and (
            receipt["status"] != "completed" or cp["payload"]["revision"] != len(case["steps"])
        ):
            raise ValueError("Incomplete episode cannot be promoted")

    def _slot(self, tasks):
        process, uses, results = None, 0, []
        try:
            for case, saved in tasks:
                if self.cancelled.is_set():
                    result = dict(
                        id=case["id"],
                        status="cancelled",
                        progress=saved,
                        seed=episode_seed(self.root_seed, case),
                    )
                else:
                    if process is None or uses >= 4 or process.poll() is not None:
                        stop(process)
                        process, uses = self._start_worker(), 0
                    try:
                        result = self._episode(process, case, saved)
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        result = dict(
                            id=case["id"],
                            status="worker_failed",
                            error=str(exc)[:1000],
                            progress=saved,
                        )
                    uses += 1
                    if result["status"] != "completed":
                        stop(process)
                        process = None
                atomic_json(self.directory / (case["id"] + ".result.json"), envelope(result))
                results.append(result)
        finally:
            stop(process)
        return results

    def run(self, *, resume=False):
        with self._start_lock:
            if self._started:
                raise ValueError("Campaign instances are single-use")
            self._started = True
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self.directory / ".runner.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            self.source = digest(identity())
            manifest = dict(
                schema_version=1,
                source_sha256=self.source,
                cases=self.cases,
                root_seed=self.root_seed,
                fidelity="reduced_single_loop_core",
            )
            path = self.directory / "manifest.json"
            if resume:
                if unpack(path) != manifest:
                    raise ValueError("Resume manifest/source/cases differ")
            else:
                if path.exists():
                    raise ValueError("Existing campaign requires explicit resume")
                atomic_json(path, envelope(manifest))
            pending, results = [], []
            for case in self.cases:
                result_path = self.directory / (case["id"] + ".result.json")
                progress_path = self.directory / (case["id"] + ".progress.json")
                if resume and result_path.exists():
                    result = unpack(result_path)
                    if result["status"] == "completed":
                        self._validate_receipt(case, result["receipt"], completed=True)
                        results.append(result)
                        continue
                saved = unpack(progress_path) if resume and progress_path.exists() else None
                if saved is not None:
                    self._validate_receipt(case, saved)
                pending.append((case, saved))
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [
                    pool.submit(self._slot, pending[i :: self.workers]) for i in range(self.workers)
                ]
                try:
                    for future in futures:
                        results.extend(future.result())
                except BaseException:
                    self.cancel()
                    raise
            report = dict(
                manifest_sha256=digest(manifest),
                results=sorted(results, key=lambda r: r["id"]),
                completed=all(r["status"] == "completed" for r in results),
                validated=False,
                plant_referenced=False,
            )
            atomic_json(self.directory / "report.json", envelope(report))
            return report
        finally:
            os.close(fd)
            lock.unlink()
