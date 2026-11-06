"""Bounded Linux process transport for trusted host commands, not a security sandbox."""

import hashlib
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from atomic_model_workspace.common import canonical, digest, loads

MAX_STDOUT = 64 * 1024
MAX_STDERR = 8 * 1024
MAX_REQUEST = 256 * 1024


class AgentFailure(ValueError):
    def __init__(self, reason, telemetry):
        super().__init__(reason)
        self.reason, self.telemetry = reason, telemetry


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def launch_identity(command, identity_files=()):
    if sys.platform != "linux":
        raise ValueError("Agent process supervision requires Linux")
    if (
        not isinstance(command, list)
        or not 1 <= len(command) <= 64
        or any(not isinstance(a, str) or "\0" in a for a in command)
        or sum(len(a) for a in command) > 4096
    ):
        raise ValueError("Invalid host argv")
    executable = shutil.which(command[0])
    if executable is None:
        raise ValueError("Agent executable unavailable")
    files = list(identity_files)
    if len(command) > 1 and not command[1].startswith("-") and Path(command[1]).is_file():
        files.append(command[1])
    if len(files) > 16:
        raise ValueError("Agent identity file limit")
    hashes = []
    for file in files:
        with Path(file).open("rb") as stream:
            raw = stream.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise ValueError("Agent identity file size limit")
        hashes.append(dict(name=Path(file).name, sha256=hashlib.sha256(raw).hexdigest()))
    return dict(
        argv_sha256=digest(command),
        executable_sha256=_file_hash(executable),
        files=hashes,
        dependency_scope="argv/executable/declared entry files only; external dependencies and services not locked",
    )


def exchange(command, request, *, timeout_s, rss_limit_bytes, cancelled=lambda: False):
    raw = (canonical(request) + "\n").encode()
    if len(raw) > MAX_REQUEST:
        raise ValueError("agent_request_limit")
    started = time.monotonic()
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "atomic_model_agents.launcher",
            str(os.getpid()),
            json.dumps(command),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    offset = 0
    peak = 0
    reason = None
    selector = selectors.DefaultSelector()
    for stream, kind, event in [
        (process.stdin, "stdin", selectors.EVENT_WRITE),
        (process.stdout, "stdout", selectors.EVENT_READ),
        (process.stderr, "stderr", selectors.EVENT_READ),
    ]:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, event, kind)
    try:
        while selector.get_map() or process.poll() is None:
            if cancelled():
                reason = "agent_cancelled"
                break
            if time.monotonic() - started >= timeout_s:
                reason = "agent_timeout"
                break
            try:
                rss = next(
                    (
                        int(line.split()[1]) * 1024
                        for line in Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        if line.startswith("VmRSS:")
                    ),
                    0,
                )
                peak = max(peak, rss)
                if rss > rss_limit_bytes:
                    reason = "agent_rss_limit"
                    break
            except (OSError, ValueError):
                pass
            for key, event in selector.select(timeout=0.01):
                stream = key.fileobj
                kind = key.data
                if kind == "stdin":
                    try:
                        n = os.write(stream.fileno(), raw[offset : offset + 4096])
                        offset += n
                    except BrokenPipeError:
                        n = 0
                        offset = len(raw)
                    if offset == len(raw):
                        selector.unregister(stream)
                        stream.close()
                else:
                    try:
                        chunk = os.read(stream.fileno(), 4096)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    outputs[kind].extend(chunk)
                    limit = MAX_STDOUT if kind == "stdout" else MAX_STDERR
                    if len(outputs[kind]) > limit:
                        reason = "agent_" + kind + "_limit"
                        break
            if reason:
                break
    finally:
        # Also terminate same-session descendants retaining inherited pipe FDs.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    telemetry = dict(
        wall_s=time.monotonic() - started,
        peak_rss_bytes=peak,
        exit_code=process.returncode,
        stdout_bytes=len(outputs["stdout"]),
        stderr_bytes=len(outputs["stderr"]),
        stdout_sha256=hashlib.sha256(outputs["stdout"]).hexdigest(),
        stderr_sha256=hashlib.sha256(outputs["stderr"]).hexdigest(),
    )
    if reason is None and telemetry["wall_s"] > timeout_s:
        reason = "agent_timeout"
    if reason is None and process.returncode != 0:
        reason = "agent_exit"
    if reason is None:
        try:
            lines = bytes(outputs["stdout"]).splitlines()
            if len(lines) != 1:
                raise ValueError("Expected one JSONL response")
            value = loads(lines[0])
        except (ValueError, UnicodeError, RecursionError, TypeError):
            reason = "agent_jsonl"
    if reason:
        raise AgentFailure(reason, telemetry)
    return value, telemetry
