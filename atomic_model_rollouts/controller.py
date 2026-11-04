"""Synchronous/asynchronous durable scheduler; all spawned workers are joined."""

import copy
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from atomic_model_rollouts import store
from atomic_model_workspace import jobs
from atomic_model_workspace.common import Lock, atomic_json, digest, number, read


def _rss(pid):
    try:
        return next(
            (
                int(s.split()[1]) * 1024
                for s in Path(f"/proc/{pid}/status").read_text().splitlines()
                if s.startswith("VmRSS:")
            ),
            0,
        )
    except (OSError, ValueError):
        return 0


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def _clear_lock(path, pid):
    lock = path / ".lock"
    if lock.exists():
        info = store.process_info(pid)
        if lock.read_text().strip() != str(pid) or (info and info["state"] != "Z"):
            raise ValueError("Unknown/live native lock owner; recovery refused")
        lock.unlink()


def request(folder, kind):
    if kind not in ("pause", "cancel"):
        raise ValueError("Unknown rollout request")
    value = store.load(folder, history=False)["payload"]
    if value["status"] != "running" or value["controller"] is None:
        raise ValueError("No active rollout controller")
    atomic_json(
        Path(folder) / (kind.upper() + ".json"),
        dict(schema_version=1, token=value["controller"]["token"], kind=kind),
    )
    return dict(requested=kind)


class Controller:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        store.load(self.folder)
        self.thread = None
        self.error = None
        self.guard = threading.Lock()

    def snapshot(self):
        return store.load(self.folder, history=False)

    def start(self, *, resume=False, expected_sha256=None, wall_s=None):
        with self.guard:
            if self.thread and self.thread.is_alive():
                raise ValueError("Controller already active")
            ready = threading.Event()
            self.error = None
            self.thread = threading.Thread(
                target=self._drive, args=(ready, resume, expected_sha256, wall_s), daemon=True
            )
            self.thread.start()
            ready.wait()
            if self.error:
                raise ValueError(self.error)
            return self.snapshot()

    def wait(self):
        if self.thread:
            self.thread.join()
        if self.error:
            raise RuntimeError(self.error)
        return self.snapshot()

    def run(self, **kwargs):
        self.start(**kwargs)
        return self.wait()

    def pause(self):
        return request(self.folder, "pause")

    def cancel(self):
        return request(self.folder, "cancel")

    def close(self):
        if self.thread and self.thread.is_alive():
            try:
                self.pause()
            except ValueError:
                pass
            self.thread.join()

    def _commit(self, **changes):
        self.value = store.commit(self.folder, self.value, **changes)
        return self.value

    def _entries(self):
        return copy.deepcopy(self.value["payload"]["entries"])

    def _launch(self, id, cpu):
        p = self.value["payload"]
        cfg = p["config"]
        entry = p["entries"][id]
        if len(entry["attempts"]) >= cfg["max_attempts"]:
            return None
        root = self.folder / "attempts" / id
        root.mkdir(exist_ok=True)
        if len(list(root.iterdir())) >= cfg["max_attempts"] + 4:
            raise ValueError("Orphan preparation limit; inspect before cleanup")
        nonce = secrets.token_hex(16)
        work = root / nonce
        work.mkdir()
        data = dict(
            schema_version=1,
            source=p["source"],
            parent_pid=os.getpid(),
            cpu=cpu,
            job=str(self.folder / "jobs" / id),
            decision_slice=cfg["decision_slice"],
            job_wall_s=cfg["job_wall_s"],
        )
        atomic_json(work / "input.json", data)
        before = store.native(self.folder, id)
        head = read(self.folder / "jobs" / id / "HEAD.json")
        started = time.monotonic()
        log = (work / "worker.log").open("wb")
        env = dict(
            os.environ,
            **{
                name: "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        )
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "atomic_model_rollouts.worker", str(work / "input.json")],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        recorded = False
        try:
            info = store.process_info(process.pid)
            attempt = dict(
                id=nonce,
                pid=process.pid,
                start_ticks=info["start_ticks"] if info else None,
                cpu=cpu,
                outcome="running",
                before_checkpoint_sha256=head["sha256"],
                after_checkpoint_sha256=None,
                before_decisions=before["episode"]["steps"],
                after_decisions=None,
                wall_s=None,
                peak_rss_bytes=None,
                exit_code=None,
                audit_wall_s=None,
                log_truncated=False,
            )
            entries = self._entries()
            entries[id]["status"] = "running"
            entries[id]["reason"] = "spawned"
            entries[id]["attempts"].append(attempt)
            self._commit(entries=entries)
            recorded = True
            atomic_json(work / "GO.json", dict(pid=process.pid, input_sha256=digest(data)))
        except BaseException:
            _stop(process)
            log.close()
            if recorded:
                self._finish(
                    dict(
                        id=id, process=process, started=started, cpu=cpu, work=work, log=log, peak=0
                    ),
                    "launch_handshake_failed",
                )
            raise
        return dict(id=id, process=process, started=started, cpu=cpu, work=work, log=log, peak=0)

    def _finish(self, item, outcome):
        process = item["process"]
        _stop(process)
        item["log"].close()
        id = item["id"]
        cfg = self.value["payload"]["config"]
        elapsed = time.monotonic() - item["started"]
        started = time.monotonic()
        entries = self._entries()
        entry = entries[id]
        attempt = entry["attempts"][-1]
        try:
            _clear_lock(self.folder / "jobs" / id, process.pid)
            native = store.native(self.folder, id)
            if native["status"] not in jobs.TERMINAL and outcome != "exited":
                jobs.interrupt_job(
                    self.folder / "jobs" / id, outcome, cancelled=outcome == "cancelled"
                )
                native = store.native(self.folder, id)
            head = read(self.folder / "jobs" / id / "HEAD.json")
            attempt.update(
                after_checkpoint_sha256=head["sha256"], after_decisions=native["episode"]["steps"]
            )
            entry["accepted_decisions"] = native["episode"]["steps"]
            if native["status"] == "completed":
                status = "completed"
                reason = "time_limit"
            elif native["status"] in ("failed", "rejected"):
                status = "failed"
                reason = native["reason"]
            elif (
                outcome == "exited"
                and process.returncode == 0
                and native["reason"] == "decision_budget"
            ):
                status = "ready"
                reason = "decision_slice"
            else:
                status = "interrupted"
                reason = (
                    outcome if outcome != "exited" else "worker_exit_" + str(process.returncode)
                )
        except (ValueError, OSError, KeyError, TypeError):
            status = "failed"
            reason = "native_validation_or_lock_failure"
        size = (item["work"] / "worker.log").stat().st_size
        if size > cfg["log_limit_bytes"]:
            with (item["work"] / "worker.log").open("r+b") as stream:
                stream.truncate(cfg["log_limit_bytes"])
            attempt["log_truncated"] = True
        attempt.update(
            outcome=outcome,
            wall_s=elapsed,
            peak_rss_bytes=item["peak"],
            exit_code=process.returncode,
            audit_wall_s=time.monotonic() - started,
        )
        entry.update(status=status, reason=reason)
        if status == "ready" and len(entry["attempts"]) >= cfg["max_attempts"]:
            entry.update(status="failed", reason="attempt_budget")
        self._commit(entries=entries)

    def _drive(self, ready, resume, expected_sha, wall):
        active = {}
        owner = None
        started = time.monotonic()
        stop_reason = None
        try:
            with Lock(self.folder):
                self.value = store.load(self.folder)
                p = self.value["payload"]
                cfg = p["config"]
                limit = (
                    cfg["controller_wall_s"]
                    if wall is None
                    else number(wall, 0.001, cfg["controller_wall_s"], "controller wall")
                )
                if (self.folder / "CONTROLLER.json").exists():
                    raise ValueError("Controller lease exists; inspect/recover explicitly")
                if not set(p["cpus"]) <= os.sched_getaffinity(0):
                    raise ValueError("Stored CPU budget is unavailable")
                if p["status"] in ("completed", "completed_with_failures"):
                    ready.set()
                    return
                if p["status"] != "initialized" and (
                    not resume or expected_sha != self.value["sha256"]
                ):
                    raise ValueError("Explicit CAS resume required")
                if p["status"] == "running":
                    raise ValueError("Lost controller must be recovered first")
                process = store.process_info(os.getpid())
                tid = threading.get_native_id()
                thread = store.process_info(os.getpid(), tid)
                owner = dict(
                    pid=os.getpid(),
                    tid=tid,
                    process_start_ticks=process["start_ticks"],
                    thread_start_ticks=thread["start_ticks"],
                    boot_id=store.boot_id(),
                    token=secrets.token_hex(16),
                )
                with (self.folder / "CONTROLLER.json").open("x") as stream:
                    stream.write(json.dumps(owner))
                entries = self._entries()
                if resume:
                    for entry in entries.values():
                        if entry["status"] == "interrupted":
                            entry["status"] = (
                                "ready"
                                if len(entry["attempts"]) < cfg["max_attempts"]
                                else "failed"
                            )
                self._commit(status="running", reason="running", controller=owner, entries=entries)
                for marker in ("PAUSE.json", "CANCEL.json"):
                    (self.folder / marker).unlink(missing_ok=True)
            ready.set()
            cap = min(cfg["workers"], cfg["total_rss_bytes"] // cfg["worker_rss_bytes"])
            while True:
                for kind in ("cancel", "pause"):
                    path = self.folder / (kind.upper() + ".json")
                    if path.exists() and read(path).get("token") == owner["token"]:
                        stop_reason = "cancelled" if kind == "cancel" else "paused"
                if time.monotonic() - started >= limit:
                    stop_reason = stop_reason or "deadline"
                for id, item in list(active.items()):
                    process = item["process"]
                    rss = _rss(process.pid)
                    item["peak"] = max(item["peak"], rss)
                    outcome = (
                        "cancelled"
                        if stop_reason == "cancelled"
                        else "pause_requested"
                        if stop_reason
                        else "worker_deadline"
                        if time.monotonic() - item["started"] >= cfg["job_wall_s"]
                        else "worker_rss_limit"
                        if rss > cfg["worker_rss_bytes"]
                        else "worker_output_limit"
                        if (item["work"] / "worker.log").stat().st_size > cfg["log_limit_bytes"]
                        else "exited"
                        if process.poll() is not None
                        else None
                    )
                    if outcome:
                        self._finish(item, outcome)
                        active.pop(id)
                if (
                    not stop_reason
                    and sum(_rss(item["process"].pid) for item in active.values())
                    > cfg["total_rss_bytes"]
                ):
                    victim = max(active, key=lambda key: _rss(active[key]["process"].pid))
                    self._finish(active.pop(victim), "aggregate_rss_limit")
                if stop_reason:
                    if not active:
                        break
                else:
                    used = {item["cpu"] for item in active.values()}
                    queued = [
                        id
                        for id, entry in self.value["payload"]["entries"].items()
                        if entry["status"] == "ready"
                    ]
                    queued.sort(
                        key=lambda id: (len(self.value["payload"]["entries"][id]["attempts"]), id)
                    )
                    for id in queued:
                        if len(active) >= cap:
                            break
                        cpu = next(c for c in self.value["payload"]["cpus"] if c not in used)
                        try:
                            item = self._launch(id, cpu)
                        except (ValueError, OSError, KeyError, TypeError):
                            entries = self._entries()
                            entries[id].update(status="failed", reason="launch_or_preflight_failed")
                            self._commit(entries=entries)
                            continue
                        if item is None:
                            entries = self._entries()
                            entries[id].update(status="failed", reason="attempt_budget")
                            self._commit(entries=entries)
                        else:
                            active[id] = item
                            used.add(cpu)
                    if not active and not any(
                        e["status"] == "ready" for e in self.value["payload"]["entries"].values()
                    ):
                        break
                time.sleep(0.01)
            statuses = [e["status"] for e in self.value["payload"]["entries"].values()]
            state = (
                "cancelled"
                if stop_reason == "cancelled"
                else "paused"
                if stop_reason or any(s == "interrupted" for s in statuses)
                else "completed"
                if all(s == "completed" for s in statuses)
                else "completed_with_failures"
            )
            self._commit(status=state, reason=stop_reason or state, controller=None)
        except BaseException as exc:
            self.error = type(exc).__name__ + ": " + str(exc)
        finally:
            for item in list(active.values()):
                try:
                    self._finish(item, "controller_interrupted")
                except BaseException:
                    _stop(item["process"])
                    item["log"].close()
            if owner is not None:
                lease = self.folder / "CONTROLLER.json"
                if lease.exists() and read(lease).get("token") == owner["token"]:
                    lease.unlink()
            ready.set()


def recover(folder, *, expected_sha256):
    folder = Path(folder)
    with Lock(folder):
        value = store.load(folder)
        p = value["payload"]
        if (
            value["sha256"] != expected_sha256
            or p["status"] != "running"
            or p["controller"] is None
        ):
            raise ValueError("Stale or non-recoverable rollout")
        owner = p["controller"]
        lease = folder / "CONTROLLER.json"
        if lease.exists() and read(lease) != owner:
            raise ValueError("Unknown controller lease")
        if store.owner_alive(owner):
            raise ValueError("Controller thread is still alive")
        for id, entry in p["entries"].items():
            if entry["status"] != "running":
                continue
            attempt = entry["attempts"][-1]
            info = store.process_info(attempt["pid"])
            if info and info["state"] != "Z":
                raise ValueError("Worker PID exists; no unknown process is stopped")
            lock = folder / "jobs" / id / ".lock"
            if lock.exists() and lock.read_text().strip() != str(attempt["pid"]):
                raise ValueError("Unknown native lock owner")
        entries = copy.deepcopy(p["entries"])
        for id, entry in entries.items():
            if entry["status"] != "running":
                continue
            attempt = entry["attempts"][-1]
            _clear_lock(folder / "jobs" / id, attempt["pid"])
            native = store.native(folder, id)
            if native["status"] not in jobs.TERMINAL:
                jobs.interrupt_job(folder / "jobs" / id, "controller_lost")
            native = store.native(folder, id)
            head = read(folder / "jobs" / id / "HEAD.json")
            attempt.update(
                outcome="recovered_controller_loss",
                after_checkpoint_sha256=head["sha256"],
                after_decisions=native["episode"]["steps"],
            )
            entry.update(
                status="completed"
                if native["status"] == "completed"
                else "failed"
                if native["status"] in ("failed", "rejected")
                else "interrupted",
                reason="controller_lost",
                accepted_decisions=native["episode"]["steps"],
            )
        result = store.commit(
            folder, value, status="paused", reason="recovered", controller=None, entries=entries
        )
        lease.unlink(missing_ok=True)
        return result
