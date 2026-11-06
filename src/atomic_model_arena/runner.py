"""Fixed CPU worker, train-only selection and self-verifying benchmark artifacts."""

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from atomic_model_agents import jobs, transport
from atomic_model_arena import metrics, protocol
from atomic_model_workspace.common import (
    atomic_json,
    canonical,
    digest,
    keys,
    local_file,
    number,
    read,
)

MAX_FILES = 4096
MAX_TOTAL = 512 * 1024**2
MAX_FILE = 32 * 1024**2


def execute(pack, commands, output, cpus):
    """Private worker: imports/native work happen after worker affinity setup."""
    output = Path(output)
    if protocol.normalize(pack) != pack:
        raise ValueError("Noncanonical arena protocol")
    launch = {id: transport.launch_identity(command) for id, command in commands.items()}
    machine = protocol.hardware()
    if machine["cpu_affinity"] != cpus or any(
        v != "1" for v in machine["execution_environment"]["threads"].values()
    ):
        raise ValueError("Arena CPU/thread budget not applied")
    atomic_json(output / "hardware.json", machine)
    atomic_json(output / "launch.json", launch)
    records = []
    states = {}
    work = output / "work"
    work.mkdir()

    def phase(name, selected=None):
        for entry in protocol.schedule(pack, name, selected):
            id = entry["task_id"]
            folder = work / id
            started = time.monotonic()
            jobs.create(
                pack["cases"][entry["case_id"]],
                pack["agents"][entry["agent_id"]],
                folder,
                commands[entry["agent_id"]],
            )
            value = jobs.run(
                folder,
                commands[entry["agent_id"]],
                max_decisions=pack["budgets"]["max_decisions"],
                deadline_s=pack["budgets"]["job_wall_s"],
            )
            wall = time.monotonic() - started
            state = value["payload"]
            if state["launch"] != launch[entry["agent_id"]]:
                raise ValueError("Arena agent identity changed between episodes")
            artifact = output / "episodes" / id
            jobs.export(folder, artifact)
            states[id] = state
            records.append(
                dict(
                    **entry,
                    job_path="episodes/" + id,
                    checkpoint_sha256=value["sha256"],
                    observed_wall_s=wall,
                    metrics=metrics.evaluate(state, pack),
                )
            )
            atomic_json(
                output / "progress.json",
                dict(schema_version=1, finished=len(records), last_task=id, phase=name),
            )

    phase("train")
    selection = metrics.select(pack, records)
    atomic_json(output / "selection.json", selection)  # Freeze before any dev/test execution.
    if selection["selected_agent"]:
        phase("dev", selection["selected_agent"])
        phase("test", selection["selected_agent"])
    atomic_json(output / "tasks.json", records)
    report = metrics.report(pack, records, selection, machine)
    atomic_json(output / "report.json", report)
    (output / "transitions.jsonl").write_bytes(metrics.transitions(pack, records, states))
    shutil.rmtree(work)
    (output / "progress.json").unlink()
    files = sorted(p for p in output.rglob("*") if p.is_file())
    if (
        len(files) > MAX_FILES
        or any(p.stat().st_size > MAX_FILE for p in files)
        or sum(p.stat().st_size for p in files) > MAX_TOTAL
    ):
        raise ValueError("Arena artifact size limit")
    atomic_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            source=protocol.source_identity(),
            artifacts={
                p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            validated=False,
            plant_referenced=False,
        ),
    )
    verify(output)
    return dict(
        schema_version=1,
        episodes=len(records),
        report_sha256=digest(report),
        status=report["status"],
        validated=False,
        plant_referenced=False,
    )


def run(spec, commands, output, *, cpu=None):
    if sys.platform != "linux":
        raise ValueError("Arena execution requires Linux")
    pack = protocol.normalize(spec)
    keys(commands, pack["agents"])
    for command in commands.values():
        transport.launch_identity(command)
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if cpu is None else cpu
    if type(cpu) is not int or cpu not in allowed:
        raise ValueError("CPU is outside caller affinity")
    output = Path(output).resolve()
    request = dict(schema_version=1, pack=pack, commands=commands, output=str(output), cpus=[cpu])
    if len((canonical(request) + "\n").encode()) > transport.MAX_REQUEST:
        raise ValueError("Arena worker request size limit")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "protocol.json", pack)
    try:
        receipt, cost = transport.exchange(
            [sys.executable, "-m", "atomic_model_arena.worker"],
            request,
            timeout_s=pack["budgets"]["total_wall_s"],
            rss_limit_bytes=pack["budgets"]["worker_rss_bytes"],
        )
        keys(
            receipt,
            (
                "schema_version",
                "episodes",
                "report_sha256",
                "status",
                "validated",
                "plant_referenced",
            ),
        )
        if receipt["report_sha256"] != digest(read(output / "report.json")):
            raise ValueError("Arena worker receipt mismatch")
    except (transport.AgentFailure, ValueError, OSError) as exc:
        atomic_json(
            output / "controller_failure.json",
            dict(
                schema_version=1,
                reason=exc.reason
                if isinstance(exc, transport.AgentFailure)
                else "arena_worker_or_receipt_failure",
                telemetry=getattr(exc, "telemetry", {}),
            ),
        )
        raise
    # Controller observations are outside the immutable benchmark manifest and
    # returned to the caller, not silently mixed into an already published report.
    return dict(**receipt, controller_cost=cost, output=str(output))


def verify(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(manifest, ("schema_version", "source", "artifacts", "validated", "plant_referenced"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source"] != protocol.source_identity()
        or manifest["validated"] is not False
        or manifest["plant_referenced"] is not False
        or not isinstance(manifest["artifacts"], dict)
        or not 6 <= len(manifest["artifacts"]) <= MAX_FILES
    ):
        raise ValueError("Arena schema/source/inventory mismatch")
    total = local_file(folder, "manifest.json").stat().st_size
    for name, sha in manifest["artifacts"].items():
        if name == "manifest.json":
            raise ValueError("Arena self-inventory")
        with local_file(folder, name).open("rb") as stream:
            raw = stream.read(MAX_FILE + 1)
        total += len(raw)
        if len(raw) > MAX_FILE or total > MAX_TOTAL or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Arena artifact hash/size limit")
    pack = protocol.normalize(read(local_file(folder, "protocol.json")))
    if pack != read(folder / "protocol.json"):
        raise ValueError("Noncanonical arena pack")
    machine = read(local_file(folder, "hardware.json"))
    protocol.validate_hardware(machine)
    launch = read(local_file(folder, "launch.json"))
    keys(launch, pack["agents"])
    records = read(local_file(folder, "tasks.json"))
    if not isinstance(records, list) or len(records) > 384:
        raise ValueError("Arena task count limit")
    expected_files = {
        "protocol.json",
        "hardware.json",
        "launch.json",
        "selection.json",
        "tasks.json",
        "report.json",
        "transitions.jsonl",
    }
    states = {}
    derived = []
    for entry in records:
        keys(
            entry,
            (
                "case_id",
                "agent_id",
                "repeat",
                "task_id",
                "phase",
                "job_path",
                "checkpoint_sha256",
                "observed_wall_s",
                "metrics",
            ),
        )
        number(entry["observed_wall_s"], 0, 1e9, "observed_wall_s")
        if (
            entry["job_path"] != "episodes/" + entry["task_id"]
            or len(entry["task_id"]) != 64
            or any(c not in "0123456789abcdef" for c in entry["task_id"])
        ):
            raise ValueError("Arena task path")
        if entry["task_id"] in states:
            raise ValueError("Duplicate arena task")
        path = local_file(folder, entry["job_path"] + "/manifest.json").parent
        jobs.replay(path)  # Native replay; no external policy invocation.
        head = read(path / "HEAD.json")
        value = read(local_file(path, head["checkpoint"]))
        state = value["payload"]
        if entry["case_id"] not in pack["cases"] or entry["agent_id"] not in pack["agents"]:
            raise ValueError("Arena task case/agent")
        if (
            state["scenario"] != pack["cases"][entry["case_id"]]
            or state["agent"] != pack["agents"][entry["agent_id"]]
            or state["launch"] != launch[entry["agent_id"]]
            or value["sha256"] != entry["checkpoint_sha256"]
        ):
            raise ValueError("Arena task protocol/launch binding mismatch")
        expected_files.update(
            entry["job_path"] + "/" + name
            for name in [*read(path / "manifest.json")["artifacts"], "manifest.json"]
        )
        state_metrics = metrics.evaluate(state, pack)
        if state_metrics != entry["metrics"]:
            raise ValueError("Arena task metric recomputation mismatch")
        derived.append(dict(entry, metrics=state_metrics))
        states[entry["task_id"]] = state
    selection = metrics.select(pack, derived)
    expected_schedule = (
        protocol.schedule(pack, "train")
        + protocol.schedule(pack, "dev", selection["selected_agent"])
        + protocol.schedule(pack, "test", selection["selected_agent"])
    )
    actual_schedule = [
        {key: e[key] for key in ("case_id", "agent_id", "repeat", "task_id", "phase")}
        for e in records
    ]
    if actual_schedule != expected_schedule:
        raise ValueError("Arena schedule missing/reordered or holdout contamination")
    if read(local_file(folder, "selection.json")) != selection:
        raise ValueError("Arena selection differs from train-only recomputation")
    report = metrics.report(pack, derived, selection, machine)
    if read(local_file(folder, "report.json")) != report:
        raise ValueError("Arena report differs from recomputation")
    if local_file(folder, "transitions.jsonl").read_bytes() != metrics.transitions(
        pack, derived, states
    ):
        raise ValueError("Arena learning transitions differ")
    if set(manifest["artifacts"]) != expected_files:
        raise ValueError("Arena inventory contains missing/undeclared artifacts")
    actual = {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != expected_files | {"manifest.json"}:
        raise ValueError("Arena unexpected filesystem artifacts")
    return report
