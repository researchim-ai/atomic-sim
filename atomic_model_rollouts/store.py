"""CAS metadata journal, frozen project matrix and explicit Linux process ownership."""

import hashlib
import os
import secrets
import shutil
from pathlib import Path

from atomic_model_reference import metrics as curve_metrics
from atomic_model_results import bundle as row_audit
from atomic_model_workspace import jobs
from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, number, read
from atomic_model_workspace.project import load_project, normalize, provenance

MAX_JOBS = 64
MAX_REVISIONS = 4096


def source_identity():
    return dict(
        rollouts_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        workspace=provenance(),
        curve_metrics_sha256=hashlib.sha256(Path(curve_metrics.__file__).read_bytes()).hexdigest(),
        row_audit_sha256=hashlib.sha256(Path(row_audit.__file__).read_bytes()).hexdigest(),
    )


def options(value):
    keys(
        value,
        (
            "workers",
            "cpu_budget",
            "worker_rss_bytes",
            "total_rss_bytes",
            "job_wall_s",
            "controller_wall_s",
            "decision_slice",
            "max_attempts",
            "seeds",
            "log_limit_bytes",
        ),
    )
    for key, lo, hi in [
        ("workers", 1, 8),
        ("cpu_budget", 1, 8),
        ("worker_rss_bytes", 128 * 1024**2, 4 * 1024**3),
        ("total_rss_bytes", 128 * 1024**2, 32 * 1024**3),
        ("decision_slice", 1, 128),
        ("max_attempts", 1, 8),
        ("log_limit_bytes", 1024, 1024**2),
    ]:
        if type(value[key]) is not int or not lo <= value[key] <= hi:
            raise ValueError("Rollout budget: " + key)
    if (
        value["workers"] > value["cpu_budget"]
        or value["total_rss_bytes"] < value["worker_rss_bytes"]
    ):
        raise ValueError("Rollout CPU/memory admission budget")
    seeds = value["seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 8
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Rollout seeds")
    return dict(
        value,
        seeds=sorted(seeds),
        job_wall_s=number(value["job_wall_s"], 0.05, 3600, "job_wall_s"),
        controller_wall_s=number(value["controller_wall_s"], 0.05, 3600, "controller_wall_s"),
    )


def default_options():
    count = min(2, len(os.sched_getaffinity(0)))
    return dict(
        workers=count,
        cpu_budget=count,
        worker_rss_bytes=256 * 1024**2,
        total_rss_bytes=count * 256 * 1024**2,
        job_wall_s=30,
        controller_wall_s=300,
        decision_slice=128,
        max_attempts=8,
        seeds=[17],
        log_limit_bytes=65536,
    )


def expected(project, cfg):
    result = {}
    for sid, scenario in project["scenarios"].items():
        for pid, policy in project["policies"].items():
            for seed in cfg["seeds"]:
                id = digest([sid, pid, seed])
                spec = dict(
                    schema_version=1,
                    scenario=dict(scenario, seed=seed),
                    policy=policy,
                    decision_timeout_s=project["limits"]["decision_timeout_s"],
                )
                result[id] = dict(scenario_id=sid, policy_id=pid, seed=seed, spec=spec)
    if len(result) > MAX_JOBS:
        raise ValueError("Rollout matrix exceeds 64 jobs")
    return result


def publish(folder, p):
    if p["revision"] >= MAX_REVISIONS:
        raise ValueError("Rollout journal budget")
    value = dict(payload=p, sha256=digest(p))
    name = "revisions/" + value["sha256"] + ".json"
    path = Path(folder) / name
    if path.exists():
        if read(path) != value:
            raise ValueError("Rollout revision collision")
    else:
        atomic_json(path, value)
    atomic_json(
        Path(folder) / "HEAD.json", dict(schema_version=1, checkpoint=name, sha256=value["sha256"])
    )
    return value


def commit(folder, value, **changes):
    if value["payload"]["source"] != source_identity():
        raise ValueError("Rollout source changed")
    return publish(
        folder,
        dict(
            value["payload"],
            revision=value["payload"]["revision"] + 1,
            parent_sha256=value["sha256"],
            **changes,
        ),
    )


def create(project_path, folder, config=None):
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise ValueError("Rollouts require Linux")
    project = load_project(project_path)
    cfg = options(config or default_options())
    matrix = expected(project, cfg)
    available = sorted(os.sched_getaffinity(0))
    if cfg["cpu_budget"] > len(available):
        raise ValueError("CPU budget exceeds caller affinity")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("jobs", "revisions", "attempts"):
            (folder / name).mkdir()
        entries = {}
        for id, item in matrix.items():
            jobs.create_job(item["spec"], folder / "jobs" / id)
            entries[id] = dict(
                **{k: item[k] for k in ("scenario_id", "policy_id", "seed")},
                status="ready",
                reason="initialized",
                accepted_decisions=0,
                attempts=[],
            )
        p = dict(
            schema_version=1,
            revision=0,
            parent_sha256=None,
            campaign_id=secrets.token_hex(16),
            project=project,
            project_sha256=digest(project),
            config=cfg,
            cpus=available[: cfg["cpu_budget"]],
            source=source_identity(),
            status="initialized",
            reason="initialized",
            controller=None,
            entries=entries,
        )
        with Lock(folder):
            return publish(folder, p)
    except BaseException:
        shutil.rmtree(folder)
        raise


def validate(p):
    keys(
        p,
        (
            "schema_version",
            "revision",
            "parent_sha256",
            "campaign_id",
            "project",
            "project_sha256",
            "config",
            "cpus",
            "source",
            "status",
            "reason",
            "controller",
            "entries",
        ),
    )
    cfg = options(p["config"])
    project = normalize(p["project"], p["project"]["scenarios"], p["project"]["policies"])
    if (
        type(p["schema_version"]) is not int
        or p["schema_version"] != 1
        or type(p["revision"]) is not int
        or not 0 <= p["revision"] < MAX_REVISIONS
        or p["source"] != source_identity()
        or project != p["project"]
        or digest(project) != p["project_sha256"]
        or cfg != p["config"]
    ):
        raise ValueError("Rollout schema/source/project mismatch")
    if p["status"] not in (
        "initialized",
        "running",
        "paused",
        "cancelled",
        "completed",
        "completed_with_failures",
    ) or not isinstance(p["reason"], str):
        raise ValueError("Rollout lifecycle")
    cpus = p["cpus"]
    if (
        not isinstance(cpus, list)
        or len(cpus) != cfg["cpu_budget"]
        or any(type(c) is not int or c < 0 for c in cpus)
        or len(set(cpus)) != len(cpus)
    ):
        raise ValueError("Rollout CPU shape")
    matrix = expected(project, cfg)
    keys(p["entries"], matrix)
    for id, entry in p["entries"].items():
        keys(
            entry,
            (
                "scenario_id",
                "policy_id",
                "seed",
                "status",
                "reason",
                "accepted_decisions",
                "attempts",
            ),
        )
        if (
            any(entry[k] != matrix[id][k] for k in ("scenario_id", "policy_id", "seed"))
            or entry["status"] not in ("ready", "running", "completed", "failed", "interrupted")
            or type(entry["accepted_decisions"]) is not int
            or not 0 <= entry["accepted_decisions"] <= 128
            or not isinstance(entry["attempts"], list)
            or len(entry["attempts"]) > cfg["max_attempts"]
        ):
            raise ValueError("Rollout entry schema")
    if p["status"] == "running" and p["controller"] is None:
        raise ValueError("Running rollout lacks controller identity")
    if p["controller"] is not None:
        owner = p["controller"]
        keys(owner, ("pid", "tid", "process_start_ticks", "thread_start_ticks", "boot_id", "token"))
        if (
            any(
                type(owner[k]) is not int or owner[k] <= 0
                for k in ("pid", "tid", "process_start_ticks", "thread_start_ticks")
            )
            or not isinstance(owner["boot_id"], str)
            or len(owner["boot_id"]) != 36
        ):
            raise ValueError("Rollout controller identity schema")
        _hex(owner["token"], 32)
    for entry in p["entries"].values():
        for index, attempt in enumerate(entry["attempts"]):
            keys(
                attempt,
                (
                    "id",
                    "pid",
                    "start_ticks",
                    "cpu",
                    "outcome",
                    "before_checkpoint_sha256",
                    "after_checkpoint_sha256",
                    "before_decisions",
                    "after_decisions",
                    "wall_s",
                    "peak_rss_bytes",
                    "exit_code",
                    "audit_wall_s",
                    "log_truncated",
                ),
            )
            _hex(attempt["id"], 32)
            _hex(attempt["before_checkpoint_sha256"], 64)
            if (
                type(attempt["pid"]) is not int
                or attempt["pid"] <= 0
                or attempt["cpu"] not in cpus
                or type(attempt["cpu"]) is not int
                or (
                    attempt["start_ticks"] is not None
                    and (type(attempt["start_ticks"]) is not int or attempt["start_ticks"] < 0)
                )
            ):
                raise ValueError("Rollout attempt process schema")
            if (
                type(attempt["before_decisions"]) is not int
                or not 0 <= attempt["before_decisions"] <= 128
                or type(attempt["log_truncated"]) is not bool
            ):
                raise ValueError("Rollout attempt counts")
            if attempt["outcome"] == "running":
                if (
                    index != len(entry["attempts"]) - 1
                    or entry["status"] != "running"
                    or p["controller"] is None
                    or any(
                        attempt[k] is not None
                        for k in (
                            "after_checkpoint_sha256",
                            "after_decisions",
                            "wall_s",
                            "peak_rss_bytes",
                            "exit_code",
                            "audit_wall_s",
                        )
                    )
                ):
                    raise ValueError("Rollout active attempt mismatch")
            else:
                allowed = {
                    "exited",
                    "cancelled",
                    "pause_requested",
                    "worker_deadline",
                    "worker_rss_limit",
                    "worker_output_limit",
                    "aggregate_rss_limit",
                    "controller_interrupted",
                    "recovered_controller_loss",
                    "launch_handshake_failed",
                }
                if attempt["outcome"] not in allowed:
                    raise ValueError("Unknown rollout outcome")
                if attempt["after_checkpoint_sha256"] is not None:
                    _hex(attempt["after_checkpoint_sha256"], 64)
                if attempt["after_decisions"] is not None and (
                    type(attempt["after_decisions"]) is not int
                    or not attempt["before_decisions"] <= attempt["after_decisions"] <= 128
                ):
                    raise ValueError("Rollout accepted prefix decreased")
                for name in ("wall_s", "audit_wall_s"):
                    if attempt[name] is not None:
                        number(attempt[name], 0, 1e9, name)
                if attempt["peak_rss_bytes"] is not None and (
                    type(attempt["peak_rss_bytes"]) is not int or attempt["peak_rss_bytes"] < 0
                ):
                    raise ValueError("Rollout RSS schema")
                if attempt["exit_code"] is not None and (
                    type(attempt["exit_code"]) is not int or not -255 <= attempt["exit_code"] <= 255
                ):
                    raise ValueError("Rollout exit schema")
    return p


def _hex(value, length):
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected rollout hex identity")


def load(folder, *, history=True):
    folder = Path(folder)
    head = read(folder / "HEAD.json")
    keys(head, ("schema_version", "checkpoint", "sha256"))
    if type(head["schema_version"]) is not int or head["schema_version"] != 1:
        raise ValueError("Rollout HEAD schema")
    value = read(local_file(folder, head["checkpoint"]))
    keys(value, ("payload", "sha256"))
    if (
        head["checkpoint"] != "revisions/" + value["sha256"] + ".json"
        or head["sha256"] != value["sha256"]
        or digest(value["payload"]) != value["sha256"]
    ):
        raise ValueError("Rollout HEAD hash")
    p = validate(value["payload"])
    if history:
        sha = p["parent_sha256"]
        revision = p["revision"]
        newer = p
        while revision:
            revision -= 1
            old = read(local_file(folder, "revisions/" + str(sha) + ".json"))
            keys(old, ("payload", "sha256"))
            q = old["payload"]
            if (
                old["sha256"] != sha
                or digest(q) != sha
                or q["revision"] != revision
                or any(
                    q[k] != p[k]
                    for k in (
                        "schema_version",
                        "campaign_id",
                        "project",
                        "project_sha256",
                        "config",
                        "cpus",
                        "source",
                    )
                )
            ):
                raise ValueError("Rollout journal chain changed")
            for id, item in q["entries"].items():
                following = newer["entries"][id]
                if (
                    following["attempts"][
                        : len(item["attempts"]) - int(item["status"] == "running")
                    ]
                    != item["attempts"][: len(item["attempts"]) - int(item["status"] == "running")]
                ):
                    raise ValueError("Rollout attempt history changed")
            for id, item in q["entries"].items():
                following = newer["entries"][id]
                if (
                    len(following["attempts"]) < len(item["attempts"])
                    or following["accepted_decisions"] < item["accepted_decisions"]
                ):
                    raise ValueError("Rollout journal progress decreased")
                for before, after in zip(item["attempts"], following["attempts"]):
                    if any(
                        before[k] != after[k]
                        for k in (
                            "id",
                            "pid",
                            "start_ticks",
                            "cpu",
                            "before_checkpoint_sha256",
                            "before_decisions",
                        )
                    ):
                        raise ValueError("Rollout attempt identity changed")
            sha = q["parent_sha256"]
            newer = q
        if sha is not None:
            raise ValueError("Rollout initial parent")
    return value


def native(folder, id):
    value = load(folder, history=False)["payload"]
    if id not in value["entries"]:
        raise ValueError("Unknown rollout task")
    job = jobs.load_job(local_file(folder, "jobs/" + id + "/HEAD.json").parent)
    if job["spec"] != expected(value["project"], value["config"])[id]["spec"]:
        raise ValueError("Rollout native spec mismatch")
    row_audit.audit_native(job)
    return job


def process_info(pid, tid=None):
    try:
        base = Path("/proc") / str(pid)
        path = base / "stat" if tid is None else base / "task" / str(tid) / "stat"
        fields = path.read_text().rsplit(")", 1)[1].split()
        return dict(state=fields[0], start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as exc:
        raise ValueError("Cannot prove process identity") from exc


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owner_alive(owner):
    if owner["boot_id"] != boot_id():
        raise ValueError("Foreign boot/controller; explicit host recovery required")
    process = process_info(owner["pid"])
    thread = process_info(owner["pid"], owner["tid"])
    return bool(
        process
        and thread
        and process["state"] != "Z"
        and process["start_ticks"] == owner["process_start_ticks"]
        and thread["start_ticks"] == owner["thread_start_ticks"]
    )
