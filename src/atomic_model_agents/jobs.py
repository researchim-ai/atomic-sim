"""CAS journals: commit model, accepted response and explicit agent state together."""

import hashlib
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from atomic_model_agents import protocol, transport
from atomic_model_runtime.episode import EOSEpisode, EpisodeRejected
from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, number, read
from atomic_model_workspace.jobs import _row
from atomic_model_workspace.policies import observation_frame
from atomic_model_workspace.project import prepare_scenario, provenance

MAX_REVISIONS = 512
STATUSES = ("initialized", "running", "completed", "paused", "cancelled", "failed", "rejected")


def source_identity():
    return dict(
        agent_runtime_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        workspace=provenance(),
    )


def _publish(folder, p):
    if p["revision"] >= MAX_REVISIONS:
        raise ValueError("Agent journal revision budget")
    sha = digest(p)
    name = "revisions/" + sha + ".json"
    value = dict(payload=p, sha256=sha)
    path = Path(folder) / name
    if path.exists():
        if read(path) != value:
            raise ValueError("Agent revision collision")
    else:
        atomic_json(path, value)
    atomic_json(Path(folder) / "HEAD.json", dict(schema_version=1, checkpoint=name, sha256=sha))
    return value


def _transition(folder, value, **updates):
    p = dict(
        value["payload"],
        revision=value["payload"]["revision"] + 1,
        parent_sha256=value["sha256"],
        **updates,
    )
    if p["source"] != source_identity():
        raise ValueError("Agent source changed during execution")
    return _publish(folder, p)


def create(scenario, agent, folder, command, *, identity_files=()):
    scenario, env = prepare_scenario(scenario)
    desc = protocol.descriptor(agent, env)
    launch = transport.launch_identity(command, identity_files)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "revisions").mkdir()
    p = dict(
        schema_version=1,
        job_id=secrets.token_hex(16),
        revision=0,
        parent_sha256=None,
        scenario=scenario,
        agent=desc,
        launch=launch,
        source=source_identity(),
        status="initialized",
        reason="initialized",
        episode=env.export_episode(),
        agent_state=desc["initial_state"],
        rows=[dict(model=_row(env), request=None, response=None)],
        attempts=[],
    )
    with Lock(folder):
        return _publish(folder, p)


def _validate(p):
    keys(
        p,
        (
            "schema_version",
            "job_id",
            "revision",
            "parent_sha256",
            "scenario",
            "agent",
            "launch",
            "source",
            "status",
            "reason",
            "episode",
            "agent_state",
            "rows",
            "attempts",
        ),
    )
    if (
        type(p["schema_version"]) is not int
        or p["schema_version"] != 1
        or type(p["revision"]) is not int
        or not 0 <= p["revision"] < MAX_REVISIONS
        or p["source"] != source_identity()
        or p["status"] not in STATUSES
        or not isinstance(p["reason"], str)
        or not isinstance(p["job_id"], str)
        or len(p["job_id"]) != 32
        or any(c not in "0123456789abcdef" for c in p["job_id"])
        or not isinstance(p["attempts"], list)
        or len(p["attempts"]) > 256
    ):
        raise ValueError("Agent checkpoint schema/source mismatch")
    scenario, env = prepare_scenario(p["scenario"])
    if scenario != p["scenario"] or protocol.descriptor(p["agent"], env) != p["agent"]:
        raise ValueError("Noncanonical agent scenario/permissions")
    rows = p["rows"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 1025
        or rows[0] != dict(model=_row(env), request=None, response=None)
    ):
        raise ValueError("Agent initial row mismatch")
    current = protocol.state(p["agent"]["initial_state"])
    prefixes = [env.export_episode()]
    for row in rows[1:]:
        keys(row, ("model", "request", "response"))
        request = protocol.request(env, p["agent"], current, p["job_id"])
        if request != row["request"]:
            raise ValueError("Agent request/observation/state mismatch")
        action, current = protocol.response(row["response"], request, env, p["agent"])
        frame = observation_frame(env)
        env.step([action[n] for n in env.config.action_channels])
        number(row["model"]["policy_wall_s"], 0, 60, "policy_wall_s")
        prefixes.append(env.export_episode())
        if _row(env, action, frame, row["model"]["policy_wall_s"]) != row["model"]:
            raise ValueError("Agent native row replay mismatch")
    if (
        env.export_episode() != p["episode"]
        or current != p["agent_state"]
        or (p["status"] == "completed") != env._done
    ):
        raise ValueError("Agent final native/state/completion mismatch")
    accepted = 0
    requests = [row["request"] for row in rows[1:]]
    if not env._done:
        requests.append(protocol.request(env, p["agent"], current, p["job_id"]))
    for attempt in p["attempts"]:
        keys(attempt, ("decision", "request_sha256", "outcome", "telemetry"))
        if type(attempt["decision"]) is not int or not 0 <= attempt["decision"] < len(rows):
            raise ValueError("Agent attempt decision mismatch")
        if (
            accepted >= len(requests)
            or attempt["decision"] != accepted
            or attempt["request_sha256"] != digest(requests[accepted])
        ):
            raise ValueError("Agent attempt request/order mismatch")
        outcome = attempt["outcome"]
        allowed = {
            "accepted",
            "agent_cancelled",
            "agent_timeout",
            "agent_rss_limit",
            "agent_stdout_limit",
            "agent_stderr_limit",
            "agent_exit",
            "agent_jsonl",
            "agent_response_contract",
            "agent_identity_changed",
            "agent_launch_io",
        }
        criterion = (
            outcome[10:] if isinstance(outcome, str) and outcome.startswith("numerical:") else ""
        )
        numerical = (
            bool(criterion)
            and len(criterion) <= 64
            and all(c.isascii() and (c.isalnum() or c in "_-") for c in criterion)
        )
        if not isinstance(outcome, str) or (outcome not in allowed and not numerical):
            raise ValueError("Agent attempt outcome schema")
        measured = attempt["telemetry"]
        if measured:
            keys(
                measured,
                (
                    "wall_s",
                    "peak_rss_bytes",
                    "exit_code",
                    "stdout_bytes",
                    "stderr_bytes",
                    "stdout_sha256",
                    "stderr_sha256",
                ),
            )
            number(measured["wall_s"], 0, 1e9, "agent wall_s")
            for field, high in [
                ("peak_rss_bytes", 2**63 - 1),
                ("stdout_bytes", transport.MAX_STDOUT + 4096),
                ("stderr_bytes", transport.MAX_STDERR + 4096),
            ]:
                if type(measured[field]) is not int or not 0 <= measured[field] <= high:
                    raise ValueError("Agent telemetry count schema")
            if type(measured["exit_code"]) is not int or not -255 <= measured["exit_code"] <= 255:
                raise ValueError("Agent telemetry exit schema")
            for field in ("stdout_sha256", "stderr_sha256"):
                value = measured[field]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise ValueError("Agent telemetry hash schema")
        elif outcome != "agent_launch_io":
            raise ValueError("Agent telemetry missing")
        if outcome == "accepted":
            if measured["wall_s"] > p["agent"]["decision_timeout_s"] or measured["exit_code"] != 0:
                raise ValueError("Agent accepted decision budget/exit mismatch")
            accepted += 1
    if accepted != len(rows) - 1:
        raise ValueError("Agent accepted attempt count mismatch")
    return prefixes


def load(folder):
    folder = Path(folder)
    head = read(folder / "HEAD.json")
    keys(head, ("schema_version", "checkpoint", "sha256"))
    if type(head["schema_version"]) is not int or head["schema_version"] != 1:
        raise ValueError("Agent HEAD schema")
    value = read(local_file(folder, head["checkpoint"]))
    keys(value, ("payload", "sha256"))
    if (
        head["checkpoint"] != "revisions/" + value["sha256"] + ".json"
        or head["sha256"] != value["sha256"]
        or digest(value["payload"]) != value["sha256"]
    ):
        raise ValueError("Agent HEAD checksum mismatch")
    prefixes = _validate(value["payload"])
    sha = value["payload"]["parent_sha256"]
    revision = value["payload"]["revision"]
    while revision:
        revision -= 1
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("Agent parent identity")
        older = read(local_file(folder, "revisions/" + sha + ".json"))
        keys(older, ("payload", "sha256"))
        old = older["payload"]
        if (
            digest(old) != sha
            or older["sha256"] != sha
            or old["revision"] != revision
            or old["source"] != value["payload"]["source"]
            or old["job_id"] != value["payload"]["job_id"]
        ):
            raise ValueError("Agent journal chain mismatch")
        latest = value["payload"]
        if (
            old["scenario"] != latest["scenario"]
            or old["agent"] != latest["agent"]
            or old["launch"] != latest["launch"]
            or type(old["revision"]) is not int
            or not 1 <= len(old["rows"]) <= len(latest["rows"])
            or old["rows"] != latest["rows"][: len(old["rows"])]
            or old["attempts"] != latest["attempts"][: len(old["attempts"])]
            or old["episode"] != prefixes[len(old["rows"]) - 1]
            or old["agent_state"]
            != (
                old["rows"][-1]["response"]["state"]
                if len(old["rows"]) > 1
                else old["agent"]["initial_state"]
            )
        ):
            raise ValueError("Agent journal accepted-prefix mismatch")
        sha = old["parent_sha256"]
    if sha is not None:
        raise ValueError("Agent initial parent mismatch")
    return value


def run(folder, command, *, identity_files=(), max_decisions=None, deadline_s=3600):
    if max_decisions is not None and (
        type(max_decisions) is not int or not 1 <= max_decisions <= 1024
    ):
        raise ValueError("Agent decision budget")
    number(deadline_s, 0.001, 3600, "deadline_s")
    started = time.monotonic()
    accepted = 0
    folder = Path(folder)
    with Lock(folder):
        value = load(folder)
        p = value["payload"]
        if p["launch"] != transport.launch_identity(command, identity_files):
            raise ValueError("Agent command/entry identity changed")
        if p["status"] in ("completed", "rejected"):
            return value
        if p["status"] in ("failed", "cancelled"):
            raise ValueError("Explicit CAS resume required")
        env = EOSEpisode.from_episode(p["episode"])
        while not env._done:
            remaining = deadline_s - (time.monotonic() - started)
            if (folder / "CANCEL.json").exists():
                return _transition(folder, value, status="cancelled", reason="cancel_requested")
            if remaining <= 0 or (max_decisions is not None and accepted >= max_decisions):
                return _transition(
                    folder,
                    value,
                    status="paused",
                    reason="deadline" if remaining <= 0 else "decision_budget",
                )
            if len(p["attempts"]) >= 256:
                raise ValueError("Agent attempt budget exhausted")
            req = protocol.request(env, p["agent"], p["agent_state"], p["job_id"])
            telemetry = {}
            try:
                reply, telemetry = transport.exchange(
                    command,
                    req,
                    timeout_s=min(p["agent"]["decision_timeout_s"], remaining),
                    rss_limit_bytes=p["agent"]["rss_limit_bytes"],
                    cancelled=lambda: (folder / "CANCEL.json").exists(),
                )
                if p["launch"] != transport.launch_identity(command, identity_files):
                    raise transport.AgentFailure("agent_identity_changed", telemetry)
                action, new_state = protocol.response(reply, req, env, p["agent"])
                frame = observation_frame(env)
                env.step([action[n] for n in env.config.action_channels])
            except (transport.AgentFailure, ValueError, OSError) as exc:
                if isinstance(exc, transport.AgentFailure):
                    reason = exc.reason
                    telemetry = exc.telemetry
                elif isinstance(exc, OSError):
                    reason = "agent_launch_io"
                elif isinstance(exc, EpisodeRejected):
                    reason = "numerical:" + exc.criterion
                else:
                    reason = "agent_response_contract"
                # Publish the previous native checkpoint/state; a failed reply is not accepted.
                attempts = [
                    *p["attempts"],
                    dict(
                        decision=req["decision"],
                        request_sha256=digest(req),
                        outcome=reason,
                        telemetry=telemetry,
                    ),
                ]
                status = (
                    "cancelled"
                    if reason == "agent_cancelled"
                    else "rejected"
                    if reason.startswith("numerical:")
                    else "paused"
                    if reason == "agent_timeout" and remaining < p["agent"]["decision_timeout_s"]
                    else "failed"
                )
                return _transition(folder, value, status=status, reason=reason, attempts=attempts)
            row = dict(
                model=_row(env, action, frame, telemetry["wall_s"]), request=req, response=reply
            )
            attempts = [
                *p["attempts"],
                dict(
                    decision=req["decision"],
                    request_sha256=digest(req),
                    outcome="accepted",
                    telemetry=telemetry,
                ),
            ]
            value = _transition(
                folder,
                value,
                status="completed" if env._done else "running",
                reason="time_limit" if env._done else "accepted_decision",
                episode=env.export_episode(),
                agent_state=new_state,
                rows=[*p["rows"], row],
                attempts=attempts,
            )
            p = value["payload"]
            accepted += 1
        return value


def cancel(folder):
    # Deliberately independent of the writer lock, so an in-flight agent can stop.
    atomic_json(Path(folder) / "CANCEL.json", dict(schema_version=1, requested=True))


def resume(folder, *, expected_sha256, command, identity_files=()):
    with Lock(folder):
        value = load(folder)
        if value["sha256"] != expected_sha256:
            raise ValueError("Stale agent resume checkpoint")
        if value["payload"]["status"] not in ("paused", "cancelled", "failed", "running"):
            raise ValueError("Agent job cannot resume")
        if value["payload"]["launch"] != transport.launch_identity(command, identity_files):
            raise ValueError("Agent command/entry identity changed")
        (Path(folder) / "CANCEL.json").unlink(missing_ok=True)
        return _transition(folder, value, status="paused", reason="resume_requested")


def export(folder, output):
    with Lock(folder):
        value = load(folder)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        try:
            (output / "revisions").mkdir()
            sha = value["sha256"]
            total = 0
            while sha:
                name = "revisions/" + sha + ".json"
                source = local_file(folder, name)
                total += source.stat().st_size
                if total > 128 * 1024**2:
                    raise ValueError("Agent export total size limit")
                item = read(source)
                atomic_json(output / name, item)
                sha = item["payload"]["parent_sha256"]
            atomic_json(output / "HEAD.json", read(Path(folder) / "HEAD.json"))
            atomic_json(
                output / "manifest.json",
                dict(
                    schema_version=1,
                    source=source_identity(),
                    artifacts={
                        p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(output.rglob("*"))
                        if p.is_file()
                    },
                    validated=False,
                    plant_referenced=False,
                ),
            )
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        schema_version=1,
        checkpoint_sha256=value["sha256"],
        output=str(output),
        validated=False,
        plant_referenced=False,
    )


def replay(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(manifest, ("schema_version", "source", "artifacts", "validated", "plant_referenced"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["source"] != source_identity()
        or manifest["validated"] is not False
        or manifest["plant_referenced"] is not False
        or not isinstance(manifest["artifacts"], dict)
        or not 2 <= len(manifest["artifacts"]) <= MAX_REVISIONS + 1
    ):
        raise ValueError("Agent export schema/source")
    total = 0
    for name, sha in manifest["artifacts"].items():
        with local_file(folder, name).open("rb") as stream:
            raw = stream.read(32 * 1024**2 + 1)
        total += len(raw)
        if (
            len(raw) > 32 * 1024**2
            or total > 128 * 1024**2
            or hashlib.sha256(raw).hexdigest() != sha
        ):
            raise ValueError("Agent export artifact hash/size")
    with tempfile.TemporaryDirectory(prefix="atomic-agent-replay-") as tmp:
        root = Path(tmp)
        for name in manifest["artifacts"]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_file(folder, name), target)
        value = load(root)
        expected = {"HEAD.json"}
        sha = value["sha256"]
        while sha:
            name = "revisions/" + sha + ".json"
            expected.add(name)
            sha = read(root / name)["payload"]["parent_sha256"]
        if set(manifest["artifacts"]) != expected:
            raise ValueError("Agent export inventory mismatch")
    return dict(
        schema_version=1,
        replay_exact=True,
        accepted_decisions=len(value["payload"]["rows"]) - 1,
        agent_executed=False,
        validated=False,
        plant_referenced=False,
    )
