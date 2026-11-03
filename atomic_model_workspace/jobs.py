"""Persistent single-writer jobs: immutable revisions and an atomic HEAD pointer."""

import os
import time
import uuid
from pathlib import Path

from atomic_model_runtime.episode import EOSEpisode, EpisodeRejected
from atomic_model_workspace.common import Lock, atomic_json, digest, keys, local_file, number, read
from atomic_model_workspace.policies import decide, observation_frame, prepare_policy
from atomic_model_workspace.project import prepare_scenario, provenance

TERMINAL = ("completed", "rejected", "failed")
STATUSES = (*TERMINAL, "initialized", "running", "paused", "cancelled")


def prepare_job(spec):
    keys(spec, ("schema_version", "scenario", "policy", "decision_timeout_s"))
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Expected job schema1")
    scenario, env = prepare_scenario(spec["scenario"])
    return dict(
        schema_version=1,
        scenario=scenario,
        policy=prepare_policy(spec["policy"], env),
        decision_timeout_s=number(spec["decision_timeout_s"], 0.001, 60, "decision_timeout_s"),
    ), env


def _row(env, action=None, frame=None, elapsed=0.0):
    s = env._model.snapshot()
    return dict(
        decision=env._steps,
        time_s=s["time_s"],
        observation=observation_frame(env),
        policy_input=frame,
        heat_power_w=action,
        policy_wall_s=elapsed,
        ledger={
            k: s[k]
            for k in (
                "total_energy_change_j",
                "external_heat_j",
                "exported_drag_heat_j",
                "entropy_change_j_k",
                "external_entropy_j_k",
                "generated_entropy_j_k",
                "energy_residual_j",
                "entropy_residual_j_k",
                "wall_ledgers",
            )
        },
        circuits={
            name: {
                k: c[k]
                for k in ("mass_residual_kg", "momentum_residual_kg_m_s", "energy_residual_j")
            }
            for name, c in s["circuits"].items()
        },
    )


def _publish(folder, state):
    folder = Path(folder)
    name = f"revisions/{state['revision']:08d}-{uuid.uuid4().hex}.json"
    envelope = dict(payload=state, sha256=digest(state))
    atomic_json(folder / name, envelope)
    atomic_json(
        folder / "HEAD.json", dict(schema_version=1, checkpoint=name, sha256=envelope["sha256"])
    )


def load_job(folder, *, current_source=True):
    folder = Path(folder)
    head = read(folder / "HEAD.json")
    keys(head, ("schema_version", "checkpoint", "sha256"))
    if type(head["schema_version"]) is not int or head["schema_version"] != 1:
        raise ValueError("Unknown HEAD schema")
    checkpoint = read(local_file(folder, head["checkpoint"]))
    keys(checkpoint, ("payload", "sha256"))
    state = checkpoint["payload"]
    if digest(state) != checkpoint["sha256"] or head["sha256"] != checkpoint["sha256"]:
        raise ValueError("Job checkpoint checksum mismatch")
    keys(
        state,
        (
            "schema_version",
            "revision",
            "spec",
            "spec_sha256",
            "provenance",
            "status",
            "reason",
            "episode",
            "rows",
        ),
    )
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != 1
        or type(state["revision"]) is not int
        or state["revision"] < 0
        or state["status"] not in STATUSES
        or not isinstance(state["reason"], str)
    ):
        raise ValueError("Malformed job lifecycle")
    if current_source and state["provenance"] != provenance():
        raise ValueError("Job source identity mismatch; explicit migration required")
    spec, _ = prepare_job(state["spec"])
    if spec != state["spec"] or digest(spec) != state["spec_sha256"]:
        raise ValueError("Job specification identity mismatch")
    env = EOSEpisode.from_episode(state["episode"])
    if env.config.as_dict() != spec["scenario"]["config"]:
        raise ValueError("Job episode configuration mismatch")
    rows = state["rows"]
    if not isinstance(rows, list) or len(rows) != env._steps + 1:
        raise ValueError("Job trajectory count mismatch")
    previous = None
    for i, row in enumerate(rows):
        if row["decision"] != i or row["time_s"] != min(
            i * env.config.action_dt_s, env.config.episode_horizon_s
        ):
            raise ValueError("Job trajectory clock mismatch")
        number(row["policy_wall_s"], 0, 1e9, "policy wall time")
        if i:
            if (
                row["policy_input"] != previous["observation"]
                or row["heat_power_w"] != env._history[i - 1]["commands"]
                or decide(spec["policy"], row["policy_input"], env.config.action_channels)
                != row["heat_power_w"]
            ):
                raise ValueError("Job policy/action trace mismatch")
        elif row["policy_input"] is not None or row["heat_power_w"] is not None:
            raise ValueError("Initial row cannot contain a policy decision")
        previous = row
    expected = _row(
        env, rows[-1]["heat_power_w"], rows[-1]["policy_input"], rows[-1]["policy_wall_s"]
    )
    if rows[-1] != expected or (state["status"] == "completed") != env._done:
        raise ValueError("Job final state or completion mismatch")
    return state


def create_job(spec, folder):
    spec, env = prepare_job(spec)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "revisions").mkdir()
    state = dict(
        schema_version=1,
        revision=0,
        spec=spec,
        spec_sha256=digest(spec),
        provenance=provenance(),
        status="initialized",
        reason="initialized",
        episode=env.export_episode(),
        rows=[_row(env)],
    )
    with Lock(folder):
        _publish(folder, state)
    return state


def _transition(folder, state, status, reason):
    state.update(revision=state["revision"] + 1, status=status, reason=reason)
    _publish(folder, state)
    return state


def cancel_job(folder):
    state = load_job(folder)
    if state["status"] in TERMINAL:
        raise ValueError("Terminal job cannot be cancelled")
    atomic_json(Path(folder) / "CANCEL.json", dict(schema_version=1, request="cancel"))
    return dict(requested=True)


def pause_job(folder):
    state = load_job(folder)
    if state["status"] in TERMINAL:
        raise ValueError("Terminal job cannot be paused")
    atomic_json(Path(folder) / "PAUSE.json", dict(schema_version=1, request="pause"))
    return dict(requested=True)


def interrupt_job(folder, reason, *, cancelled=False):
    """Supervisor calls only after its own worker has exited and has been joined."""
    with Lock(folder):
        state = load_job(folder)
        if state["status"] not in TERMINAL:
            _transition(folder, state, "cancelled" if cancelled else "paused", reason)
        return state


def resume_job(folder):
    with Lock(folder):
        state = load_job(folder)
        if state["status"] in TERMINAL:
            raise ValueError("Terminal job cannot be resumed")
        (Path(folder) / "CANCEL.json").unlink(missing_ok=True)
        (Path(folder) / "PAUSE.json").unlink(missing_ok=True)
        return _transition(folder, state, "paused", "resume_requested")


def run_job(folder, *, max_decisions=None, deadline_s=None):
    if max_decisions is not None and (
        type(max_decisions) is not int or not 1 <= max_decisions <= 1024
    ):
        raise ValueError("Invalid job decision budget")
    if deadline_s is not None:
        number(deadline_s, 0.000001, 3600, "deadline_s")
    start = time.monotonic()
    folder = Path(folder)
    with Lock(folder):
        state = load_job(folder)
        if state["status"] in TERMINAL:
            return state
        if state["status"] == "cancelled" and not (folder / "CANCEL.json").exists():
            raise ValueError("Cancelled job requires explicit resume")
        env = EOSEpisode.from_episode(state["episode"])
        accepted = 0
        while not env._done:
            if (folder / "CANCEL.json").exists():
                return _transition(folder, state, "cancelled", "cancel_requested")
            if (folder / "PAUSE.json").exists():
                return _transition(folder, state, "paused", "pause_requested")
            if deadline_s is not None and time.monotonic() - start >= deadline_s:
                return _transition(folder, state, "paused", "deadline")
            if max_decisions is not None and accepted >= max_decisions:
                return _transition(folder, state, "paused", "decision_budget")
            frame = observation_frame(env)
            t0 = time.monotonic()
            action = decide(state["spec"]["policy"], frame, env.config.action_channels)
            elapsed = time.monotonic() - t0
            if elapsed > state["spec"]["decision_timeout_s"]:
                return _transition(folder, state, "failed", "policy_deadline")
            try:
                env.step([action[k] for k in env.config.action_channels])
            except EpisodeRejected as exc:
                return _transition(folder, state, "rejected", "numerical:" + exc.criterion)
            state["episode"] = env.export_episode()
            state["rows"].append(_row(env, action, frame, elapsed))
            accepted += 1
            _transition(
                folder,
                state,
                "completed" if env._done else "running",
                "time_limit" if env._done else "accepted_decision",
            )
        return state


def worker(folder):
    """Spawn target: exceptions stay local; supervisor records exit and accepted prefix."""
    try:
        run_job(folder)
    except Exception:
        # A process crash must be visible to the parent, never converted into success.
        raise SystemExit(2)


def clear_exited_worker_lock(folder, process):
    if process.is_alive() or process.exitcode is None:
        raise ValueError("Cannot recover a live worker lock")
    path = Path(folder) / ".lock"
    if path.exists():
        # Only remove the lock demonstrably owned by the worker we joined.
        if path.read_text() != str(process.pid):
            raise ValueError("Lock owner differs from exited worker")
        os.unlink(path)
