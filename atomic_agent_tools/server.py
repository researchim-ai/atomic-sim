"""Bounded local tool sessions over the public, sensor-only PWR environment."""

import hashlib
import logging
import math
import platform
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from agent_envs import EnvConfig, PWRPlantEnv
from atomic_agent_tools import controlled
from atomic_agent_tools.protocol import (
    ACTION,
    CONFIG,
    ENVELOPE,
    LIMITS,
    PROTOCOL_VERSION,
    SCHEMAS,
    SESSION,
    UNITS,
    ToolError,
    canonical,
    catalog,
    validate,
)
from atomic_model_runtime.episode import EOSEpisode, EpisodeRejected, model_card, runtime_provenance
from atomic_pwr_agent import Episode as ControlledEpisode
from core.status import DomainError
from plants.vver import PASSPORT
from sdk.experiment import _source_fingerprint

LOG = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.glob("*.py")):
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
    return dict(
        adapter_sha256=h.hexdigest(),
        physics_source_sha256=_source_fingerprint(),
        protocol_version=PROTOCOL_VERSION,
        episode_contract="2.1.0",
        eos_runtime=runtime_provenance(),
        controlled_runtime=controlled.provenance(),
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        versions={name: metadata.version(name) for name in ["numpy", "scipy", "gymnasium"]},
    )


def keys(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ToolError("invalid_checkpoint", "Missing or unknown checkpoint fields")


@dataclass
class Session:
    env: PWRPlantEnv | EOSEpisode | ControlledEpisode
    revision: int
    generation: str
    phase: str
    transition: dict


class AgentToolServer:
    """One serialized local worker; host exports are not agent tools."""

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions = {}
        self._checkpoints = {}
        self._requests = 0
        self._model_transactions = 0
        self.captured = provenance()

    def _sources(self):
        if provenance() != self.captured:
            raise ToolError(
                "source_drift", "Simulator or tool runtime changed; start a matching worker"
            )

    def _charge(self):
        if self._model_transactions >= LIMITS["max_model_transactions"]:
            raise ToolError("resource_limit", "Model transaction budget exhausted")
        self._model_transactions += 1

    def _config(self, config):
        validate(CONFIG, config, "config")
        values = dict(seed=0, action_dt_s=0.01, episode_horizon_s=1.0)
        values.update(config)
        try:
            result = EnvConfig.from_dict(values)
        except ValueError as exc:
            raise ToolError("invalid_argument", str(exc)) from exc
        if (
            math.ceil(result.episode_horizon_s / result.action_dt_s - 1e-12)
            > LIMITS["max_episode_decisions"]
        ):
            raise ToolError("resource_limit", "Horizon needs too many decisions for this worker")
        return result

    def _new_name(self, name):
        validate(SESSION, name, "session_id")
        if name in self._sessions:
            raise ToolError("already_exists", "Session already exists")
        if len(self._sessions) >= LIMITS["max_sessions"]:
            raise ToolError("resource_limit", "Session capacity exhausted")

    def _session(self, name):
        if name not in self._sessions:
            raise ToolError("not_found", "Unknown session")
        return self._sessions[name]

    def _payload(self, s):
        value = dict(
            schema_version=1,
            provenance=deepcopy(self.captured),
            episode=(
                controlled.export_episode(s.env)
                if isinstance(s.env, ControlledEpisode)
                else s.env.export_episode()
            ),
            phase=s.phase,
            transition=deepcopy(s.transition),
        )
        if (
            len(canonical(dict(payload=value, sha256=digest(value))).encode())
            > LIMITS["max_checkpoint_bytes"]
        ):
            raise ToolError("resource_limit", "Episode checkpoint size limit exceeded")
        return value

    def _view(self, name, s):
        if isinstance(s.env, ControlledEpisode):
            self._payload(s)  # Enforce the same checkpoint byte budget as other models.
            return controlled.view(name, s)
        native = self._payload(s)["episode"]
        meta = s.env.observation_metadata()
        token = digest(
            dict(
                generation=s.generation,
                revision=s.revision,
                phase=s.phase,
                episode=digest(native),
                transition=s.transition,
            )
        )
        observations = []
        for i, (channel, value) in enumerate(
            zip(meta["channels"], s.env.read_observation().tolist())
        ):
            observations.append(
                dict(
                    name=channel["name"],
                    unit=channel["unit"],
                    value=value,
                    valid=meta["valid"][i],
                    sample_time_s=meta["sample_time_s"][i],
                    saturated=meta["saturated"][i],
                )
            )
        return dict(
            session_id=name,
            revision=s.revision,
            observation_id=token,
            status=s.phase,
            time_s=meta["time_s"],
            observation=observations,
            action_channels={
                k: dict(
                    unit=getattr(s.env, "action_units", UNITS)[k],
                    minimum=float(s.env.action_space.low[i]),
                    maximum=float(s.env.action_space.high[i]),
                )
                for i, k in enumerate(s.env.config.action_channels)
            },
            last_transition=deepcopy(s.transition),
            episode_contract=getattr(s.env, "contract_version", "2.1.0"),
            model=model_card() if isinstance(s.env, EOSEpisode) else PASSPORT.as_dict(),
        )

    def _guard(self, name, args):
        s = self._session(name)
        current = self._view(name, s)
        if (
            args["expected_revision"] != s.revision
            or args["observation_id"] != current["observation_id"]
        ):
            raise ToolError("stale_observation", "Read a fresh observation before this action")
        return s

    def _load(self, checkpoint):
        keys(checkpoint, ["payload", "sha256"])
        p = checkpoint["payload"]
        try:
            if len(canonical(checkpoint).encode()) > LIMITS["max_checkpoint_bytes"]:
                raise ToolError("resource_limit", "Checkpoint is too large")
            if digest(p) != checkpoint["sha256"]:
                raise ToolError("invalid_checkpoint", "Checkpoint checksum mismatch")
            keys(p, ["schema_version", "provenance", "episode", "phase", "transition"])
            if (
                type(p["schema_version"]) is not int
                or p["schema_version"] != 1
                or p["provenance"] != self.captured
            ):
                raise ToolError("invalid_checkpoint", "Checkpoint source/runtime mismatch")
            if p["phase"] not in ["active", "completed", "failed", "cancelled"]:
                raise ToolError("invalid_checkpoint", "Unknown lifecycle state")
            keys(p["transition"], ["reward", "terminated", "truncated", "reason"])
            tr = p["transition"]
            native = p["episode"]
            if isinstance(native, dict) and native.get("contract_version") == controlled.CONTRACT:
                self._charge()
                env = controlled.restore(native, p["phase"], tr)
                return Session(env, 0, uuid.uuid4().hex, p["phase"], deepcopy(tr))
            if not isinstance(native, dict) or not isinstance(native.get("config"), dict):
                raise ToolError("invalid_checkpoint", "Malformed native episode")
            is_eos = native.get("contract_version") == EOSEpisode.contract_version
            if not is_eos:
                fields = CONFIG["properties"]
                subset = {key: native["config"][key] for key in fields}
                expected = self._config(subset).as_dict()
                if native["config"] != expected or native.get("base_config") != expected:
                    raise ToolError("invalid_checkpoint", "Unsupported episode configuration")
            reward = tr["reward"]
            if reward is not None and (
                type(reward) not in (int, float) or not math.isfinite(reward)
            ):
                raise ToolError("invalid_checkpoint", "Invalid reward")
            if type(tr["terminated"]) is not bool or type(tr["truncated"]) is not bool:
                raise ToolError("invalid_checkpoint", "Invalid stop flags")
            self._charge()
            env = (
                EOSEpisode.from_episode(deepcopy(native))
                if is_eos
                else PWRPlantEnv.from_episode(deepcopy(native))
            )
            reason = native["reason"]
            expected_phase = (
                "active"
                if not native["done"]
                else "completed"
                if reason in ["time_limit", "task_terminal"]
                else "failed"
            )
            if p["phase"] not in [expected_phase, "cancelled"] or (
                p["phase"] == "cancelled" and native["done"]
            ):
                raise ToolError("invalid_checkpoint", "Inconsistent lifecycle")
            if (
                tr["reason"] != reason
                or tr["terminated"] != (reason == "task_terminal")
                or tr["truncated"] != (native["done"] and reason != "task_terminal")
            ):
                raise ToolError("invalid_checkpoint", "Inconsistent termination flags")
            if (reward is None) != (native["steps"] == 0 and not native["failure_log"]):
                raise ToolError("invalid_checkpoint", "Inconsistent reward presence")
            return Session(env, 0, uuid.uuid4().hex, p["phase"], deepcopy(tr))
        except ToolError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, RuntimeError) as exc:
            raise ToolError("invalid_checkpoint", "Native episode validation failed") from exc

    def export_session(self, name):
        """Trusted host API: full simulator/RNG state, intentionally not an agent tool."""
        with self._lock:
            self._sources()
            p = self._payload(self._session(name))
            return dict(payload=p, sha256=digest(p))

    def import_session(self, name, checkpoint):
        """Trusted host API; creates a new token generation rather than reusing old guards."""
        with self._lock:
            self._sources()
            self._new_name(name)
            candidate = self._load(checkpoint)
            view = self._view(name, candidate)
            self._sources()
            self._sessions[name] = candidate
            return view

    def invoke(self, tool, args):
        with self._lock:
            self._sources()
            if tool not in SCHEMAS:
                raise ToolError("unknown_tool", "Unsupported tool")
            validate(SCHEMAS[tool], args)
            if tool == "describe_simulator":
                return dict(
                    **catalog(),
                    model=PASSPORT.as_dict(),
                    models=[PASSPORT.as_dict(), model_card(), controlled.model_card()],
                    provenance=deepcopy(self.captured),
                )
            if tool == "create_controlled_episode":
                name = args["session_id"]
                self._new_name(name)
                self._charge()
                try:
                    env = ControlledEpisode(**args.get("config", {}))
                except ValueError as exc:
                    raise ToolError("invalid_argument", str(exc)) from exc
                candidate = Session(env, 0, uuid.uuid4().hex, "active", controlled.transition(env))
                result = self._view(name, candidate)
                self._sources()
                self._sessions[name] = candidate
                return result
            if tool in ("create_episode", "create_eos_episode"):
                name = args["session_id"]
                self._new_name(name)
                if tool == "create_episode":
                    config = self._config(args.get("config", {}))
                    self._charge()
                    env = PWRPlantEnv(config=config)
                    env.reset(seed=config.seed)
                else:
                    self._charge()
                    try:
                        env = EOSEpisode(args.get("config", {}))
                    except ValueError as exc:
                        raise ToolError("invalid_argument", str(exc)) from exc
                candidate = Session(
                    env,
                    0,
                    uuid.uuid4().hex,
                    "active",
                    dict(reward=None, terminated=False, truncated=False, reason="running"),
                )
                result = self._view(name, candidate)
                self._sources()
                self._sessions[name] = candidate
                return result
            if tool == "forget_checkpoint":
                cid = args["checkpoint_id"]
                if cid not in self._checkpoints:
                    raise ToolError("not_found", "Unknown checkpoint handle")
                del self._checkpoints[cid]
                return dict(checkpoint_id=cid, forgotten=True)
            name = args["session_id"]
            if tool == "inspect_episode":
                return self._view(name, self._session(name))
            s = self._guard(name, args)
            if tool == "export_controlled_case":
                if not isinstance(s.env, ControlledEpisode):
                    raise ToolError("invalid_argument", "Tool requires a controlled PWR episode")
                if s.env.observe()["revision"] == 0:
                    raise ToolError("invalid_state", "No accepted steps to export")
                case = s.env.training_case(
                    case_id=args["case_id"],
                    family=args["family"],
                    split=args.get("split", "validation"),
                )
                return dict(model=controlled.model_card(), case=case)
            if tool == "save_checkpoint":
                p = self._payload(s)
                checkpoint = dict(payload=p, sha256=digest(p))
                cid = digest(checkpoint)
                if cid not in self._checkpoints:
                    size = len(canonical(checkpoint).encode())
                    if (
                        len(self._checkpoints) >= LIMITS["max_checkpoints"]
                        or sum(len(canonical(c).encode()) for c in self._checkpoints.values())
                        + size
                        > LIMITS["max_checkpoint_total_bytes"]
                    ):
                        raise ToolError(
                            "resource_limit",
                            "Checkpoint capacity exhausted; forget an unused handle",
                        )
                    self._checkpoints[cid] = checkpoint
                return dict(checkpoint_id=cid, session=self._view(name, s))
            if tool == "restore_checkpoint":
                cid = args["checkpoint_id"]
                if cid not in self._checkpoints:
                    raise ToolError("not_found", "Unknown checkpoint handle")
                candidate = self._load(self._checkpoints[cid])
                if isinstance(candidate.env, ControlledEpisode) != isinstance(
                    s.env, ControlledEpisode
                ):
                    raise ToolError(
                        "invalid_checkpoint", "Cannot change controlled model through restore"
                    )
                candidate.generation = s.generation
                candidate.revision = s.revision + 1
            elif tool == "close_episode":
                del self._sessions[name]
                return dict(session_id=name, closed=True)
            elif tool == "cancel_episode":
                if s.phase != "active":
                    raise ToolError("invalid_state", "Only active episodes can be cancelled")
                candidate = Session(
                    s.env, s.revision + 1, s.generation, "cancelled", deepcopy(s.transition)
                )
            elif tool == "step_controlled_episode":
                if not isinstance(s.env, ControlledEpisode):
                    raise ToolError("invalid_argument", "Tool requires a controlled PWR episode")
                if s.phase != "active":
                    raise ToolError("invalid_state", "Episode is not active")
                from sdk.api import validate_commands

                try:
                    commands = validate_commands(args["action"])
                except ValueError as exc:
                    raise ToolError("invalid_argument", str(exc)) from exc
                self._charge()
                # Candidate replay makes a failed step/serialization/source check noncommitting.
                env = ControlledEpisode.restore(s.env.checkpoint())
                try:
                    env.step(commands, observation_id=env.observe()["observation_id"])
                except (DomainError, ValueError, RuntimeError) as exc:
                    raise ToolError(
                        "numerical_rejected", "Controlled step rejected; no state committed"
                    ) from exc
                tr = controlled.transition(env)
                candidate = Session(
                    env,
                    s.revision + 1,
                    s.generation,
                    "completed" if tr["terminated"] or tr["truncated"] else "active",
                    tr,
                )
            else:
                if isinstance(s.env, ControlledEpisode):
                    raise ToolError(
                        "invalid_argument", "Use step_controlled_episode for this model"
                    )
                if s.phase != "active":
                    raise ToolError("invalid_state", "Episode is not active")
                native = s.env.export_episode()
                if native["steps"] >= LIMITS["max_episode_decisions"]:
                    raise ToolError("resource_limit", "Decision budget exhausted")
                action = args["action"]
                if set(action) != set(s.env.config.action_channels):
                    raise ToolError(
                        "invalid_argument", "Action must name exactly the configured channels"
                    )
                is_eos = isinstance(s.env, EOSEpisode)
                if is_eos:
                    try:
                        s.env.validate_action(action)
                    except ValueError as exc:
                        raise ToolError("invalid_argument", str(exc)) from exc
                else:
                    validate(ACTION, action)
                self._charge()
                env = (
                    EOSEpisode.from_episode(native) if is_eos else PWRPlantEnv.from_episode(native)
                )
                try:
                    _, reward, terminated, truncated, info = env.step(
                        [action[k] for k in env.config.action_channels]
                    )
                except EpisodeRejected as exc:
                    raise ToolError(
                        "numerical_rejected",
                        f"EOS step rejected ({exc.criterion}); no state committed",
                    ) from exc
                phase = (
                    "active"
                    if not (terminated or truncated)
                    else "completed"
                    if info["reason"] in ["time_limit", "task_terminal"]
                    else "failed"
                )
                candidate = Session(
                    env,
                    s.revision + 1,
                    s.generation,
                    phase,
                    dict(
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        reason=info["reason"],
                    ),
                )
            result = self._view(name, candidate)
            canonical(result)
            self._sources()
            self._sessions[name] = candidate
            return result

    def handle_request(self, request):
        request_id = None
        with self._lock:
            if self._requests >= LIMITS["max_requests"]:
                return failure(None, "resource_limit", "Request budget exhausted")
            self._requests += 1
            try:
                if (
                    isinstance(request, dict)
                    and isinstance(request.get("request_id"), str)
                    and 1 <= len(request["request_id"]) <= 128
                ):
                    request_id = request["request_id"]
                validate(ENVELOPE, request, "request")
                request_id = request["request_id"]
                result = self.invoke(request["tool"], request["arguments"])
                return dict(
                    protocol_version=PROTOCOL_VERSION, request_id=request_id, ok=True, result=result
                )
            except ToolError as exc:
                return failure(request_id, exc.code, str(exc))
            except DomainError:
                return failure(
                    request_id, "model_rejected", "Model rejected initialization or checkpoint"
                )
            except Exception:
                LOG.exception("Unexpected simulator tool error; candidate was not committed")
                return failure(
                    request_id,
                    "internal_error",
                    "Unexpected simulator error; state was not committed",
                )


def failure(request_id, code, message):
    return dict(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        ok=False,
        error=dict(code=code, message=message),
    )
