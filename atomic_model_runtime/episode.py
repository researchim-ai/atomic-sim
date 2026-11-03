"""Public deterministic EOS episode, reusing the qualified assembly equations."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium import spaces

from core.network import NetworkError
from scripts.controlled_eos_assembly import ControlledEOSAssembly
from scripts.inertial_graph import InertialGraphError

MODEL_ID = "eos-thermal-assembly-v1"
CONTRACT = "eos-agent-1.0.0"
PRESETS = ("mixed_ports", "uniform", "gravity")


def model_card():
    return dict(
        model_id=MODEL_ID,
        version="1.0.0",
        status="implemented",
        plant_referenced=False,
        validated=False,
        phenomenon="two finite-inventory Euler circuits with caloric thermal walls",
        approximations=[
            "synthetic stiffened-caloric EOS",
            "first-order hydrostatic Rusanov",
            "same-area terminal volumes",
            "ideal cached temperatures, no sensor noise",
        ],
        applicability="restricted single-phase synthetic research model; not calibrated water or reactor plant",
        reward="zero; no intrinsic benchmark objective",
        presets=list(PRESETS),
    )


class EpisodeRejected(ValueError):
    def __init__(self, criterion, message):
        super().__init__(message)
        self.criterion = criterion


def _keys(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Missing or unknown EOS episode fields")


@dataclass(frozen=True)
class EpisodeConfig:
    preset: str
    action_dt_s: float
    episode_horizon_s: float
    max_step_s: float
    action_channels: tuple

    def as_dict(self):
        return dict(
            preset=self.preset,
            action_dt_s=self.action_dt_s,
            episode_horizon_s=self.episode_horizon_s,
            max_step_s=self.max_step_s,
        )


class EOSEpisode:
    contract_version = CONTRACT
    model_id = MODEL_ID

    def __init__(self, config=None):
        cfg = dict(preset="mixed_ports", action_dt_s=0.01, episode_horizon_s=0.1, max_step_s=0.0005)
        if config is not None:
            if not isinstance(config, dict) or set(config) - set(cfg):
                raise ValueError("Unknown EOS episode configuration")
            cfg.update(config)
        if cfg["preset"] not in PRESETS:
            raise ValueError("Unknown EOS preset")
        for name, low, high in [
            ("action_dt_s", 0.001, 0.05),
            ("episode_horizon_s", 0.001, 2.0),
            ("max_step_s", 0.0001, 0.001),
        ]:
            value = cfg[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("EOS configuration outside finite limits")
            cfg[name] = float(value)
        if math.ceil(cfg["episode_horizon_s"] / cfg["action_dt_s"] - 1e-12) > 1024:
            raise ValueError("EOS decision budget exceeded")
        self._spec = json.loads(
            (Path(__file__).parent / "presets" / f"{cfg['preset']}.json").read_text()
        )
        self._model = ControlledEOSAssembly(self._spec, max_step_s=cfg["max_step_s"])
        state = self._model.snapshot()
        names = tuple(state["active_heat_power_w"])
        self.config = EpisodeConfig(**cfg, action_channels=names)
        self.action_units = {name: "W" for name in names}
        self.action_space = spaces.Box(
            low=-10000.0, high=10000.0, shape=(len(names),), dtype=np.float64
        )
        self._ports = tuple(state["port_temperatures_k"])
        self._steps = 0
        self._history = []
        self._done = False
        self._reason = "running"
        self._cache = np.array([state["port_temperatures_k"][p] for p in self._ports], dtype=float)

    def read_observation(self):
        return self._cache.copy()

    def observation_metadata(self):
        time = self._model.snapshot()["time_s"]
        return dict(
            channels=[dict(name=n, unit="K") for n in self._ports],
            valid=[True] * len(self._ports),
            sample_time_s=[time] * len(self._ports),
            saturated=[False] * len(self._ports),
            time_s=time,
        )

    def validate_action(self, action):
        if not isinstance(action, dict) or set(action) != set(self.config.action_channels):
            raise ValueError("EOS action requires every heat port exactly once")
        result = {}
        for name in self.config.action_channels:
            value = action[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not -10000 <= value <= 10000
            ):
                raise ValueError("EOS heat power outside [-10000,10000] W")
            result[name] = float(value)
        return result

    def step(self, action):
        if self._done:
            raise ValueError("EOS episode completed")
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError("EOS action must be a numeric vector")
        array = np.asarray(action)
        if (
            array.shape != self.action_space.shape
            or array.dtype.kind not in "fiu"
            or any(isinstance(v, (bool, np.bool_)) for v in action)
        ):
            raise ValueError("EOS action shape or type mismatch")
        powers = self.validate_action(
            dict(zip(self.config.action_channels, array.astype(float).tolist()))
        )
        before = self._model.snapshot()
        end = min((self._steps + 1) * self.config.action_dt_s, self.config.episode_horizon_s)
        if end <= before["time_s"]:
            raise ValueError("EOS episode clock cannot advance")
        checkpoint = self._model.checkpoint()
        try:
            after = self._model.step(
                powers,
                end - before["time_s"],
                expected_revision=before["revision"],
                command_id=f"agent.step-{self._steps + 1}",
            )
            if after["time_s"] != end:
                raise EpisodeRejected("clock_alignment", "EOS decision clock mismatch")
        except InertialGraphError as exc:
            raise EpisodeRejected(
                exc.report["criterion"], "EOS step rejected; accepted state retained"
            ) from exc
        except NetworkError as exc:
            raise EpisodeRejected(
                "numerical_contract", "EOS numerical contract rejected the step"
            ) from exc
        except Exception:
            self._model.restore(checkpoint)
            raise
        self._history.append(dict(t=before["time_s"], dt_s=end - before["time_s"], commands=powers))
        self._steps += 1
        self._done = end >= self.config.episode_horizon_s
        self._reason = "time_limit" if self._done else "running"
        self._cache = np.array([after["port_temperatures_k"][p] for p in self._ports], dtype=float)
        return self.read_observation(), 0.0, False, self._done, dict(reason=self._reason)

    def export_episode(self):
        return dict(
            schema_version=1,
            contract_version=CONTRACT,
            config=self.config.as_dict(),
            base_config=self.config.as_dict(),
            model_checkpoint=self._model.checkpoint(),
            observation=self._cache.tolist(),
            steps=self._steps,
            done=self._done,
            reason=self._reason,
            reward_total=0.0,
            action_log=json.loads(json.dumps(self._history)),
            failure_log=[],
        )

    @classmethod
    def from_episode(cls, p):
        _keys(
            p,
            (
                "schema_version",
                "contract_version",
                "config",
                "base_config",
                "model_checkpoint",
                "observation",
                "steps",
                "done",
                "reason",
                "reward_total",
                "action_log",
                "failure_log",
            ),
        )
        if (
            type(p["schema_version"]) is not int
            or p["schema_version"] != 1
            or p["contract_version"] != CONTRACT
            or p["config"] != p["base_config"]
        ):
            raise ValueError("EOS episode identity mismatch")
        result = cls(p["config"])
        steps = p["steps"]
        history = p["action_log"]
        if (
            type(steps) is not int
            or not 0 <= steps <= 1024
            or not isinstance(history, list)
            or len(history) != steps
            or type(p["done"]) is not bool
            or type(p["reward_total"]) not in (int, float)
            or p["reward_total"] != 0
            or p["failure_log"] != []
        ):
            raise ValueError("EOS episode counters mismatch")
        result._model.restore(p["model_checkpoint"])
        state = result._model.snapshot()
        if state["revision"] != steps or len(state["history"]) != steps:
            raise ValueError("EOS model/control counter mismatch")
        current = 0.0
        for i, (row, receipt) in enumerate(zip(history, state["history"]), 1):
            _keys(row, ("t", "dt_s", "commands"))
            end = min(i * result.config.action_dt_s, result.config.episode_horizon_s)
            for v in [row["t"], row["dt_s"]]:
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ValueError("Malformed EOS action clock")
            if row["t"] != current or row["dt_s"] != end - current or end <= current:
                raise ValueError("EOS action history clock mismatch")
            powers = result.validate_action(row["commands"])
            if (
                receipt["sequence"] != i
                or receipt["time_s"] != current
                or receipt["commit_revision"] != i
                or receipt["command_id"] != f"agent.step-{i}"
                or receipt["heat_power_w"] != powers
            ):
                raise ValueError("EOS native control receipts mismatch")
            current = end
        done = current == result.config.episode_horizon_s
        reason = "time_limit" if done else "running"
        if state["time_s"] != current or p["done"] != done or p["reason"] != reason:
            raise ValueError("EOS completion mismatch")
        obs = [state["port_temperatures_k"][n] for n in result._ports]
        if (
            not isinstance(p["observation"], list)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in p["observation"])
            or p["observation"] != obs
        ):
            raise ValueError("EOS cached observation mismatch")
        result._cache = np.array(obs)
        result._steps = steps
        result._history = json.loads(json.dumps(history))
        result._done = done
        result._reason = reason
        return result


def runtime_provenance():
    from scripts.controlled_eos_assembly import signature

    h = hashlib.sha256()
    root = Path(__file__).parent
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".json"):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
    return dict(public_runtime_sha256=h.hexdigest(), native_signature=signature())
