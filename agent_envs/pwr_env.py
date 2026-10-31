"""Versioned PWR environment: cached measurements, episode clocks and replay.

The observation Box preserves the original four-channel layout. Measurement
quality and timestamps are public metadata; full plant state is evaluator-only.
Python policies are trusted in-process; sandboxing belongs to the job runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent_envs.config import ACTION_BOUNDS, OBSERVATION_SCHEMA, EnvConfig
from core.status import DomainError, ModelLimitError, SolverError, classify_model_error
from instrumentation.channels import Sensor
from instrumentation.replay import rng_from_payload, rng_payload
from plants.vver import VVERPlant
from sdk.api import apply_commands, export_snapshot, restore_snapshot

FAILURE_REWARD = -10.0


def step_reward(power_fraction: float, failed: bool) -> float:
    return FAILURE_REWARD if failed else -abs(float(power_fraction) - 1.0)


class PWRPlantEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_dt_s=0.2, sensor_tau_s=0.5, sensor_noise=0.0, *, config=None):
        super().__init__()
        if config is not None and (action_dt_s, sensor_tau_s, sensor_noise) != (0.2, 0.5, 0.0):
            raise ValueError("Use config or legacy constructor arguments, not both")
        self.base_config = (
            (EnvConfig.from_dict(config) if isinstance(config, dict) else config)
            if config is not None
            else EnvConfig(
                action_dt_s=action_dt_s, sensor_tau_s=sensor_tau_s, sensor_noise=sensor_noise
            )
        )
        if not isinstance(self.base_config, EnvConfig):
            raise ValueError("config must be EnvConfig or a mapping")
        self.base_config = EnvConfig.from_dict(self.base_config.as_dict())
        self.config = self.base_config
        exact_low = np.array([ACTION_BOUNDS[k][0] for k in self.config.action_channels])
        exact_high = np.array([ACTION_BOUNDS[k][1] for k in self.config.action_channels])
        low, high = exact_low.astype(np.float32), exact_high.astype(np.float32)
        low = np.where(low.astype(float) < exact_low, np.nextafter(low, np.float32(np.inf)), low)
        high = np.where(
            high.astype(float) > exact_high, np.nextafter(high, np.float32(-np.inf)), high
        )
        self.action_space = spaces.Box(
            low,
            high,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([0.0, 400.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([1.5, 2000.0, 20.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.plant = VVERPlant()
        self._steps = 0
        self._done = False
        self._initialized = False
        self.sensor_saturated = False
        self._bind_sensors()
        self._cached = self._truth().astype(np.float32)

    @property
    def action_dt_s(self):
        return self.config.action_dt_s

    @property
    def sensor_tau_s(self):
        return self.config.sensor_tau_s

    @property
    def sensor_noise(self):
        return self.config.sensor_noise

    def _truth(self):
        return np.array(
            [
                self.plant.thermal_power_w_from_state(self.plant.pack())
                / self.plant.fission_power_nom,
                self.plant.fuel_temperature,
                self.plant.network.volumes["pressurizer"].pressure_pa / 1e6,
                self.plant.rod,
            ]
        )

    def _bind_sensors(self):
        self._sensors = [
            Sensor(
                self.config.sensor_tau_s,
                value=float(v),
                noise_std=self.config.sensor_noise if i < 3 else 0.0,
                fault=self.config.sensor_faults[i],
            )
            for i, v in enumerate(self._truth())
        ]
        self.power_sensor, self.fuel_sensor, self.pressure_sensor, self.rod_sensor = self._sensors
        self._valid = [fault == "" for fault in self.config.sensor_faults]
        self._sample_times = [0.0] * 4
        self._saturation = [False] * 4

    def _sample(self, dt):
        raw = []
        for i, (sensor, truth) in enumerate(zip(self._sensors, self._truth())):
            value = sensor.step(float(truth), dt, self.np_random)
            good = sensor.quality == "good" and np.isfinite(value)
            self._valid[i] = bool(good)
            if good:
                self._sample_times[i] = self.plant.time_s
            raw.append(value if np.isfinite(value) else float(self._cached[i]))
        raw = np.asarray(raw)
        clipped = np.clip(raw, self.observation_space.low, self.observation_space.high)
        self._saturation = (clipped != raw).tolist()
        self.sensor_saturated = any(self._saturation)
        self._cached = clipped.astype(np.float32)

    def _observe(self):
        return self._cached.copy()

    def read_observation(self):
        """Read without advancing physics, sensors, clocks or RNG."""
        return self._observe()

    def observation_metadata(self):
        return dict(
            schema_version=1,
            channels=deepcopy(list(OBSERVATION_SCHEMA)),
            valid=list(self._valid),
            sample_time_s=list(self._sample_times),
            saturated=list(self._saturation),
            time_s=self.plant.time_s,
        )

    def _info(self, reason="running"):
        return dict(
            out_of_domain=False,
            applicability="in_domain",
            reason=reason,
            sensor_saturated=self.sensor_saturated,
            measurement=self.observation_metadata(),
            constraint_power_error=abs(float(self._cached[0]) - self.config.target_power_fraction),
            scram_latched=bool(self.plant.scram_latched),
        )

    def reset(self, *, seed=None, options=None):
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be an object")
        options = options or {}
        allowed = {
            "initial_commands",
            "timeline",
            "target_power_fraction",
            "episode_horizon_s",
            "sensor_faults",
        }
        if set(options) - allowed:
            raise ValueError("Unknown reset options")
        config = replace(self.base_config, **options)
        seed = seed if seed is not None else config.seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be a nonnegative integer or None")
        # Validate and initialize off to the side before mutating this episode.
        plant = VVERPlant()
        apply_commands(plant, config.initial_commands)
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.config, self.plant = config, plant
        self._steps, self._event_index, self._done = 0, 0, False
        self._reason = "running"
        self._episode_seed = seed
        self._failure_log = []
        self._initialized = True
        self._reward_total = 0.0
        self._metrics = dict(tracking_iae_s=0.0, action_cost=0.0, violation_time_s=0.0)
        self._action_log = []
        self._event_log = []
        self._bind_sensors()
        self._cached = np.clip(
            self._truth(), self.observation_space.low, self.observation_space.high
        ).astype(np.float32)
        self.sensor_saturated = False
        self._apply_events()
        # No lag advance or random draw at reset, including t=0 events.
        self._cached = np.clip(
            self._truth(), self.observation_space.low, self.observation_space.high
        ).astype(np.float32)
        return self._observe(), self._info()

    def _apply_events(self):
        while self._event_index < len(self.config.timeline):
            event = self.config.timeline[self._event_index]
            if event["t"] > self.plant.time_s:
                break
            apply_commands(self.plant, event["commands"])
            self._event_log.append(deepcopy(event))
            self._event_index += 1

    def _action(self, action):
        array = np.asarray(action)
        if array.shape != self.action_space.shape or array.dtype.kind not in "fiu":
            raise ValueError("Action must have exactly the declared numeric shape")
        values = array.astype(float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Action must be finite")
        commands = dict(zip(self.config.action_channels, values.tolist()))
        from sdk.api import validate_commands

        return validate_commands(commands)

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError("reset required before step or after episode completion")
        commands = self._action(action)
        # Save the entire accepted decision boundary, not just the last substep.
        physical = self.plant.snapshot()
        saved = {
            k: deepcopy(getattr(self, k))
            for k in (
                "_sensors",
                "_cached",
                "_valid",
                "_sample_times",
                "_saturation",
                "_event_index",
                "_event_log",
                "sensor_saturated",
            )
        }
        rng = deepcopy(self.np_random.bit_generator.state)
        ledgers = {
            k: deepcopy(getattr(self.plant, k))
            for k in (
                "last_energy_ledger",
                "last_protection_event_s",
                "last_motion_events",
                "applicability",
            )
        }
        start = self.plant.time_s
        target = min(start + self.config.action_dt_s, self.config.episode_horizon_s)
        try:
            apply_commands(self.plant, commands)
            while self.plant.time_s < target:
                self._apply_events()
                end = target
                if self._event_index < len(self.config.timeline):
                    end = min(end, self.config.timeline[self._event_index]["t"])
                dt = end - self.plant.time_s
                self.plant.advance(
                    dt, method=self.config.method, rtol=self.config.rtol, atol=self.config.atol
                )
                self._sample(dt)
            self._apply_events()
        except Exception as exc:
            self.plant.restore(physical)
            for key, value in saved.items():
                setattr(self, key, value)
            self.power_sensor, self.fuel_sensor, self.pressure_sensor, self.rod_sensor = (
                self._sensors
            )
            self.np_random.bit_generator.state = rng
            for key, value in ledgers.items():
                setattr(self.plant, key, value)
            if not isinstance(exc, (DomainError, SolverError)):
                raise  # Programming errors must be visible, never rewarded as physics.
            self._done = True
            self._reward_total += FAILURE_REWARD
            reason = (
                "model_limit"
                if isinstance(exc, ModelLimitError)
                else ("domain_error" if isinstance(exc, DomainError) else "solver_error")
            )
            self._reason = reason
            self._failure_log.append(dict(t=start, commands=commands, reason=reason))
            info = self._info(reason)
            info.update(
                out_of_domain=isinstance(exc, DomainError),
                applicability=exc.applicability.value
                if isinstance(exc, DomainError)
                else "solver_error",
                error_type=type(exc).__name__,
                error_class=classify_model_error(exc),
                message=str(exc),
                reward_terms={"failure": FAILURE_REWARD},
                episode_complete=False,
            )
            return self._observe(), FAILURE_REWARD, False, True, info
        dt = self.plant.time_s - start
        self._steps += 1
        error = abs(float(self._truth()[0]) - self.config.target_power_fraction)
        cost = (
            sum((v / (ACTION_BOUNDS[k][1] - ACTION_BOUNDS[k][0])) ** 2 for k, v in commands.items())
            * dt
        )
        violation = dt if error > self.config.power_error_limit else 0.0
        terms = dict(
            tracking=-error * dt,
            action=-self.config.action_cost_weight * cost,
            violation=-self.config.violation_weight * violation,
        )
        reward = float(sum(terms.values()))
        self._reward_total += reward
        self._metrics["tracking_iae_s"] += error * dt
        self._metrics["action_cost"] += cost
        self._metrics["violation_time_s"] += violation
        self._action_log.append(dict(t=start, dt_s=dt, commands=commands))
        terminated = bool(self.config.task == "stop_on_scram" and self.plant.scram_latched)
        truncated = bool(not terminated and self.plant.time_s >= self.config.episode_horizon_s)
        self._done = terminated or truncated
        self._reason = "task_terminal" if terminated else "time_limit" if truncated else "running"
        info = self._info(
            "task_terminal" if terminated else "time_limit" if truncated else "running"
        )
        info.update(reward_terms=terms, episode_complete=bool(terminated or truncated))
        return self._observe(), reward, terminated, truncated, info

    def evaluator_metrics(self):
        """Trusted evaluator API. Completion must rank before cumulative reward."""
        return dict(
            self._metrics,
            reward_total=self._reward_total,
            steps=self._steps,
            completed_horizon=self.plant.time_s >= self.config.episode_horizon_s,
            reason=self._reason,
            eligible_for_ranking=self._reason in ("task_terminal", "time_limit"),
            ranking_key=[int(self._reason in ("task_terminal", "time_limit")), self._reward_total],
            time_s=self.plant.time_s,
        )

    def export_episode(self):
        if not self._initialized:
            raise RuntimeError("reset required before exporting episode")
        return dict(
            schema_version=1,
            contract_version="2.1.0",
            episode_seed=self._episode_seed,
            base_config=self.base_config.as_dict(),
            config=self.config.as_dict(),
            plant=export_snapshot(self.plant),
            sensors=[asdict(s) for s in self._sensors],
            rng=rng_payload(self.np_random),
            action_rng=rng_payload(self.action_space.np_random),
            observation=self._cached.tolist(),
            valid=list(self._valid),
            sample_times=list(self._sample_times),
            saturation=list(self._saturation),
            steps=self._steps,
            event_index=self._event_index,
            done=self._done,
            reason=self._reason,
            failure_log=deepcopy(self._failure_log),
            reward_total=self._reward_total,
            metrics=deepcopy(self._metrics),
            action_log=deepcopy(self._action_log),
            event_log=deepcopy(self._event_log),
        )

    @classmethod
    def from_episode(cls, payload):
        # Strict finite JSON tree, with schema and config validation before restore.
        import json

        if (
            not isinstance(payload, dict)
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
        ):
            raise ValueError("Unsupported episode schema")
        json.dumps(payload, allow_nan=False)
        env = cls(config=EnvConfig.from_dict(payload["base_config"]))
        env.reset(seed=0)
        expected = set(env.export_episode())
        if set(payload) != expected:
            raise ValueError("Unexpected or missing episode fields")
        if payload["contract_version"] != "2.1.0":
            raise ValueError("Episode contract version mismatch")
        from agent_envs.episode_validation import validate_episode_metadata

        config = EnvConfig.from_dict(payload["config"])
        if config.action_channels != env.base_config.action_channels:
            raise ValueError("Episode action schema mismatch")
        mutable = {
            "initial_commands",
            "timeline",
            "target_power_fraction",
            "episode_horizon_s",
            "sensor_faults",
        }
        if any(
            value != env.base_config.as_dict()[key]
            for key, value in config.as_dict().items()
            if key not in mutable
        ):
            raise ValueError("Episode immutable configuration mismatch")
        restore_snapshot(env.plant, payload["plant"])
        validate_episode_metadata(payload)
        env.config = config
        if (
            type(payload["done"]) is not bool
            or type(payload["steps"]) is not int
            or payload["steps"] < 0
        ):
            raise ValueError("Invalid episode counters")
        if type(payload["event_index"]) is not int or not 0 <= payload["event_index"] <= len(
            config.timeline
        ):
            raise ValueError("Invalid event index")
        if len(payload["sensors"]) != 4:
            raise ValueError("Four sensor states required")
        sensors = []
        for s in payload["sensors"]:
            if set(s) != set(asdict(Sensor(1.0))):
                raise ValueError("Invalid sensor state fields")
            sensor = Sensor(**s)
            sensor.held = s["held"]
            sensors.append(sensor)
        obs = np.array(payload["observation"], dtype=np.float32)
        if not env.observation_space.contains(obs):
            raise ValueError("Invalid cached observation")
        for key in ("valid", "saturation"):
            if len(payload[key]) != 4 or any(type(x) is not bool for x in payload[key]):
                raise ValueError("Invalid measurement flags")
        if len(payload["sample_times"]) != 4 or any(
            isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= env.plant.time_s
            for t in payload["sample_times"]
        ):
            raise ValueError("Invalid measurement timestamps")
        if not 0 <= env.plant.time_s <= config.episode_horizon_s:
            raise ValueError("Episode clock outside horizon")
        env._sensors = sensors
        env.power_sensor, env.fuel_sensor, env.pressure_sensor, env.rod_sensor = sensors
        env._np_random = rng_from_payload(payload["rng"])
        env._np_random_seed = -1
        env.action_space._np_random = rng_from_payload(payload["action_rng"])
        env._cached, env._valid = obs, list(payload["valid"])
        env._sample_times = list(payload["sample_times"])
        env._saturation = list(payload["saturation"])
        env.sensor_saturated = any(env._saturation)
        env._steps, env._event_index, env._done = (
            payload["steps"],
            payload["event_index"],
            payload["done"],
        )
        env._reward_total = payload["reward_total"]
        env._reason = payload["reason"]
        env._episode_seed = payload["episode_seed"]
        env._failure_log = deepcopy(payload["failure_log"])
        env._metrics = deepcopy(payload["metrics"])
        env._action_log, env._event_log = (
            deepcopy(payload["action_log"]),
            deepcopy(payload["event_log"]),
        )
        return env
