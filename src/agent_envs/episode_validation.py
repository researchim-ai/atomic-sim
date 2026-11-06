"""Strict finite metadata validation for portable episode checkpoints."""

from dataclasses import asdict

from agent_envs.config import EnvConfig, number
from instrumentation.channels import Sensor
from sdk.api import validate_commands

REASONS = {"running", "task_terminal", "time_limit", "domain_error", "model_limit", "solver_error"}


def validate_episode_metadata(p):
    config = EnvConfig.from_dict(p["config"])
    time = number(p["plant"]["time_s"], "episode time", 0, config.episode_horizon_s)
    if p["episode_seed"] is not None and (
        type(p["episode_seed"]) is not int or not 0 <= p["episode_seed"] < 2**63
    ):
        raise ValueError("Invalid episode seed")
    if (
        type(p["done"]) is not bool
        or p["reason"] not in REASONS
        or p["done"] != (p["reason"] != "running")
    ):
        raise ValueError("Invalid completion reason")
    if p["reason"] == "time_limit" and time != config.episode_horizon_s:
        raise ValueError("Time limit requires completed horizon")
    if p["reason"] == "running" and time >= config.episode_horizon_s:
        raise ValueError("Running episode cannot be at the horizon")
    if p["reason"] == "task_terminal" and (
        config.task != "stop_on_scram" or not p["plant"]["scram_latched"]
    ):
        raise ValueError("Task terminal requires SCRAM task and latch")
    number(p["reward_total"], "reward", -1e300, 1e300)
    if not isinstance(p["metrics"], dict) or set(p["metrics"]) != {
        "tracking_iae_s",
        "action_cost",
        "violation_time_s",
    }:
        raise ValueError("Unknown metric fields")
    for key, value in p["metrics"].items():
        number(value, key, 0, 1e300)
    if type(p["steps"]) is not int or p["steps"] < 0 or len(p["action_log"]) != p["steps"]:
        raise ValueError("Action log does not match step count")
    end = 0.0
    for row in p["action_log"]:
        if set(row) != {"t", "dt_s", "commands"} or row["t"] != end:
            raise ValueError("Invalid action log continuity")
        dt = number(row["dt_s"], "decision duration", 1e-15, config.action_dt_s + 1e-12)
        commands = validate_commands(row["commands"])
        if set(commands) != set(config.action_channels):
            raise ValueError("Action log schema mismatch")
        end = row["t"] + dt
    if abs(end - time) > 1e-12 * max(1, time):
        raise ValueError("Action log time mismatch")
    if type(p["event_index"]) is not int or not 0 <= p["event_index"] <= len(config.timeline):
        raise ValueError("Invalid event index")
    if p["event_log"] != list(config.timeline[: p["event_index"]]):
        raise ValueError("Event log does not match configuration")
    if any(e["t"] > time for e in p["event_log"]) or any(
        e["t"] <= time for e in config.timeline[p["event_index"] :]
    ):
        raise ValueError("Event cursor does not match physical time")
    failures = p["failure_log"]
    failed = p["reason"] in {"domain_error", "model_limit", "solver_error"}
    if not isinstance(failures, list) or len(failures) != int(failed):
        raise ValueError("Invalid failure history")
    if failed:
        row = failures[0]
        if (
            set(row) != {"t", "commands", "reason"}
            or row["t"] != time
            or row["reason"] != p["reason"]
        ):
            raise ValueError("Failure history mismatch")
        if set(validate_commands(row["commands"])) != set(config.action_channels):
            raise ValueError("Failure action schema mismatch")
    for sensor in p["sensors"]:
        if not isinstance(sensor, dict) or set(sensor) != set(asdict(Sensor(1.0))):
            raise ValueError("Invalid sensor state fields")
        for key in (
            "tau_s",
            "value",
            "bias",
            "noise_std",
            "quantum",
            "drift_per_s",
            "sample_period_s",
            "held",
            "sample_elapsed_s",
        ):
            low = (
                1e-15
                if key == "tau_s"
                else 0
                if key in ("noise_std", "quantum", "sample_period_s", "sample_elapsed_s")
                else -1e300
            )
            number(sensor[key], key, low, 1e300)
        if (
            type(sensor["failed"]) is not bool
            or sensor["fault"] not in ("", "stuck", "open")
            or sensor["quality"] not in ("good", "bad", "held")
        ):
            raise ValueError("Invalid sensor flags")
        if sensor["failed"] and sensor["fault"] != "stuck":
            raise ValueError("Inconsistent failed sensor state")
