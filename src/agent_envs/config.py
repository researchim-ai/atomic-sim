"""Versioned, JSON-portable episode contract for the reduced PWR engine."""

import math
from dataclasses import asdict, dataclass

from sdk.api import validate_commands

ACTION_BOUNDS = {
    "rod_command": (-1.0, 1.0),
    "throttle": (0.0, 1.0),
    "feed_bias_kg_s": (-1e5, 1e5),
    "boron_ppm": (0.0, 2e4),
    "sink_temperature_k": (273.15, 373.15),
}
OBSERVATION_SCHEMA = (
    {"name": "thermal_power_fraction", "unit": "1", "scale": "nominal thermal power"},
    {"name": "fuel_temperature", "unit": "K", "scale": "absolute"},
    {"name": "primary_pressure", "unit": "MPa", "scale": "Pa / 1e6"},
    {"name": "rod_insertion", "unit": "1", "scale": "fraction"},
)


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a number")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(name + " outside finite bounds")
    return float(value)


@dataclass(frozen=True)
class EnvConfig:
    schema_version: int = 1
    engine: str = "vver"
    seed: int | None = None
    task: str = "hold_power"
    episode_horizon_s: float = 10.0
    action_dt_s: float = 0.2
    target_power_fraction: float = 1.0
    sensor_tau_s: float = 0.5
    sensor_noise: float = 0.0
    action_channels: tuple = ("rod_command",)
    method: str = "BDF"
    rtol: float = 1e-4
    atol: float = 1e-2
    action_cost_weight: float = 0.0
    violation_weight: float = 0.0
    power_error_limit: float = 0.1
    initial_commands: dict = None
    timeline: tuple = ()
    sensor_faults: tuple = ("", "", "", "")

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported EnvConfig schema")
        if self.engine != "vver" or self.task not in ("hold_power", "stop_on_scram"):
            raise ValueError("Unsupported engine/task")
        if self.method not in ("BDF", "Radau"):
            raise ValueError("Only BDF/Radau are supported")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be a nonnegative integer or None")
        for key, low, high in (
            ("episode_horizon_s", 1e-9, 3600.0),
            ("action_dt_s", 1e-9, 5.0),
            ("target_power_fraction", 0.0, 1.5),
            ("sensor_tau_s", 1e-9, 1e6),
            ("sensor_noise", 0.0, 1e3),
            ("rtol", 1e-12, 1e-2),
            ("atol", 1e-12, 1.0),
            ("action_cost_weight", 0.0, 1e6),
            ("violation_weight", 0.0, 1e6),
            ("power_error_limit", 0.0, 1.5),
        ):
            object.__setattr__(self, key, number(getattr(self, key), key, low, high))
        channels = tuple(self.action_channels)
        if (
            not channels
            or len(set(channels)) != len(channels)
            or any(k not in ACTION_BOUNDS for k in channels)
        ):
            raise ValueError("Invalid or duplicate action channels")
        object.__setattr__(self, "action_channels", channels)
        faults = tuple(self.sensor_faults)
        if len(faults) != 4 or any(v not in ("", "stuck", "open") for v in faults):
            raise ValueError("Four valid sensor faults required")
        object.__setattr__(self, "sensor_faults", faults)
        object.__setattr__(self, "initial_commands", validate_commands(self.initial_commands))
        events = []
        previous = -1.0
        for event in self.timeline:
            if not isinstance(event, dict) or set(event) != {"t", "commands"}:
                raise ValueError("Timeline events require t and commands")
            t = number(event["t"], "event time", 0, self.episode_horizon_s)
            if t < previous:
                raise ValueError("Timeline must be ordered")
            previous = t
            events.append(dict(t=t, commands=validate_commands(event["commands"])))
        object.__setattr__(self, "timeline", tuple(events))

    def as_dict(self):
        result = asdict(self)
        for key in ("action_channels", "timeline", "sensor_faults"):
            result[key] = list(result[key])
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("EnvConfig must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("Unknown or malformed EnvConfig fields") from exc
