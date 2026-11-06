"""JSON Schema tool descriptors and strict validation of the emitted subset."""

import json
import math
import re
from copy import deepcopy

from agent_envs.config import ACTION_BOUNDS, OBSERVATION_SCHEMA
from atomic_agent_tools.controlled import action_channels, model_card

PROTOCOL_VERSION = "1.2.0"
LIMITS = dict(
    max_sessions=8,
    max_checkpoints=32,
    max_checkpoint_bytes=4 * 1024 * 1024,
    max_checkpoint_total_bytes=16 * 1024 * 1024,
    max_requests=4096,
    max_model_transactions=4096,
    max_episode_decisions=1024,
    max_line_bytes=65536,
)
UNITS = dict(
    rod_command="1/s", throttle="1", feed_bias_kg_s="kg/s", boron_ppm="ppm", sink_temperature_k="K"
)


class ToolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def obj(properties, required=()):
    return dict(
        type="object", properties=properties, required=list(required), additionalProperties=False
    )


def bounded(low, high, kind="number"):
    return dict(type=kind, minimum=low, maximum=high)


SESSION = dict(type="string", pattern="^[A-Za-z][A-Za-z0-9_-]{0,63}$", maxLength=64)
HASH = dict(type="string", pattern="^[0-9a-f]{64}$", maxLength=64)
GUARD = dict(expected_revision=bounded(0, 2**53 - 1, "integer"), observation_id=HASH)
CONFIG = obj(
    dict(
        seed=bounded(0, 2**63 - 1, "integer"),
        task=dict(type="string", enum=["hold_power", "stop_on_scram"]),
        episode_horizon_s=bounded(0.001, 10.0),
        action_dt_s=bounded(0.001, 0.05),
        target_power_fraction=bounded(0, 1.5),
        sensor_tau_s=bounded(1e-6, 1e3),
        sensor_noise=bounded(0, 1.0),
        action_channels=dict(
            type="array",
            items=dict(type="string", enum=list(ACTION_BOUNDS)),
            minItems=1,
            maxItems=5,
            uniqueItems=True,
        ),
        sensor_faults=dict(
            type="array",
            items=dict(type="string", enum=["", "stuck", "open"]),
            minItems=4,
            maxItems=4,
        ),
    )
)
ACTION = obj({name: bounded(*bounds) for name, bounds in ACTION_BOUNDS.items()})
ACTION["minProperties"] = 1
EOS_CONFIG = obj(
    dict(
        preset=dict(type="string", enum=["mixed_ports", "uniform", "gravity"]),
        action_dt_s=bounded(0.001, 0.05),
        episode_horizon_s=bounded(0.001, 2.0),
        max_step_s=bounded(0.0001, 0.001),
    )
)
GENERIC_ACTION = dict(
    type="object",
    minProperties=1,
    maxProperties=84,
    additionalProperties=dict(type="number", minimum=-100000.0, maximum=100000.0),
)
CONTROLLED_CONFIG = obj(
    dict(
        duration_s=dict(type="number", exclusiveMinimum=0, maximum=5),
        max_steps=bounded(1, 256, "integer"),
        tier=dict(type="string", enum=["training", "reference"]),
        seed=bounded(0, 2**32 - 1, "integer"),
    )
)
# Shared SDK bounds; Boolean flags remain booleans in JSON and JSON Schema.

CONTROLLED_ACTION = obj(
    {
        name: {k: v for k, v in channel.items() if k != "unit"}
        for name, channel in action_channels().items()
    }
)
SCHEMAS = {
    "describe_simulator": obj({}),
    "create_controlled_episode": obj(
        dict(session_id=SESSION, config=CONTROLLED_CONFIG), ["session_id"]
    ),
    "step_controlled_episode": obj(
        dict(session_id=SESSION, action=CONTROLLED_ACTION, **GUARD),
        ["session_id", "action", *GUARD],
    ),
    "export_controlled_case": obj(
        dict(
            session_id=SESSION,
            case_id=dict(type="string", minLength=1, maxLength=80, pattern="^[A-Za-z0-9_-]+$"),
            family=dict(type="string", minLength=1, maxLength=80, pattern="^[A-Za-z0-9_-]+$"),
            split=dict(type="string", enum=["calibration", "validation"]),
            **GUARD,
        ),
        ["session_id", "case_id", "family", *GUARD],
    ),
    "create_episode": obj(dict(session_id=SESSION, config=CONFIG), ["session_id"]),
    "create_eos_episode": obj(dict(session_id=SESSION, config=EOS_CONFIG), ["session_id"]),
    "inspect_episode": obj(dict(session_id=SESSION), ["session_id"]),
    "step_episode": obj(
        dict(session_id=SESSION, action=GENERIC_ACTION, **GUARD), ["session_id", "action", *GUARD]
    ),
    "save_checkpoint": obj(dict(session_id=SESSION, **GUARD), ["session_id", *GUARD]),
    "restore_checkpoint": obj(
        dict(session_id=SESSION, checkpoint_id=HASH, **GUARD),
        ["session_id", "checkpoint_id", *GUARD],
    ),
    "cancel_episode": obj(dict(session_id=SESSION, **GUARD), ["session_id", *GUARD]),
    "close_episode": obj(dict(session_id=SESSION, **GUARD), ["session_id", *GUARD]),
    "forget_checkpoint": obj(dict(checkpoint_id=HASH), ["checkpoint_id"]),
}
DESCRIPTIONS = {
    "export_controlled_case": "Export accepted commands as a bounded training-runtime case, without file access or a physical-validation claim.",
    "create_controlled_episode": "Create a deterministic controlled research PWR episode with explicit full model state, not sensor measurements.",
    "step_controlled_episode": "Apply partial persistent SDK commands to the controlled PWR. Empty action holds settings. No built-in reward.",
    "create_eos_episode": "Create a deterministic experimental EOS thermal episode from a packaged preset; not a reactor plant.",
    "describe_simulator": "List supported model, observations, physical action units, limits and tool schemas.",
    "create_episode": "Create a seeded reduced PWR episode. Sensor observations only; no hidden simulator state.",
    "inspect_episode": "Read cached measurements without advancing physics, sensors or random generators.",
    "step_episode": "Apply one complete named action in physical units, using the latest observation token.",
    "save_checkpoint": "Save model/sensor/RNG state behind an opaque handle; returns no hidden state.",
    "restore_checkpoint": "Restore a saved handle. Session revision advances; old observation tokens become stale.",
    "cancel_episode": "Cancel between calls. Does not interrupt an already running native solver.",
    "close_episode": "Release the session. Reusing its name creates a new observation-token generation.",
    "forget_checkpoint": "Release an opaque checkpoint handle. Does not alter any simulation.",
}
ENVELOPE = obj(
    dict(
        request_id=dict(type="string", minLength=1, maxLength=128),
        tool=dict(type="string", enum=list(SCHEMAS)),
        arguments=dict(type="object"),
    ),
    ["request_id", "tool", "arguments"],
)


def validate(schema, value, path="arguments"):
    """Validate only constructs used in these descriptors, not arbitrary schemas."""
    kind = schema["type"]
    good = (
        isinstance(value, dict)
        if kind == "object"
        else isinstance(value, list)
        if kind == "array"
        else isinstance(value, str)
        if kind == "string"
        else type(value) is bool
        if kind == "boolean"
        else type(value) is int
        if kind == "integer"
        else type(value) in (int, float)
    )
    if not good:
        raise ToolError("invalid_argument", f"{path}: expected {kind}")
    if kind in ["number", "integer"]:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if (
            not finite
            or value < schema.get("minimum", -math.inf)
            or value > schema.get("maximum", math.inf)
            or ("exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"])
        ):
            raise ToolError("invalid_argument", f"{path}: outside finite bounds")
    if kind == "object":
        if len(value) > schema.get("maxProperties", 2**31) or any(
            not isinstance(k, str) or len(k) > 128 for k in value
        ):
            raise ToolError("invalid_argument", f"{path}: invalid object keys or size")
        if isinstance(schema.get("additionalProperties"), dict):
            for key, item in value.items():
                validate(schema["additionalProperties"], item, path + "." + key)
        allowed = schema.get("properties")
        if allowed is not None:
            if set(value) - set(allowed) or set(schema.get("required", [])) - set(value):
                raise ToolError("invalid_argument", f"{path}: missing or unknown fields")
            for key, item in value.items():
                validate(allowed[key], item, path + "." + key)
        if len(value) < schema.get("minProperties", 0):
            raise ToolError("invalid_argument", f"{path}: empty object")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 2**31):
            raise ToolError("invalid_argument", f"{path}: wrong array length")
        for i, item in enumerate(value):
            validate(schema["items"], item, f"{path}[{i}]")
        if schema.get("uniqueItems") and len({canonical(v) for v in value}) != len(value):
            raise ToolError("invalid_argument", f"{path}: duplicate array entries")
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 2**31) or (
            "pattern" in schema and not re.fullmatch(schema["pattern"], value)
        ):
            raise ToolError("invalid_argument", f"{path}: invalid string")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError("invalid_argument", f"{path}: unsupported value")


def catalog():
    return dict(
        protocol_version=PROTOCOL_VERSION,
        transport="local JSON Lines; not MCP or JSON-RPC",
        limits=deepcopy(LIMITS),
        controlled_pwr=dict(
            model=model_card(),
            action_channels=action_channels(),
            config_schema=deepcopy(CONTROLLED_CONFIG),
        ),
        observation_schema=deepcopy(list(OBSERVATION_SCHEMA)),
        action_channels={
            k: dict(unit=UNITS[k], minimum=b[0], maximum=b[1]) for k, b in ACTION_BOUNDS.items()
        },
        tools=[
            dict(
                name=name,
                description=DESCRIPTIONS[name],
                input_schema=dict(
                    deepcopy(SCHEMAS[name]),
                    **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
                ),
            )
            for name in SCHEMAS
        ],
    )
