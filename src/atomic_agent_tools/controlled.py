"""Explicit full-state controlled-PWR tools; no sensor-only model substitution."""

import hashlib
from pathlib import Path

from atomic_model_controlled_plant.session import identity as core_identity
from atomic_model_training import schema, spec
from atomic_model_workspace.common import digest
from atomic_pwr_agent import Episode
from atomic_pwr_agent import episode as adapter
from sdk.api import _COMMANDS, _FLAGS

CONTRACT = "controlled-pwr-agent-v1"
UNITS = dict(
    rod_command="1/s", throttle="1", feed_bias_kg_s="kg/s", boron_ppm="ppm", sink_temperature_k="K"
)


def provenance():
    return dict(
        core=core_identity(),
        dependencies={
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in [("episode", adapter), ("schema", schema), ("spec", spec)]
        },
    )


def model_card():
    return dict(
        model_id="pwr-reduced-flow-controlled",
        episode_contract=CONTRACT,
        observation_kind="full_model_state",
        validated=False,
        plant_referenced=False,
        scope="reduced research PWR; deterministic nominal initialization",
    )


def action_channels():
    # Reuse the SDK's admitted domains rather than maintain a second set of limits.
    result = {
        k: dict(type="number", unit=UNITS[k], minimum=v[0], maximum=v[1])
        for k, v in _COMMANDS.items()
    }
    result.update({k: dict(type="boolean", unit="1") for k in _FLAGS})
    return result


def export_episode(env):
    return dict(contract_version=CONTRACT, checkpoint=env.checkpoint())


def transition(env):
    obs = env.observe()
    return dict(
        reward=None,
        terminated=obs["terminated"],
        truncated=obs["truncated"],
        reason="scram" if obs["terminated"] else "time_limit" if obs["truncated"] else "running",
    )


def view(name, session):
    obs = session.env.observe()
    return dict(
        session_id=name,
        revision=session.revision,
        observation_id=digest(
            dict(
                generation=session.generation,
                revision=session.revision,
                phase=session.phase,
                observation=obs,
            )
        ),
        status=session.phase,
        time_s=obs["time_s"],
        observation=[
            dict(
                name=field["name"],
                unit=field["unit"],
                value=value,
                valid=True,
                sample_time_s=obs["time_s"],
                saturated=False,
            )
            for field, value in zip(obs["state_schema"], obs["state"])
        ],
        observation_kind="full_model_state",
        action_channels=action_channels(),
        last_transition=session.transition.copy(),
        episode_contract=CONTRACT,
        model=model_card(),
        core_revision=obs["revision"],
        thermal_power_w=obs["thermal_power_w"],
    )


def restore(native, phase, last_transition):
    if not isinstance(native, dict) or set(native) != {"contract_version", "checkpoint"}:
        raise ValueError("Malformed controlled episode")
    env = Episode.restore(native["checkpoint"])
    expected = transition(env)
    done = expected["terminated"] or expected["truncated"]
    if (
        last_transition.get("reward") is not None
        or type(last_transition.get("terminated")) is not bool
        or type(last_transition.get("truncated")) is not bool
    ):
        raise ValueError("Invalid controlled transition types")
    if last_transition != expected or phase not in (
        ["completed"] if done else ["active", "cancelled"]
    ):
        raise ValueError("Inconsistent controlled lifecycle")
    return env
