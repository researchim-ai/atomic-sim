"""Data-only permissions, fresh observation binding and finite typed actions."""

from atomic_model_workspace.common import canonical, digest, identifier, keys, loads, number
from atomic_model_workspace.policies import observation_frame

MAX_STATE = 32 * 1024


def state(value):
    raw = canonical(value).encode()
    if len(raw) > MAX_STATE:
        raise ValueError("agent_state_limit")
    return loads(raw)


def descriptor(value, env):
    keys(
        value,
        (
            "schema_version",
            "agent_id",
            "agent_version",
            "observations",
            "actions",
            "initial_state",
            "decision_timeout_s",
            "rss_limit_bytes",
        ),
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("agent_schema")
    identifier(value["agent_id"])
    identifier(value["agent_version"])
    permitted = {c["name"] for c in env.observation_metadata()["channels"]}
    for name, valid in [("observations", permitted), ("actions", set(env.config.action_channels))]:
        chosen = value[name]
        if (
            not isinstance(chosen, list)
            or not chosen
            or any(not isinstance(v, str) for v in chosen)
            or len(set(chosen)) != len(chosen)
            or not set(chosen) <= valid
        ):
            raise ValueError("agent_permissions")
    rss = value["rss_limit_bytes"]
    if type(rss) is not int or not 16 * 1024**2 <= rss <= 16 * 1024**3:
        raise ValueError("agent_rss_budget")
    return dict(
        schema_version=1,
        agent_id=value["agent_id"],
        agent_version=value["agent_version"],
        observations=sorted(value["observations"]),
        actions=sorted(value["actions"]),
        initial_state=state(value["initial_state"]),
        decision_timeout_s=number(value["decision_timeout_s"], 0.05, 60, "decision_timeout_s"),
        rss_limit_bytes=rss,
    )


def request(env, desc, agent_state, job_id):
    frame = observation_frame(env)
    channels = [c for c in frame["channels"] if c["name"] in desc["observations"]]
    if any(not c["valid"] or c["sample_time_s"] != frame["time_s"] for c in channels):
        raise ValueError("agent_stale_observation")
    observation = dict(time_s=frame["time_s"], channels=channels)
    sha = digest(observation)
    value = dict(
        schema_version=1,
        decision=env._steps,
        observation_sha256=sha,
        observation=observation,
        allowed_actions=[dict(name=n, unit="W") for n in desc["actions"]],
        state=state(agent_state),
    )
    return dict(value, request_id=digest(dict(job_id=job_id, request=value)))


def response(value, req, env, desc):
    keys(
        value,
        ("schema_version", "request_id", "decision", "observation_sha256", "heat_power_w", "state"),
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["decision"]) is not int
        or value["decision"] != req["decision"]
        or value["request_id"] != req["request_id"]
        or value["observation_sha256"] != req["observation_sha256"]
    ):
        raise ValueError("agent_stale_response")
    keys(value["heat_power_w"], desc["actions"])
    action = dict.fromkeys(env.config.action_channels, 0.0)
    action.update(value["heat_power_w"])
    action = env.validate_action(action)
    new = state(value["state"])
    return action, new
