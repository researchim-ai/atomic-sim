"""Two stateless deterministic baselines using only named public observations.

Only built-in policy IDs are admitted. No module, shell or user code is loaded.
"""

from atomic_model_workspace.common import keys, number


def prepare_policy(spec, env):
    if (
        not isinstance(spec, dict)
        or spec.get("schema_version") != 1
        or type(spec.get("schema_version")) is not int
    ):
        raise ValueError("Expected policy schema1")
    if spec.get("kind") == "hold-v1":
        keys(spec, ("schema_version", "kind", "heat_power_w"))
        return dict(
            schema_version=1, kind="hold-v1", heat_power_w=env.validate_action(spec["heat_power_w"])
        )
    keys(
        spec,
        (
            "schema_version",
            "kind",
            "sensor",
            "actuator",
            "target_k",
            "gain_w_k",
            "bias_w",
            "limit_w",
        ),
    )
    names = [c["name"] for c in env.observation_metadata()["channels"]]
    if (
        spec["kind"] != "temperature-p-v1"
        or spec["sensor"] not in names
        or spec["actuator"] not in env.config.action_channels
    ):
        raise ValueError("Unsupported policy or observation/action channel")
    return dict(
        schema_version=1,
        kind=spec["kind"],
        sensor=spec["sensor"],
        actuator=spec["actuator"],
        target_k=number(spec["target_k"], 1, 10000, "target_k"),
        gain_w_k=number(spec["gain_w_k"], -1e6, 1e6, "gain_w_k"),
        bias_w=number(spec["bias_w"], -10000, 10000, "bias_w"),
        limit_w=number(spec["limit_w"], 0, 10000, "limit_w"),
    )


def observation_frame(env):
    meta = env.observation_metadata()
    return dict(
        time_s=meta["time_s"],
        channels=[
            dict(**c, value=float(v), valid=valid, sample_time_s=t)
            for c, v, valid, t in zip(
                meta["channels"], env.read_observation(), meta["valid"], meta["sample_time_s"]
            )
        ],
    )


def decide(policy, frame, action_channels):
    if policy["kind"] == "hold-v1":
        return policy["heat_power_w"].copy()
    sensor = next(c for c in frame["channels"] if c["name"] == policy["sensor"])
    if not sensor["valid"] or sensor["sample_time_s"] != frame["time_s"] or sensor["unit"] != "K":
        raise ValueError("Feedback requires a valid current temperature observation")
    heat = policy["bias_w"] + policy["gain_w_k"] * (policy["target_k"] - sensor["value"])
    action = dict.fromkeys(action_channels, 0.0)
    action[policy["actuator"]] = max(-policy["limit_w"], min(policy["limit_w"], heat))
    return action
