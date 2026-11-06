"""Dependency-free explicit-state demo agent: hold or clipped temperature feedback."""

import json
import sys


def main():
    req = json.loads(sys.stdin.buffer.readline(256 * 1024 + 1))
    state = req["state"].copy()
    names = [a["name"] for a in req["allowed_actions"]]
    if state["kind"] == "hold":
        actions = {n: state["heat_power_w"].get(n, 0.0) for n in names}
    elif state["kind"] == "temperature-p":
        sensor = next(c for c in req["observation"]["channels"] if c["name"] == state["sensor"])
        power = state["bias_w"] + state["gain_w_k"] * (state["target_k"] - sensor["value"])
        actions = dict.fromkeys(names, 0.0)
        actions[state["actuator"]] = max(-state["limit_w"], min(state["limit_w"], power))
    else:
        raise ValueError("Unsupported demo agent kind")
    state["decisions"] = state.get("decisions", 0) + 1
    print(
        json.dumps(
            dict(
                schema_version=1,
                request_id=req["request_id"],
                decision=req["decision"],
                observation_sha256=req["observation_sha256"],
                heat_power_w=actions,
                state=state,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
