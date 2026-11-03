"""External bounded proportional control; engineering demonstration only.

Run from checkout: python -m examples.online_thermal_control
"""

import json

from scripts.nonlinear_graph_references import pipe_spec
from scripts.online_thermal_graph import OnlineThermalGraph


def main():
    model = OnlineThermalGraph(pipe_spec())
    for _ in range(6):
        observation = model.observation()
        temperature = observation["model"]["state"]["temperatures_k"]["a"]
        power = max(-100000.0, min(100000.0, 10000.0 * (330.0 - temperature)))
        result = model.step(
            dict(thermal_nodes=dict(a=dict(heat_power_w=power))),
            10.0,
            expected_revision=observation["revision"],
        )
        state = result["observation"]["model"]["state"]
        print(
            json.dumps(
                dict(
                    revision=model.revision,
                    time_s=state["time_s"],
                    temperature_k=state["temperatures_k"]["a"],
                    power_w=power,
                )
            )
        )


if __name__ == "__main__":
    main()
