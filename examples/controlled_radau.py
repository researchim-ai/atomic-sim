"""Scheduled and online command restart on the registered reduced Radau graph."""

import json

from scripts.controlled_reduced_engine import OnlineReducedGraph, ScheduledReducedGraph
from scripts.nonlinear_graph_references import pipe_spec
from scripts.scheduled_graph_references import heater_spec


def main():
    scheduled = ScheduledReducedGraph(heater_spec())
    scheduled.advance(60.0)
    resumed = ScheduledReducedGraph(heater_spec())
    resumed.restore(scheduled.checkpoint())
    final = resumed.advance(60.0)
    online = OnlineReducedGraph(pipe_spec())
    online.step(dict(thermal_nodes=dict(a=dict(heat_power_w=1000.0))), 7.5, expected_revision=0)
    other = OnlineReducedGraph(pipe_spec())
    other.restore(online.checkpoint())
    result = other.step(
        dict(hydraulic_nodes=dict(a=dict(pressure_pa=198000.0))), 2.5, expected_revision=1
    )
    print(
        json.dumps(
            dict(
                scheduled_time_s=final["time_s"],
                scheduled_external_energy_j=final["engine"]["state"][
                    "integrated_external_energy_j"
                ],
                online_time_s=result["observation"]["model"]["time_s"],
                revision=other.revision,
                reverse_flow_kg_s=result["observation"]["model"]["engine"]["state"]["flows_kg_s"][
                    "pipe"
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
