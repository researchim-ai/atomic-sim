"""Explicit DP -> reduced Radau -> DP state handoff; fixed commands only."""

import json

from scripts.nonlinear_graph_references import pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph
from scripts.reduced_residual_engine import ReducedResidualEngine
from scripts.registered_graph_system import graph_request


def main():
    spec = pipe_spec()
    first = NonlinearThermalGraph(spec)
    first.advance(7.5)
    reduced = ReducedResidualEngine(graph_request(spec))
    reduced.import_legacy_checkpoint(first.checkpoint())
    reduced.advance(7.5)
    handoff = reduced.export_legacy_checkpoint()
    final = NonlinearThermalGraph(spec)
    final.restore(handoff["checkpoint"])
    final.advance(5.0)
    print(
        json.dumps(
            dict(
                time_s=final.snapshot()["time_s"],
                temperatures_k=final.snapshot()["temperatures_k"],
                energy_residual_j=final.snapshot()["energy_residual_j"],
                native_dp_proposal_reset=handoff["native_dp_proposal_reset"],
            )
        )
    )


if __name__ == "__main__":
    main()
