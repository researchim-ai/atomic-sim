"""Pressure-driven inertial network: reversal, mechanical ledger and exact restart."""

import json

from scripts.inertial_graph import InertialGraph
from scripts.inertial_graph_references import cases


def main():
    spec = next(spec for name, spec, _ in cases() if name == "boundary_reversal")
    a = InertialGraph(spec, max_step_s=0.05)
    a.advance(10)
    reversal = a.step(
        dict(a=198000.0, b=200000.0),
        10,
        expected_revision=a.revision,
        command_id="operator:reverse-1",
    )
    transition = reversal["boundary_transition"]
    assert transition["left"]["kinetic_energy_j"] == transition["right"]["kinetic_energy_j"]
    b = InertialGraph(spec, max_step_s=0.05)
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert a.advance(10) == b.advance(10)
    state = a.snapshot()["state"]
    print(
        dict(
            time_s=state["time_s"],
            flows_kg_s=state["flows_kg_s"],
            kinetic_energy_j=state["kinetic_energy_j"],
            exported_heat_j=state["exported_heat_j"],
            energy_residual_j=state["energy_residual_j"],
            exact_local_restart=True,
        )
    )


if __name__ == "__main__":
    main()
