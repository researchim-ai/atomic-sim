"""Two-way thermal/inertial feedback, friction heating and exact joint restart."""

import json

from scripts.thermal_inertial_graph import ThermalInertialGraph
from scripts.thermal_inertial_references import cases


def main():
    request = next(r for name, r, _ in cases() if name == "boundary_reversal")
    a = ThermalInertialGraph(request, max_step_s=0.05)
    a.advance(10)
    a.step(
        dict(a=198000.0, b=200000.0),
        10,
        expected_revision=a.revision,
        command_id="operator:reverse-1",
    )
    b = ThermalInertialGraph(request, max_step_s=0.05)
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert a.advance(5) == b.advance(5)
    state = a.snapshot()["state"]
    print(
        dict(
            time_s=state["time_s"],
            flows_kg_s=state["flows_kg_s"],
            temperatures_k=state["temperatures_k"],
            friction_to_fluid_j=state["friction_to_fluid_j"],
            total_energy_residual_j=state["total_energy_residual_j"],
            exact_local_restart=True,
        )
    )


if __name__ == "__main__":
    main()
