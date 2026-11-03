"""Natural startup of a closed heated/cooled Boussinesq loop; exact joint restart."""

import json

from scripts.buoyant_graph_references import loop
from scripts.buoyant_inertial_graph import BuoyantInertialGraph


def main():
    request = loop()
    a = BuoyantInertialGraph(request, max_step_s=0.1)
    assert a.snapshot()["state"]["flows_kg_s"]["e0"] == 0
    a.advance(20)
    b = BuoyantInertialGraph(request, max_step_s=0.1)
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert a.advance(20) == b.advance(20)
    state = a.snapshot()["state"]
    print(
        dict(
            time_s=state["time_s"],
            circulation_kg_s=state["flows_kg_s"]["e0"],
            temperatures_k=state["temperatures_k"],
            buoyancy_work_j=state["buoyancy_work_j"],
            total_energy_residual_j=state["total_energy_residual_j"],
            exact_local_restart=True,
            full_eos=False,
        )
    )


if __name__ == "__main__":
    main()
