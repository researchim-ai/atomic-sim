"""Dynamic mass/EOS pressure, heater command and exact thermodynamic restart."""

import json

from scripts.compressible_storage_graph import CompressibleStorageGraph
from scripts.storage_graph_references import cases


def main():
    request = next(r for name, r, _ in cases() if name == "mass_energy_equalization")
    a = CompressibleStorageGraph(request, max_step_s=0.05)
    a.advance(5)
    a.step(dict(a=100.0, b=0.0), 5, expected_revision=a.revision, command_id="operator:heat-1")
    b = CompressibleStorageGraph(request, max_step_s=0.05)
    b.restore(json.loads(json.dumps(a.checkpoint())))
    assert a.advance(5) == b.advance(5)
    state = a.snapshot()["state"]
    print(
        dict(
            time_s=state["time_s"],
            masses_kg=state["masses_kg"],
            densities_kg_m3=state["densities_kg_m3"],
            pressures_pa=state["pressures_pa"],
            energy_residual_j=float(state["energy_residual_j"]),
            generated_entropy_j_k=state["generated_entropy_j_k"],
            entropy_residual_j_k=float(state["entropy_residual_j_k"]),
            exact_local_restart=True,
        )
    )


if __name__ == "__main__":
    main()
