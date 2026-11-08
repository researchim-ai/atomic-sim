"""Finite donor/receiver HEM release: phase appearance, balances, mesh/time and replay.

Closed adiabatic graph; not a nozzle, atmospheric break or critical-flow law.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.cases import baseline
from atomic_model_hem_qualified.model import QualifiedHEMNetwork, signature
from atomic_model_hem_qualified.qualification import CRITERIA, errors
from atomic_model_hem_qualified.reference import solve
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json

MESHES = (4, 8, 16, 32)
CAPS = (2e-5, 1e-5, 5e-6)
CALLS = [dict(duration_s=t, command=None) for t in (0.0002, 0.0003, 0.0005)]


def release_request(cells=4, cap=1e-5):
    if cells not in MESHES or cap not in CAPS:
        raise ValueError("Declared release mesh/cap only")
    request = baseline(0.0, velocity=0.0, cells=cells)
    spec = request["model"]
    spec["max_step_s"] = cap
    # Retain an inert wall to satisfy the graph's wall contract; no heat exchange.
    spec["walls"][0]["conductance_w_k"] = 0.0
    spec["nodes"][1]["initial"].update(pressure_pa=0.95e6, quality=0.1)
    donor_entropy = eos.px(1e6, 0.0).entropy_j_kg_k
    for edge in spec["edges"]:
        for index, point in enumerate(edge["initial"]):
            # Four constant physical blocks: every refinement has identical stocks.
            block = index // (cells // 4)
            pressure = 1e6 - (block + 0.5) / 4 * 5e4
            liquid, vapour = eos.saturation(pressure)
            quality = (donor_entropy - liquid.entropy_j_kg_k) / (vapour.entropy_j_kg_k - liquid.entropy_j_kg_k)
            point.update(pressure_pa=pressure, quality=quality)
    return request


def inventories(model):
    mass, momentum, relative_energy, *_ = model.primitive(model._y)
    return dict(mass_kg=float(sum(mass)), energy_j=float(sum(relative_energy + mass * model.eref)))


def restricted_pressure(model):
    mass, momentum, relative_energy, *_ = model.primitive(model._y)
    energy = relative_energy + mass * model.eref
    pressure = []
    for indices in model.edge_indices:
        for group in np.array_split(indices, 4):
            m, p, e = float(sum(mass[group])), float(sum(momentum[group])), float(sum(energy[group]))
            state = eos.rho_u(m / float(sum(model.volumes[group])), e / m - 0.5 * (p / m) ** 2,
                              pressure_seed_pa=1e6, temperature_seed_k=453.0)
            pressure.append(state.pressure_pa)
    return np.array(pressure)


def physical_checks(initial, final):
    donor_before, donor_after = initial["cells"][0], final["cells"][0]
    receiver_before, receiver_after = initial["cells"][1], final["cells"][1]
    ledger = final["global_ledger"]
    return dict(
        donor_loses_mass=donor_after["mass_kg"] < donor_before["mass_kg"],
        receiver_gains_mass=receiver_after["mass_kg"] > receiver_before["mass_kg"],
        donor_depressurizes=donor_after["pressure_pa"] < donor_before["pressure_pa"],
        donor_flashes=donor_before["quality"] == 0 and donor_after["phase"] == "two_phase" and donor_after["quality"] > 0,
        positive_mass=min(cell["mass_kg"] for cell in final["cells"]) > 0,
        mass_balance=abs(ledger["mass_residual_kg"]) <= CRITERIA["mass_balance_kg"],
        energy_balance=abs(ledger["energy_residual_j"]) <= CRITERIA["energy_balance_j"],
        entropy_lower=ledger["numerical_entropy_residual_j_k"] >= CRITERIA["numerical_entropy_lower_j_k"],
        adiabatic_closed=all(ledger[key] == 0 for key in ("external_fluid_heat_j", "external_wall_heat_j", "external_entropy_j_k")),
    )


def verify(output):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output cannot be a symlink")
    output.mkdir(parents=True, exist_ok=False)
    source, driver = signature(), sha(__file__)
    declared = dict(source=source, driver_sha256=driver, meshes=MESHES, caps_s=CAPS,
                    initial_profile="Four isentropic p/x pipe blocks from saturated-liquid donor; finite mixed receiver",
                    program=CALLS, criteria=CRITERIA, requests={str(n): release_request(n) for n in MESHES},
                    spatial_reference_cells=32, temporal_budget_fraction=0.1,
                    validated=False, plant_referenced=False)
    atomic_json(output / "protocol.json", seal(declared))
    runs, profiles, stocks = [], {}, []
    reference_request = release_request()
    references = {method: solve(reference_request, CALLS, method, max_step=2e-5) for method in ("DOP853", "Radau")}
    atomic_json(output / "references.json", seal(references))
    agreement = {key: max(errors(a, b)[key] for a, b in zip(references["DOP853"]["observations"], references["Radau"]["observations"], strict=True))
                 for key in ("pressure_error_pa", "temperature_error_k", "velocity_error_m_s")}
    for cells in MESHES:
        profiles[cells] = {}
        for cap in CAPS:
            # The coarsest cap is a temporal diagnostic on the 4-cell mesh only.
            if cells != 4 and cap == CAPS[0]:
                continue
            request = release_request(cells, cap)
            model = QualifiedHEMNetwork(request)
            stocks.append(inventories(model))
            observations = [model.snapshot()]
            for call in CALLS:
                observations.append(model.advance(call["duration_s"]))
            initial, final = observations[0], observations[-1]
            checks = physical_checks(initial, final)
            restart = None
            if cells == 4 and cap == CAPS[-1]:
                restarted = QualifiedHEMNetwork(request)
                restarted.advance(0.0002)
                restarted.advance(0.0003)
                saved = restarted.checkpoint()
                restored = QualifiedHEMNetwork(request)
                restored.restore(saved)
                resumed = restored.advance(0.0005)
                restart = restored.checkpoint() == model.checkpoint() and resumed == final
                checks["phase_restart_equal"] = restart
                atomic_json(output / "phase-checkpoint.json", saved)
            profiles[cells][cap] = restricted_pressure(model)
            record = dict(cells=cells, cap_s=cap, checks={k: bool(v) for k, v in checks.items()},
                          passed=all(checks.values()), final_ledger=final["global_ledger"])
            if cells == 4:
                record["reference_errors"] = {key: max(errors(a, b)[key] for a, b in zip(observations, references["DOP853"]["observations"], strict=True)) for key in agreement}
                record["passed"] = record["passed"] and all(v <= CRITERIA[k] for k, v in record["reference_errors"].items())
            atomic_json(output / f"mesh-{cells}-cap-{cap:g}.json", seal(dict(request=request, observations=observations, record=record)))
            runs.append(record)
            print(cells, cap, record["passed"], flush=True)
    pressure_errors = {n: float(np.mean(abs(profiles[n][CAPS[-1]] - profiles[32][CAPS[-1]]))) for n in MESHES[:-1]}
    temporal = {n: float(np.mean(abs(profiles[n][CAPS[-2]] - profiles[n][CAPS[-1]]))) for n in MESHES}
    initial_stocks_equal = all(np.allclose(list(s.values()), list(stocks[0].values()), rtol=2e-12, atol=0) for s in stocks)
    checks = dict(
        native_runs=all(r["passed"] for r in runs),
        independent_integrators_agree=all(v <= CRITERIA[k] for k, v in agreement.items()),
        identical_initial_inventories=initial_stocks_equal,
        decreasing_mesh_error=pressure_errors[4] > pressure_errors[8] > pressure_errors[16] > 0,
        temporal_error_separated=all(temporal[n] <= max(1e-4, 0.1 * pressure_errors[min(n, 16)]) for n in MESHES),
        source_unchanged=signature() == source and sha(__file__) == driver,
    )
    report = dict(schema_version=1, passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()}, runs=runs,
                  reference_agreement=agreement, mesh_pressure_errors_pa={str(n): v for n, v in pressure_errors.items()},
                  temporal_pressure_differences_pa={str(n): v for n, v in temporal.items()},
                  scope="0.001s finite-receiver graph release with equilibrium flashing; finite 32-cell reference, shared EOS, no choking/nozzle/atmospheric-break validation",
                  validated=False, plant_referenced=False)
    atomic_json(output / "report.json", seal(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = verify(parser.parse_args(argv).output)
    print(json.dumps(dict(passed=report["passed"])))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
