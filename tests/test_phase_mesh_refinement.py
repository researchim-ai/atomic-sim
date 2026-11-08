"""Fixed-geometry phase-change mesh study; finite fine grid, not continuum truth."""

import json
from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.cases import matrix
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_qualified.qualification import CRITERIA

pytestmark = pytest.mark.long_running


def request_on_mesh(name, cells, cap):
    if cells not in (4, 8, 16, 32):
        raise ValueError("Declared nested mesh only")
    request = deepcopy(dict(matrix())[name])
    model = request["model"]
    model["max_step_s"] = cap
    wall = model["walls"][0]
    for edge in model["edges"]:
        edge["initial"] = [deepcopy(edge["initial"][0]) for _ in range(cells)]
        edge["cells"] = cells
        edge["heat_power_w"] = [0.0] * cells
    # The original wall covers the first half of edge e0. Keep its physical
    # support, heat capacity and total conductance identical on every mesh.
    model["walls"] = []
    for index in range(cells // 2):
        piece = deepcopy(wall)
        piece.update(id=f"w{index}", cell=f"e0/{index}")
        for key in ("capacity_j_k", "conductance_w_k", "heat_power_w", "ambient_ua_w_k"):
            piece[key] /= cells / 2
        model["walls"].append(piece)
    return request


def restricted_pressure(model):
    mass, momentum, relative_energy, *_ = model.primitive(model._y)
    energy = relative_energy + mass * model.eref
    values = []
    for indices in model.edge_indices:
        for group in np.array_split(indices, 4):
            m = float(sum(mass[group]))
            p = float(sum(momentum[group]))
            e = float(sum(energy[group]))
            state = eos.rho_u(
                m / float(sum(model.volumes[group])),
                e / m - 0.5 * (p / m) ** 2,
                pressure_seed_pa=1e6,
                temperature_seed_k=453.0,
            )
            values.append(state.pressure_pa)
    return np.array(values)


@pytest.mark.parametrize(
    "name,phase",
    [
        ("vapor_phase_exit", "vapor"),
        ("condensation_entry", "two_phase"),
        ("liquid_phase_exit", "liquid"),
    ],
)
def test_phase_mesh_and_time_errors_are_separated(name, phase, record_testsuite_property):
    profiles = {}
    temporal = {}
    inventories = []
    baseline = dict(matrix())[name]["model"]
    for cells in (4, 8, 16, 32):
        results = []
        for cap in (1e-5, 5e-6):
            request = request_on_mesh(name, cells, cap)
            for key in ("capacity_j_k", "conductance_w_k", "heat_power_w", "ambient_ua_w_k"):
                assert sum(w[key] for w in request["model"]["walls"]) == pytest.approx(
                    baseline["walls"][0][key]
                )
            model = QualifiedHEMNetwork(request)
            mass, _, relative_energy, *_ = model.primitive(model._y)
            inventories.append((float(sum(mass)), float(sum(relative_energy + mass * model.eref))))
            initial_phases = [c["phase"] for c in model.snapshot()["cells"]]
            observed = model.advance(0.001)
            assert any(c["phase"] == phase for c in observed["cells"])
            assert initial_phases != [c["phase"] for c in observed["cells"]]
            assert min(c["mass_kg"] for c in observed["cells"]) > 0
            assert abs(observed["global_ledger"]["mass_residual_kg"]) <= CRITERIA["mass_balance_kg"]
            assert (
                abs(observed["global_ledger"]["energy_residual_j"]) <= CRITERIA["energy_balance_j"]
            )
            results.append(restricted_pressure(model))
        profiles[cells] = results[-1]
        temporal[cells] = float(np.mean(abs(results[0] - results[1])))
    np.testing.assert_allclose(
        inventories, np.tile(inventories[0], (len(inventories), 1)), rtol=2e-12, atol=0
    )
    errors = {n: float(np.mean(abs(profiles[n] - profiles[32]))) for n in (4, 8, 16)}
    record_testsuite_property(
        name,
        json.dumps(
            dict(
                mesh_pressure_errors_pa=errors,
                temporal_differences_pa=temporal,
                reference_cells_per_edge=32,
                duration_s=0.001,
            )
        ),
    )
    assert errors[4] > errors[8] > errors[16] > 0, (name, errors)
    for n in (4, 8, 16):
        assert temporal[n] <= max(1e-4, 0.1 * errors[n]), (name, n, errors, temporal)

    assert temporal[32] <= max(1e-4, 0.1 * errors[16]), (name, errors, temporal)
