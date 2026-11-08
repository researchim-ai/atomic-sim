"""Release benchmark geometry, inventories and admission gates."""

from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from scripts.verify_phase_release import MESHES, inventories, physical_checks, release_request


def test_nested_meshes_preserve_physical_inventory_and_isentropic_blocks():
    stocks = []
    entropy = eos.px(1e6, 0).entropy_j_kg_k
    for n in MESHES:
        request = release_request(n)
        model = QualifiedHEMNetwork(request)
        stocks.append(list(inventories(model).values()))
        assert request["model"]["walls"][0]["conductance_w_k"] == 0
        for edge in request["model"]["edges"]:
            assert edge["length_m"] == 2 and edge["diameter_m"] == 0.05
            for point in edge["initial"]:
                state = eos.px(point["pressure_pa"], point["quality"])
                assert state.entropy_j_kg_k == pytest.approx(entropy, abs=1e-8)
                assert 0 < point["quality"] < 0.01
    np.testing.assert_allclose(stocks, np.tile(stocks[0], (len(stocks), 1)), rtol=2e-12)


@pytest.mark.parametrize("n,cap", [(3, 1e-5), (64, 1e-5), (4, 0), (4, float("nan"))])
def test_undeclared_refinement_refused(n, cap):
    with pytest.raises(ValueError): release_request(n, cap)


def synthetic_reports():
    initial = dict(cells=[dict(mass_kg=10., pressure_pa=1e6, quality=0.), dict(mass_kg=1.)])
    final = dict(cells=[dict(mass_kg=9., pressure_pa=.99e6, quality=.001, phase="two_phase"), dict(mass_kg=2.)],
                 global_ledger=dict(mass_residual_kg=0., energy_residual_j=0., numerical_entropy_residual_j_k=0.,
                                    external_fluid_heat_j=0., external_wall_heat_j=0., external_entropy_j_k=0.))
    return initial, final


def test_all_physical_predicates_required():
    initial, final = synthetic_reports()
    assert all(physical_checks(initial, final).values())
    for cell, key, value in [(0, "mass_kg", 11.), (1, "mass_kg", .9), (0, "pressure_pa", 1.01e6), (0, "quality", 0.)]:
        changed = deepcopy(final)
        changed["cells"][cell][key] = value
        assert not all(physical_checks(initial, changed).values())
    for key, value in [("mass_residual_kg", 1.), ("energy_residual_j", 1.), ("numerical_entropy_residual_j_k", -1.), ("external_fluid_heat_j", 1.)]:
        changed = deepcopy(final)
        changed["global_ledger"][key] = value
        assert not all(physical_checks(initial, changed).values())
