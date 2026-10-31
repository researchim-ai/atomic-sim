"""The extended grid must preserve sparse conservation and legacy admission."""

import numpy as np
import pytest

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel


@pytest.mark.parametrize("flux,cells", [("rusanov", 65), ("hllc", 513), ("hllc", True)])
def test_refinement_does_not_expand_legacy_or_unbounded_grid(flux, cells):
    with pytest.raises(ValueError):
        EulerConfig(numerical_flux=flux, cells=cells)


@pytest.mark.parametrize("cells", [128, 512])
def test_refined_contact_preserves_sparse_inventory_identities_and_restart(cells):
    model = EulerLiquidChannel(
        EulerConfig(
            cells=cells,
            numerical_flux="hllc",
            boundary="periodic",
            initial_velocity_m_s=1.0,
            total_loss_coefficient=0,
            total_ua_w_k=0,
        )
    )
    jac = model.analytic_jacobian(0, model._state)
    n = model.config.cells
    assert jac.format == "csc" and jac.nnz < 100 * n
    assert jac.shape == (4 * n + 3, 4 * n + 3)
    for inventory, ledger in ((0, -3), (1, -2), (2, -1)):
        weights = np.zeros(4 * n + 3)
        weights[inventory : 3 * n : 3] = 1
        if inventory == 2:
            weights[3 * n : 4 * n] = model.config.total_heat_capacity_j_k / n
        assert jac.T @ weights - jac[ledger, :].toarray().ravel() == pytest.approx(
            np.zeros(len(weights)), abs=1e-5
        )
    model.advance(0.001, method="Radau", rtol=1e-9)
    snapshot = model.snapshot()
    restored = EulerLiquidChannel.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    assert abs(restored.observation()["energy_residual_j"]) < 1e-3
    assert restored.observation()["right_port_x_m"] == model.config.total_length_m
