"""Естественная циркуляция вне паспорта ВВЭР: напор против трения."""

import pytest

from core.status import DomainError
from physics.hydraulics.circulation import loop_residuals, natural_circulation_loop


def test_buoyancy_balances_friction_at_nonzero_flow():
    loop = natural_circulation_loop()
    assert loop.junctions["bottom"].mass_flow_kg_s > 0.0
    residual = loop_residuals(loop)
    assert residual["mass"] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1.0e-6)
    assert residual["energy"] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1.0)
    assert residual["flow_acceleration"] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1.0e-4)
    with pytest.raises(DomainError):
        natural_circulation_loop(hot_temperature_k=450.0, cold_temperature_k=450.0)
