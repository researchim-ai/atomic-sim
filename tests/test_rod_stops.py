"""Упоры проверяются по аналитическому времени хода, в том числе внутри шага."""

import numpy as np
import pytest

from core.status import SolverError
from sdk.api import create_session, export_snapshot


@pytest.mark.parametrize("method", ["BDF", "Radau"])
@pytest.mark.parametrize("position,speed,target", [(0.9, 0.4, 1.0), (0.1, -0.4, 0.0)])
def test_stop_time_dense_state_and_internal_mesh(method, position, speed, target):
    plant = create_session()
    plant.rod, plant.rod_command = position, speed
    plant.enable_continuation(1.0)

    def rhs(t, y):
        dy = np.zeros_like(y)
        dy[0] = -10 * y[0]
        dy[12] = plant._rod_rate_override
        return dy

    plant.derivatives = rhs
    result = plant._integrate_interval(plant.pack(), 0.0, 0.5, 1e-10, 1e-12, 0.1, None, method)
    hit = (target - position) / speed
    assert result.events[0]["t"] == pytest.approx(hit, abs=1e-15)
    assert result.final[12] == target
    times = np.linspace(0.0, 0.5, 51)
    values = result.dense(times)
    assert values[12] == pytest.approx(np.clip(position + speed * times, 0.0, 1.0), abs=1e-15)
    assert values[0] == pytest.approx(np.exp(-10 * times), abs=1e-7)
    assert np.array_equal(result.dense.ts, result.t)
    assert len(result.t) > 3
    assert plant._rod_rate_override is None


def test_stop_exactly_at_output_boundary_is_exact_and_is_not_repeated():
    plant = create_session()
    plant.rod = 0.98
    plant.rod_command = 0.2
    plant.advance(0.1, method="Radau", jacobian_relative_step=1e-5)
    assert plant.rod == 1.0
    assert len(plant.last_motion_events) == 1
    plant.advance(0.1, method="Radau", jacobian_relative_step=1e-5)
    assert plant.rod == 1.0
    assert plant.last_motion_events == ()


def test_failure_after_stop_restores_whole_external_step():
    plant = create_session()
    plant.rod, plant.rod_command = 0.99, 0.2
    plant.enable_continuation(1.0)
    before = export_snapshot(plant)

    def rhs(t, y):
        if t > 0.06:
            raise SolverError("after rod stop")
        dot = np.zeros_like(y)
        dot[12] = plant._rod_rate_override
        return dot

    plant.derivatives = rhs
    with pytest.raises(SolverError, match="after rod stop"):
        plant.advance(0.1, method="Radau")
    assert export_snapshot(plant) == before
    assert plant._continuation is None
    assert plant._rod_rate_override is None


def test_full_scram_respects_exact_stop_and_dense_energy_accounting():
    plant = create_session()
    plant.scram_latched = True
    plant.advance(4.0, method="Radau", jacobian_relative_step=1e-5)
    assert plant.rod == 1.0
    assert plant.last_motion_events == ({"kind": "rod_stop", "t": 3.75, "position": 1.0},)
    assert plant.last_energy_ledger["interval_s"] == 4.0


def test_restore_rejects_out_of_range_rod_atomically():
    from core.status import DomainError
    from sdk.api import restore_snapshot

    plant = create_session()
    before = export_snapshot(plant)
    invalid = export_snapshot(plant)
    invalid["state"][12] = 1.0001
    with pytest.raises(DomainError, match="стерж"):
        restore_snapshot(plant, invalid)
    assert export_snapshot(plant) == before
