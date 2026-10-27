"""Сборка невязки index-1 и таймер защиты внутри шага вывода."""

import math

import numpy as np
import pytest

from control.regulation import ProtectionLogic, high_trip_time
from core.events import EventSpec, resolve_events
from core.residual import DuplicateConstraint, IndexOneDrive, ResidualSystem, scale_residual
from core.status import DomainError


def test_index_one_start_is_consistent_and_degenerate_scheme_is_rejected():
    drive = IndexOneDrive(1.0)
    system = ResidualSystem([drive])
    state = np.array([0.2, 1.0])
    rate = np.array([1.0 - 0.2, 0.0])
    system.require_consistent(0.0, state, rate)
    time = 0.4
    amplitude = 1.0 + (0.2 - 1.0) * math.exp(-time)
    trajectory = np.array([amplitude, 1.0])
    trajectory_rate = np.array([1.0 - amplitude, 0.0])
    system.require_consistent(time, trajectory, trajectory_rate)
    jacobian = system.numerical_jacobian(0.0, state, rate)
    system.require_full_rank(jacobian)
    broken = ResidualSystem([DuplicateConstraint()])
    with pytest.raises(DomainError):
        broken.require_full_rank(broken.numerical_jacobian(0.0, np.array([1.0, 0.0]), np.zeros(2)))


def test_events_honor_time_then_priority_and_hysteresis():
    early = EventSpec("early", lambda _t, y: y[0] - 0.2, direction=1.0, priority=0)
    late = EventSpec("late", lambda _t, y: y[0] - 0.8, direction=1.0, priority=10)
    ordered = resolve_events([late, early], 0.0, np.array([0.0]), 1.0, np.array([1.0]))
    assert [item.name for item in ordered] == ["early", "late"]
    low = EventSpec("low", lambda _t, y: y[0] - 0.5, direction=1.0, priority=1)
    high = EventSpec("high", lambda _t, y: y[0] - 0.5, direction=1.0, priority=5)
    tied = resolve_events([low, high], 0.0, np.array([0.0]), 1.0, np.array([1.0]))
    assert [item.name for item in tied] == ["high", "low"]
    trip = EventSpec("trip", lambda _t, y: y[0] - 1.0, direction=1.0, hysteresis=0.2)
    assert len(resolve_events([trip], 0.0, np.array([0.5]), 1.0, np.array([1.5]))) == 1
    assert trip.armed is False
    assert resolve_events([trip], 1.0, np.array([1.5]), 2.0, np.array([0.9])) == []
    assert trip.armed is False
    assert resolve_events([trip], 2.0, np.array([0.9]), 3.0, np.array([0.5])) == []
    assert trip.armed is True
    assert scale_residual(np.array([1.0e6, 2.0]), np.array([1.0e6, 2.0])) == pytest.approx([1.0, 1.0])


def test_protection_timer_does_not_follow_the_output_step():
    def power(time: float) -> float:
        return 1.0 + 0.5 * time

    for output_step in (0.5, 0.25, 0.125):
        logic = ProtectionLogic(setpoint=1.2, delay_s=0.3)
        tripped = high_trip_time(logic, power, 0.0, 2.0, output_step)
        assert tripped == pytest.approx(0.7, abs=1.0e-9)
        assert logic.latched
