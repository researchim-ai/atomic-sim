"""Domain rejection is distinct from modifying accepted conserved stocks."""

import numpy as np
import pytest

from core.status import DomainError
from scripts.extended_natural_initialization import damped_newton


def test_damped_newton_rejects_invalid_trials_without_projection():
    rejected = []

    def residual(state):
        x = state[0]
        if not 0 < x < 5:
            rejected.append(x)
            raise DomainError("outside charging domain")
        return np.array([(x * x - 4) * 1000])

    initial = np.array([0.1])
    sol = damped_newton(residual, initial)
    assert rejected
    assert initial[0] == 0.1
    assert sol.x[0] == pytest.approx(2.0, abs=1e-5)


def test_scenario_admission_waits_for_full_history_replay(tmp_path, monkeypatch):
    from atomic_model_plant_extension.qualification import CASES, run_case
    from scripts.extended_plant_release import scenarios

    monkeypatch.setitem(CASES, "nominal", dict(duration_s=0.1, changes=[]))
    root = tmp_path / "unit"
    run_case("nominal", output=root, output_step_s=0.1, max_step_s=0.1)
    result = scenarios([root])
    assert not result["admitted"]
    assert len(result["candidates_awaiting_full_history_replay"]) == 1
