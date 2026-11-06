"""Bounded head feedback, conservation and complete controlled-session replay."""

import json

import numpy as np
import pytest

from atomic_model_controlled_plant import Session
from atomic_model_controlled_plant.qualification import CASES, run_case
from atomic_model_workspace.common import digest


def test_head_feedback_tracks_rate_without_projecting_flow():
    s = Session()
    p = s.plant
    y = p.pack().copy()
    p.feed_bias_kg_s = 7.0
    rate = p.derivatives(0.0, y)
    assert not p.feed_controller["saturated"]
    assert rate[p.feed_index] == pytest.approx(7.0 / 0.2, abs=1e-7)
    assert p.network.junctions["feed"].mass_flow_kg_s == y[p.feed_index]
    assert abs(sum(p.account_rates_w(rate).values()) - sum(p.boundary_terms_w().values())) < 0.01


def test_head_saturation_changes_actuator_only():
    s = Session()
    p = s.plant
    before = p.pack().copy()
    p.feed_bias_kg_s = 1e6
    p.refresh_algebraic(before)
    assert p.feed_controller["saturated"]
    assert p.network.junctions["feed"].pump_pressure_pa == 12e6
    np.testing.assert_array_equal(p.pack(), before)


def test_complete_source_checkpoint_and_priority():
    s = Session()
    assert not any(
        note.startswith("узлы на одной отметке:") or note.startswith("питание — кривая напора")
        for note in s.plant.passport.approximations
    )
    result = s.advance(
        0.2, commands=dict(scram=True, motor_off=True, throttle=1.0, rod_command=-0.1)
    )
    json.dumps(result, allow_nan=False)
    assert s.plant.effective_steam_throttle == 0.0 and not s.plant.motor_on
    saved = json.loads(json.dumps(s.checkpoint()))
    r = Session.restore(saved)
    s.advance(0.2)
    r.advance(0.2)
    assert s.checkpoint() == r.checkpoint()
    bad = json.loads(json.dumps(saved))
    bad["payload"]["config"]["implementation_sha256"] = "0" * 64
    bad["sha256"] = digest(bad["payload"])
    with pytest.raises(ValueError, match="source/runtime"):
        Session.restore(bad)


def test_failed_step_restores_controller_and_full_replay(monkeypatch):
    s = Session()
    s.advance(0.1)
    before = s.checkpoint()
    controller = dict(s.plant.feed_controller)
    original = s.plant.advance

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected after physical acceptance")

    monkeypatch.setattr(s.plant, "advance", fail)
    with pytest.raises(RuntimeError):
        s.advance(0.2, commands={"scram": True})
    assert s.checkpoint() == before and s.plant.feed_controller == controller
    monkeypatch.setattr(s.plant, "advance", original)
    s.advance(0.2)
    assert Session.restore(s.checkpoint()).checkpoint() == s.checkpoint()


def test_controlled_campaign_json_and_settling_fields(tmp_path, monkeypatch):
    monkeypatch.setitem(CASES, "load_rejection", dict(duration_s=0.2, changes=[]))
    r = run_case("load_rejection", output=tmp_path / "unit", max_step_s=0.1, output_step_s=0.1)
    assert r["full_horizon_accepted"] is True
    assert r["load_rejection_settling"]["final_normalized_rate_per_s"] < 1e-5
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        {"maximum_internal_step_s": float("nan")},
        {"maximum_internal_step_s": True},
        {"initialization": "unknown"},
    ],
)
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        Session(**options)
