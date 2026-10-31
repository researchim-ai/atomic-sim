"""Explicit cell-inventory tolerance policy must be transactional and reproducible."""

import json

import pytest

from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
from sdk.euler_experiment import run_euler_experiment, validate_euler_experiment


def specification():
    return dict(
        schema_version=1,
        engine="contact-euler-liquid-channel",
        name="absolute_tolerance",
        duration_s=0.002,
        step_s=0.001,
        channel=EulerConfig(cells=3, numerical_flux="hllc", boundary="periodic").as_dict(),
        solver=dict(atol_scale=0.1),
    )


@pytest.mark.parametrize("scale", [0, 1.1, 1e-7, float("nan"), True])
def test_invalid_absolute_scale_rejected_before_state_or_clock_changes(scale):
    model = EulerLiquidChannel(EulerConfig(numerical_flux="hllc"))
    before = model.snapshot()
    with pytest.raises(ValueError):
        model.advance(0.001, atol_scale=scale)
    assert model.snapshot() == before
    spec = specification()
    spec["solver"]["atol_scale"] = scale
    with pytest.raises(ValueError):
        validate_euler_experiment(spec)


def test_absolute_tolerance_is_serialized_by_real_sdk_and_default_stays_compatible(tmp_path):
    spec = specification()
    report = run_euler_experiment(spec, tmp_path / "run")
    assert report["numerical_acceptance"]
    saved = json.loads((tmp_path / "run/input.json").read_text())
    assert saved["solver"]["atol_scale"] == 0.1
    snapshot = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert EulerLiquidChannel.from_snapshot(snapshot).snapshot() == snapshot
    del spec["solver"]["atol_scale"]
    assert "atol_scale" not in validate_euler_experiment(spec)["solver"]


def test_changed_absolute_policy_restarts_continuation():
    model = EulerLiquidChannel(EulerConfig(numerical_flux="hllc"))
    model.enable_continuation(0.01)
    model.advance(0.001)
    before = model._continuation
    model.advance(0.001, atol_scale=0.1)
    assert model._continuation is not before
    assert model.time_s == 0.002
    assert abs(model.observation()["energy_residual_j"]) < 1e-3
