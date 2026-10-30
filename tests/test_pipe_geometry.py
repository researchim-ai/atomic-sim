"""Mesh refinement preserves a pipe, legacy snapshots preserve their format."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
from sdk.dynamic_experiment import run_dynamic_experiment, validate_dynamic_experiment


def pipe(n=4, heat=3000.0):
    return DynamicConfig.from_pipe(
        cells=n,
        total_length_m=3.0,
        total_loss_coefficient=30.0,
        left_pressure_pa=7.02e6,
        right_pressure_pa=7e6,
        left_temperature_k=500.0,
        right_temperature_k=500.0,
        initial_pressure_pa=7.01e6,
        initial_temperature_k=500.0,
        area_m2=0.001,
        total_heat_capacity_j_k=8000.0,
        total_ua_w_k=1000.0,
        heat_w=heat,
    )


@pytest.mark.parametrize("n", [1, 4, 8, 16, 32])
def test_refinement_preserves_length_loss_volume_and_half_boundaries(n):
    c = pipe(n)
    assert sum(c.junction_lengths_m) == pytest.approx(3.0)
    assert sum(c.junction_loss_coefficients) == pytest.approx(30.0)
    assert c.total_volume_m3 == 0.003
    assert c.junction_lengths_m[0] == pytest.approx(1.5 / n)
    assert c.junction_lengths_m[-1] == c.junction_lengths_m[0]
    model = DynamicWallChannel(c)
    assert sum(j.length_m for j in model.network.junctions.values()) == pytest.approx(3.0)
    assert sum(j.loss_coefficient for j in model.network.junctions.values()) == pytest.approx(30.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("junction_lengths_m", [1.0]),
        ("junction_lengths_m", [1.0, 1.0, 0.0, 1.0, 1.0]),
        ("junction_loss_coefficients", [1.0, 1.0, -1.0, 1.0, 1.0]),
        ("junction_loss_coefficients", [True] * 5),
        ("junction_lengths_m", [float("nan")] * 5),
    ],
)
def test_bad_connection_vectors_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(pipe(), **{field: value})


def test_pipe_derived_geometry_cannot_be_silently_overridden():
    with pytest.raises(ValueError, match="override"):
        DynamicConfig.from_pipe(
            cells=4,
            total_length_m=3.0,
            total_loss_coefficient=30.0,
            area_m2=0.001,
            total_volume_m3=1.0,
        )


def test_explicit_geometry_bundle_and_snapshot_roundtrip(tmp_path):
    spec = dict(
        schema_version=1,
        name="pipe",
        engine="pressure-driven-wall-channel",
        duration_s=0.01,
        step_s=0.005,
        hydraulics=pipe(2).as_dict(),
    )
    folder = tmp_path / "run"
    report = run_dynamic_experiment(spec, folder)
    assert report["numerical_acceptance"]
    normalized = json.loads((folder / "input.json").read_text())
    assert validate_dynamic_experiment(normalized) == normalized
    payload = json.loads((folder / "checkpoint.json").read_text())
    model = DynamicWallChannel.from_snapshot(payload)
    assert model.snapshot() == payload
    assert model.config.junction_lengths_m == (0.75, 1.5, 0.75)


def test_legacy_checkpoint_keeps_exact_dictionary_and_uniform_connections():
    payload = json.loads(Path("tests/fixtures/pressure_channel_v1.json").read_text())
    model = DynamicWallChannel.from_snapshot(payload)
    assert model.snapshot() == payload
    assert model.config.junction_lengths_m is None
    assert all(j.length_m == 1.0 for j in model.network.junctions.values())
