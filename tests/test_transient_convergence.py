"""Convergence gates cannot accept temporal error masquerading as mesh order."""

import pytest

from physics.hydraulics.initialization import equilibrated_config
from scripts.verify_pipe_mesh import configuration
from scripts.verify_pipe_transient import _pipe_signature, assess


def traces(order=1.0, time_error=1e-5):
    rows = []
    for n, extra in ((4, 0.0), (8, 0.0), (16, 0.0), (16, time_error)):
        rows.append(
            {
                t: dict(
                    right_cell_temperature_k=500.0 + t * (1.0 + n ** (-order) + extra),
                    mean_wall_temperature_k=505.0 - t * (0.5 + n ** (-order) + extra),
                    midpoint_pressure_pa=7e6,
                    inlet_mass_flow_kg_s=1.0,
                    outlet_mass_flow_kg_s=1.0,
                )
                for t in (0.0, 1.0, 2.0)
            }
        )
    return rows


def test_first_order_and_small_temporal_error_pass():
    metrics, _, checks = assess(traces())
    assert all(checks.values())
    assert metrics["right_cell_temperature_k"]["observed_order"] == pytest.approx(1.0)


def test_temporal_error_larger_than_mesh_error_fails():
    _, _, checks = assess(traces(time_error=0.1))
    assert not checks["right_cell_temperature_k"]


def test_flat_sequences_do_not_claim_infinite_order():
    rows = traces(order=0.0)
    _, _, checks = assess(rows)
    assert not checks["mean_wall_temperature_k"]


def test_wrong_physical_pipe_partition_is_rejected():
    spec = dict(hydraulics=equilibrated_config(configuration(4, 0.0), wall_offset_k=5.0).as_dict())
    spec["hydraulics"]["junction_lengths_m"][0] *= 2
    with pytest.raises(ValueError, match="physical pipe"):
        _pipe_signature(spec)


@pytest.mark.parametrize("shape", ["uniform", "sin2"])
@pytest.mark.parametrize("n", [4, 8, 16])
def test_generated_pulses_preserve_energy_and_physical_signature(shape, n):
    from physics.hydraulics.dynamic_channel import DynamicConfig, DynamicWallChannel
    from scripts.generate_pipe_pulse import make_spec

    spec = make_spec(n, shape=shape)
    signature = _pipe_signature(spec)
    assert signature["wall_pulse_shape"] == shape
    c = DynamicConfig(**spec["hydraulics"])
    model = DynamicWallChannel(c)
    base = DynamicWallChannel(equilibrated_config(configuration(n, 0.0)))
    assert model.initial_energy_j - base.initial_energy_j == pytest.approx(40000.0, abs=1e-7)
    assert model.observation()["energy_residual_j"] == 0.0


def test_pulse_generator_refuses_unsupported_shape_and_cell_count():
    from scripts.generate_pipe_pulse import make_spec

    with pytest.raises(ValueError):
        make_spec(1)
    with pytest.raises(ValueError):
        make_spec(4, shape="unknown")
