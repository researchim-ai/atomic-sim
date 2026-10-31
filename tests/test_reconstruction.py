import numpy as np
import pytest

from physics.hydraulics.reconstruction import conservative_rate, face_values
from scripts.verify_advection import evolve


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_mc_reproduces_linear_profile_including_both_boundary_outflows(sign):
    # Cell centers .5,1.5,..., boundary values at x=0,5.
    h = np.arange(5) + 0.5
    flow = np.full(6, sign)
    assert face_values(h, flow, 0.0, 5.0, scheme="muscl_mc") == pytest.approx(
        np.arange(6), abs=1e-14
    )


@pytest.mark.parametrize("scheme", ["donor", "muscl_mc"])
def test_shared_fluxes_cancel_for_mixed_signed_internal_flows(scheme):
    h = np.array([1.0, 3.0, 2.0, 5.0])
    flows = np.array([1.0, -0.5, 0.2, -2.0, -0.3])
    face = face_values(h, flows, 0.0, 4.0, scheme=scheme)
    rate = conservative_rate(h, flows, left_value=0.0, right_value=4.0, scheme=scheme)
    assert sum(rate) == pytest.approx(flows[0] * face[0] - flows[-1] * face[-1], abs=1e-14)
    assert np.all(face[1:-1] >= np.minimum(h[:-1], h[1:]))
    assert np.all(face[1:-1] <= np.maximum(h[:-1], h[1:]))


def test_periodic_flux_is_exactly_shared_and_constant_field_is_preserved():
    f = np.ones(5)
    assert conservative_rate(np.ones(4), f, scheme="muscl_mc", periodic=True) == pytest.approx(
        np.zeros(4)
    )
    with pytest.raises(ValueError, match="Periodic"):
        face_values([1.0, 2.0], [1.0, 1.0, 2.0], scheme="muscl_mc", periodic=True)


@pytest.mark.parametrize("speed", [-1.0, 1.0])
def test_exact_transport_improves_with_mc_and_preserves_periodic_inventory(speed):
    n = 64
    x = (np.arange(n) + 0.5) / n
    u = 1 + 0.2 * np.sin(2 * np.pi * x)
    exact = 1 + 0.2 * np.sin(2 * np.pi * (x - speed * 0.25))
    first = evolve(u, speed, 0.25, scheme="donor")
    second = evolve(u, speed, 0.25, scheme="muscl_mc")
    assert np.mean(abs(second - exact)) < 0.2 * np.mean(abs(first - exact))
    assert np.mean(second) == pytest.approx(np.mean(u), abs=1e-13)


@pytest.mark.parametrize(
    "values,flows", [([1.0, float("nan")], [1.0, 1.0, 1.0]), ([1.0, 2.0], [1.0, 1.0])]
)
def test_bad_transport_inputs_are_refused(values, flows):
    with pytest.raises(ValueError):
        face_values(values, flows, 0.0, 0.0, scheme="muscl_mc")
