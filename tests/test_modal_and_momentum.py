"""Модальная кинетика с утечкой, выборка датчика и импульс фаз."""

import numpy as np
import pytest

from core.mesh import Mesh1D
from core.status import DomainError
from instrumentation.channels import Sensor
from physics.hydraulics.two_phase_ext import TwoFluidMomentum, interfacial_drag_pair
from physics.neutronics.diffusion import (
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    modal_delayed_step,
)
from physics.neutronics.point_kinetics import PointKinetics


def test_leaky_mode_matches_point_kinetics():
    mesh = Mesh1D.uniform(1.0, 8)
    diffusion = np.full(mesh.n_cells, 0.01)
    sigma_a = np.full(mesh.n_cells, 0.5)
    nu_sigma_f = np.full(mesh.n_cells, 0.55)
    speed = 2200.0
    beta = 0.0065
    decay = 0.08
    removal = diffusion_operator(mesh, diffusion, sigma_a)
    fission = fission_operator(mesh, nu_sigma_f)
    k_value, shape, converged = k_eigenvalue(removal, fission, tol=1e-12)
    assert converged
    precursor0 = beta * nu_sigma_f * shape / decay
    flux, precursor = modal_delayed_step(
        mesh, diffusion, sigma_a, nu_sigma_f, speed, beta, decay, shape, precursor0, 0.05
    )
    kinetics = PointKinetics(
        np.array([beta]), np.array([decay]), generation_time_s=1.0 / (speed * nu_sigma_f[0])
    )
    rho = (k_value - 1.0) / k_value
    reference = kinetics.reference_step(np.array([1.0, speed * precursor0[np.argmax(shape)]]), rho, 0.05)
    amplitude = flux / shape
    assert amplitude == pytest.approx(reference[0], rel=1e-8)
    assert speed * precursor / shape == pytest.approx(reference[1], rel=1e-8)


def test_sensor_holds_between_samples_and_phase_momentum_cancels_drag():
    sensor = Sensor(tau_s=1.0e-6, value=3.0, sample_period_s=1.0)
    rng = np.random.default_rng(0)
    assert sensor.step(10.0, 0.4, rng) == pytest.approx(3.0)
    assert sensor.quality == "held"
    assert sensor.step(10.0, 0.4, rng) == pytest.approx(3.0)
    assert sensor.step(10.0, 0.4, rng) == pytest.approx(10.0, abs=1.0e-4)

    pair = TwoFluidMomentum(4.0, 0.5, 1.0, 3.0)
    before = pair.momentum_kg_m_s()
    drag_on_vapor, drag_on_liquid = interfacial_drag_pair(pair.vapor_velocity_m_s - pair.liquid_velocity_m_s, 2.0)
    assert drag_on_liquid == pytest.approx(-drag_on_vapor)
    external_liquid = 10.0
    external_vapor = -4.0
    pair.step(0.2, external_liquid, external_vapor, drag_on_vapor)
    assert pair.momentum_kg_m_s() - before == pytest.approx(0.2 * (external_liquid + external_vapor))
    empty = TwoFluidMomentum(1.0, 0.0, 1.0, 0.0)
    with pytest.raises(DomainError):
        empty.step(0.1, 0.0, 0.0, 0.0)
