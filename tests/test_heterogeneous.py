"""Неоднородная пластина, пик ксенона, квант датчика и фаза клапана."""

import numpy as np
import pytest

from components.equipment import orifice_mass_flow, valve_mass_flow
from core.mesh import Mesh1D
from core.status import DomainError
from instrumentation.channels import Sensor
from physics.depletion.chains import XenonIodine
from physics.hydraulics.two_phase_ext import critical_mass_flux_ideal_gas, frozen_mixture_critical_flux
from physics.neutronics.diffusion import (
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    two_region_reflector_k,
)


def test_reflected_slab_matches_transcendental_k():
    diffusion_core = 0.012
    sigma_core = 0.08
    nu_sigma_f = 0.10
    diffusion_reflector = 0.02
    sigma_reflector = 0.02
    core = 0.5
    reflector = 0.3
    expected = two_region_reflector_k(
        diffusion_core, sigma_core, nu_sigma_f, diffusion_reflector, sigma_reflector, core, reflector
    )
    cells = 160
    mesh = Mesh1D.uniform(core + reflector, cells)
    core_cells = int(round(core / mesh.dx[0]))
    diffusion = np.full(cells, diffusion_reflector)
    absorption = np.full(cells, sigma_reflector)
    fission = np.zeros(cells)
    diffusion[:core_cells] = diffusion_core
    absorption[:core_cells] = sigma_core
    fission[:core_cells] = nu_sigma_f
    removal = diffusion_operator(mesh, diffusion, absorption, left="reflective", right="dirichlet0")
    source = fission_operator(mesh, fission)
    value, _flux, converged = k_eigenvalue(removal, source, tol=1e-11)
    assert converged
    assert value == pytest.approx(expected, rel=2.0e-3)


def test_shutdown_xenon_peaks_then_sensor_and_valve_keep_their_models():
    xenon = XenonIodine()
    equilibrium = xenon.equilibrium(1.0e20, 1.0e18)
    peak = xenon.shutdown_peak_time_s(equilibrium[0], equilibrium[1])
    state = xenon.shutdown_xenon(equilibrium[0], equilibrium[1], peak)
    slope = xenon.derivatives(state[0], state[1], 0.0, 0.0)[1]
    assert slope == pytest.approx(0.0, abs=1.0e-6 * state[1] * xenon.lambda_xe)
    assert state[1] > equilibrium[1]

    sensor = Sensor(tau_s=1.0e-9, value=0.0, quantum=0.25, drift_per_s=0.1)
    measured = sensor.step(0.62, 1.0, np.random.default_rng(0))
    assert measured == pytest.approx(0.75)
    assert sensor.bias == pytest.approx(0.1)

    area = 0.01
    mixture = valve_mass_flow("mixture", area, 7.0e6, 1.0e6, quality=0.2)
    assert mixture == pytest.approx(frozen_mixture_critical_flux(7.0e6, 0.2) * area)
    vapor = valve_mass_flow("vapor", area, 1.0e6, 1.0e5, temperature_k=500.0, gamma=1.3, gas_constant_j_kg_k=461.5)
    assert vapor == pytest.approx(
        critical_mass_flux_ideal_gas(1.0e6, 500.0, 1.3, 461.5) * area
    )
    liquid = valve_mass_flow("liquid", area, 1.0e6, 0.8e6, density_kg_m3=800.0)
    assert liquid == pytest.approx(orifice_mass_flow(0.62, area, 800.0, 0.2e6))
    with pytest.raises(DomainError):
        valve_mass_flow("mixture", area, 7.0e6, 1.0e6, quality=0.0)
    with pytest.raises(DomainError):
        valve_mass_flow("liquid", area, 1.0e6, 0.8e6, density_kg_m3=800.0, quality=0.4)
