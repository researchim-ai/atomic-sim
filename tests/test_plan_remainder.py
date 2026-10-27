"""Барабан, натрий, графит, сопряжённое k, однородная кинетика и двухфазный критический поток."""

import numpy as np
import pytest

from components.separator import SteamDrum
from core.mesh import Mesh1D
from core.status import DomainError
from physics.hydraulics.two_phase_ext import frozen_mixture_critical_flux
from physics.materials.iapws import saturation_properties
from physics.neutronics.diffusion import (
    adjoint_k_eigenvalue,
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    uniform_delayed_step,
)
from physics.neutronics.point_kinetics import PointKinetics
from plants.reactor_types import BWRPlant, HTGRBrayton, SFRLoop


def test_drum_pressure_restores_geometric_volume_without_rewriting_energy():
    pressure = 6.4e6
    liquid, vapor = saturation_properties(pressure)
    volume = 20.0
    drum = SteamDrum(
        2.0,
        volume,
        pressure,
        0.55 * volume * liquid.rho_kg_m3,
        0.45 * volume * vapor.rho_kg_m3,
    )
    assert drum.occupied_volume_m3() == pytest.approx(volume, rel=1e-8)
    stored = drum.accounted_energy_j
    drum.pressure_pa = 4.0e6
    fitted = drum.fit_pressure_to_volume()
    assert fitted == pytest.approx(pressure, rel=1e-4)
    assert drum.occupied_volume_m3() == pytest.approx(volume, rel=1e-6)
    assert drum.accounted_energy_j == pytest.approx(stored)


def test_sodium_and_graphite_storage_and_bwr_heat():
    sodium = SFRLoop()
    sodium.temperature_k = sodium.steady_temperature()
    assert sodium.temperature_derivative_k_s() == pytest.approx(0.0, abs=1e-9)
    sodium.temperature_k -= 15.0
    assert sodium.temperature_derivative_k_s() > 0.0
    graphite = HTGRBrayton()
    graphite.graphite_temperature_k = graphite.core_temperature_k
    assert graphite.graphite_temperature_derivative_k_s() == pytest.approx(0.0)
    graphite.graphite_temperature_k -= 40.0
    assert graphite.graphite_temperature_derivative_k_s() > 0.0
    boiling = BWRPlant()
    report = boiling.separation()
    assert report["heat_w"] == pytest.approx(boiling.fission_power_w, rel=1e-8)
    assert report["steam_kg_s"] + report["recirculation_kg_s"] == pytest.approx(boiling.core_flow_kg_s)


def test_adjoint_k_matches_forward_and_uniform_kinetics_matches_point():
    mesh = Mesh1D.uniform(1.0, 24)
    removal = diffusion_operator(mesh, np.full(24, 0.01), np.full(24, 0.8))
    fission = fission_operator(mesh, np.full(24, 1.1))
    forward, _flux, ok = k_eigenvalue(removal, fission)
    adjoint, _weight, adjoint_ok = adjoint_k_eigenvalue(removal, fission)
    assert ok and adjoint_ok
    assert adjoint == pytest.approx(forward, rel=1e-8)

    sigma_a = 1.0
    nu_sigma_f = 1.03
    speed = 2200.0
    beta = 0.0065
    decay = 0.08
    flux = 2.0
    precursor = beta * nu_sigma_f * flux / decay
    physical = uniform_delayed_step(flux, precursor, sigma_a, nu_sigma_f, speed, beta, decay, 0.05)
    kinetics = PointKinetics(
        np.array([beta]), np.array([decay]), generation_time_s=1.0 / (speed * nu_sigma_f)
    )
    rho = (nu_sigma_f - sigma_a) / nu_sigma_f
    reference = kinetics.reference_step(np.array([flux, speed * precursor]), rho, 0.05)
    assert physical[0] == pytest.approx(reference[0], rel=1e-10)
    assert speed * physical[1] == pytest.approx(reference[1], rel=1e-10)


def test_frozen_two_phase_critical_flux_is_not_the_gas_model():
    low_quality = frozen_mixture_critical_flux(7.0e6, 0.08)
    high_quality = frozen_mixture_critical_flux(7.0e6, 0.7)
    assert low_quality > 0.0 and high_quality > 0.0
    assert low_quality != pytest.approx(high_quality, rel=1e-3)
    with pytest.raises(DomainError):
        frozen_mixture_critical_flux(7.0e6, 0.0)
    with pytest.raises(DomainError):
        frozen_mixture_critical_flux(7.0e6, 1.0)
