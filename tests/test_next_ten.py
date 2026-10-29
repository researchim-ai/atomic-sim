"""Десять пунктов плана: измерения, защита, схема, Lambda, кипение, выгорание, разреженная диффузия, свинец, стенка."""

import math

import numpy as np
import pytest

from components.heat_exchanger import HeatExchangerWall
from control.regulation import ProtectionLogic, protection_step_good_channels
from core.mesh import Mesh1D
from core.status import DomainError
from instrumentation.channels import Measurement, Sensor
from physics.depletion.predictor import number_density_exponential, predictor_corrector_density
from physics.hydraulics.closures import heat_flux_w_m2, thom_nucleate_flux_w_m2
from physics.materials.coolants import lead_internal_energy_j_kg, lead_specific_heat_j_kg_k
from physics.neutronics.diffusion import (
    adjoint_generation_time_s,
    adjoint_k_eigenvalue,
    diffusion_matvec,
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
)
from scenarios.session import run_scenario


def test_ten_plan_items(tmp_path):
    sensor = Sensor(tau_s=1.0e-9, value=0.0)
    level = Measurement("level", sensor)
    reading = level.read(1.25, 0.2, np.random.default_rng(1))
    assert reading["unit"] == "m"
    assert reading["time_s"] == pytest.approx(0.2)
    assert reading["value"] == pytest.approx(1.25, abs=1.0e-6)
    assert reading["quality"] == "good"
    with pytest.raises(ValueError):
        Measurement("void", Sensor(1.0)).read(0.0, 0.1, np.random.default_rng(0))

    logic = ProtectionLogic(setpoint=1.0, delay_s=0.1)
    assert protection_step_good_channels(logic, [2.0, 2.0, 2.0], ["good", "bad", "bad"], 0.05) is False
    paired = ProtectionLogic(setpoint=1.0, delay_s=0.1)
    assert protection_step_good_channels(paired, [2.0, 2.0, 0.0], ["good", "good", "bad"], 0.1) is True
    stuck = ProtectionLogic(setpoint=1.0, delay_s=0.1)
    assert protection_step_good_channels(stuck, [2.0, 2.0, 2.0], ["bad", "bad", "bad"], 0.1) is False

    missing = tmp_path / "noschema.yaml"
    missing.write_text("name: x\nengine: vver\n", encoding="utf-8")
    with pytest.raises(DomainError):
        run_scenario(missing)

    mesh = Mesh1D.uniform(1.0, 12)
    diffusion = np.full(12, 0.01)
    absorption = np.full(12, 0.8)
    nu_sigma = np.full(12, 1.1)
    removal = diffusion_operator(mesh, diffusion, absorption)
    fission = fission_operator(mesh, nu_sigma)
    _k, phi, ok = k_eigenvalue(removal, fission)
    _k_star, phi_star, ok_star = adjoint_k_eigenvalue(removal, fission)
    assert ok and ok_star
    generation = adjoint_generation_time_s(phi, phi_star, nu_sigma, 2200.0, mesh.dx)
    assert generation == pytest.approx(1.0 / (2200.0 * 1.1), rel=1.0e-6)
    phi_probe = np.linspace(0.2, 1.0, 12)
    assert diffusion_matvec(mesh, diffusion, absorption, phi_probe) == pytest.approx(removal @ phi_probe)

    pressure = 70.0e5
    delta = 22.65 * math.exp(-(pressure / 1.0e5) / 87.0) * math.sqrt(0.4)
    flux = thom_nucleate_flux_w_m2(600.0 + delta, 600.0, pressure)
    assert flux == pytest.approx(0.4e6, rel=1.0e-10)
    assert heat_flux_w_m2("single_phase", coefficient_w_m2_k=1000.0, wall_k=610.0, bulk_k=600.0) == pytest.approx(1.0e4)
    with pytest.raises(DomainError):
        heat_flux_w_m2("post_chf", wall_k=900.0, saturation_k=600.0, pressure_pa=pressure)

    exact = number_density_exponential(100.0, 1.0, 0.1, 2.0)
    assert predictor_corrector_density(100.0, 1.0, 0.1, 0.1, 2.0) == pytest.approx(exact)
    assert exact == pytest.approx(100.0 * math.exp(-0.2))

    step = 0.05
    temperature = 800.0
    slope = (
        lead_internal_energy_j_kg(temperature + step) - lead_internal_energy_j_kg(temperature - step)
    ) / (2.0 * step)
    assert slope == pytest.approx(lead_specific_heat_j_kg_k(temperature), rel=1.0e-6)

    wall = HeatExchangerWall(4000.0, 1000.0, 1000.0, 300.0)
    wall.advance(600.0, 400.0, 2.0)
    assert wall.temperature_k == pytest.approx(500.0 + (300.0 - 500.0) * math.exp(-1.0), rel=1.0e-12)
