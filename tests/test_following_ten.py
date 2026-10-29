"""Токи, стержень между узлами, Пикар, фазы, чувствительность и редукция."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from agent_envs.pwr_env import PWRPlantEnv
from core.status import DomainError, SolverError, classify_model_error
from coupling.picard import picard_power_temperature
from physics.hydraulics.two_phase_ext import limited_phase_heat_w, wall_heat_split
from physics.neutronics.diffusion import rod_absorption_2d
from physics.neutronics.transport import (
    diffusion_k_sparse,
    divergence_from_currents,
    face_currents_2d,
    operator_leakage,
    rayleigh_k,
    require_trained_mode,
    sparse_fixed_source,
)
from plants.vver import WORTH, VVERPlant


def test_face_currents_match_operator_leakage():
    shape = (6, 5)
    lengths = (1.2, 0.9)
    diffusion = np.full(shape, 0.012)
    diffusion[:, 0] = 0.02
    phi = np.linspace(0.2, 1.0, shape[0] * shape[1]).reshape(shape)
    leakage = operator_leakage(shape, lengths, diffusion, 0.2, phi)
    divergence = divergence_from_currents(shape, lengths, diffusion, phi)
    assert divergence.reshape(-1) == pytest.approx(leakage, rel=0.0, abs=1.0e-12)
    current_x, _current_y = face_currents_2d(shape, lengths, diffusion, phi)
    assert current_x.shape == (5, 5)


def test_partial_rod_integral_is_exact_on_any_row_count():
    base = np.ones((7, 4))
    added = rod_absorption_2d(base, 0.37, 3.0) - base
    assert float(np.sum(added)) == pytest.approx(3.0 * 0.37 * 7 * 4)


def test_picard_matches_the_scalar_root():
    power0 = 1.0e6
    sink = 300.0
    conductance = 2.0e4
    feedback = 0.002
    reference = 600.0

    def residual(temperature: float) -> float:
        power = power0 / (1.0 + feedback * (temperature - reference))
        return temperature - (sink + power / conductance)

    root = brentq(residual, 300.0, 900.0)
    temperature, power, converged = picard_power_temperature(
        power0, sink, conductance, feedback, reference
    )
    assert converged
    assert temperature == pytest.approx(root, abs=1.0e-8)
    assert power == pytest.approx(power0 / (1.0 + feedback * (temperature - reference)))


def test_wall_split_conserves_heat_and_phase_heat_stops_at_zero():
    liquid, vapor = wall_heat_split(1.0e5, 0.25)
    assert liquid + vapor == pytest.approx(1.0e5)
    assert limited_phase_heat_w(50.0, -40.0, 2.0) == pytest.approx(-25.0)
    with pytest.raises(DomainError):
        wall_heat_split(1.0, 1.2)


def test_boron_derivative_matches_the_frozen_coefficient():
    plant = VVERPlant()
    frozen = json.loads(Path("validation/frozen_vver_coefficients.json").read_text(encoding="utf-8"))
    assert frozen["frozen_before_scenario_tuning"] is True
    assert frozen["validated"] is False
    assert plant.boron_coef == pytest.approx(frozen["boron_coef_per_ppm"])
    assert plant.doppler_coef == pytest.approx(frozen["doppler_coef_per_sqrt_k"])
    assert plant.density_coef == pytest.approx(frozen["density_coef_per_kg_m3"])
    assert WORTH == pytest.approx(frozen["rod_worth"])
    before = plant.reactivity()
    plant.set_boron_ppm(plant.boron_ppm_ref + 25.0)
    assert (plant.reactivity() - before) / 25.0 == pytest.approx(plant.boron_coef)


def test_rayleigh_mode_and_fixed_source_and_error_classes():
    shape = (6, 6)
    lengths = (1.2, 1.2)
    k_value, phi, ok = diffusion_k_sparse(shape, lengths, 0.012, 0.2, 0.25)
    assert ok
    assert rayleigh_k(shape, lengths, 0.012, 0.2, 0.25, phi) == pytest.approx(k_value, rel=1.0e-8)
    assert require_trained_mode(phi, phi, 1.0e-8) == pytest.approx(0.0, abs=1.0e-12)
    with pytest.raises(DomainError):
        require_trained_mode(phi, np.arange(phi.size, dtype=float) + 1.0, 1.0e-6)
    source = operator_leakage(shape, lengths, 0.012, 0.2, phi) + 0.2 * phi
    restored = sparse_fixed_source(shape, lengths, 0.012, 0.2, source)
    assert restored == pytest.approx(phi.reshape(-1), rel=0.0, abs=1.0e-9)
    assert classify_model_error(DomainError("область")) == "domain"
    assert classify_model_error(SolverError("шаг")) == "solver"
    assert classify_model_error(ValueError("вход")) == "program"
    env = PWRPlantEnv()
    env.reset(seed=1)

    def fail(*_args, **_kwargs):
        raise SolverError("шаг")

    env.plant.advance = fail
    info = env.step(np.array([0.0], dtype=np.float32))[4]
    assert info["error_class"] == "solver"
