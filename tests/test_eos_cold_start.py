"""Обращение жидкости из холодных/газовых приближений без проекции запасов."""

import pytest

from core.status import DomainError
from physics.materials import iapws as eos
from sdk.api import create_session, export_snapshot, restore_snapshot


@pytest.mark.parametrize(
    "pressure,temperature",
    [
        (15.6082171031e6, 618.0602972834),
        (15.5976954793e6, 617.8110989055),
        (3e6, eos.saturation_temperature_k(3.0) - 0.01),
        (16e6, eos.saturation_temperature_k(16.0) - 0.01),
        (80e6, 600.0),
        (1e5, 300.0),
    ],
)
def test_cold_guesses_recover_both_density_and_energy_without_projection(
    monkeypatch, pressure, temperature
):
    target = eos.properties_pt(pressure, temperature)

    def no_projection(*args, **kwargs):
        raise AssertionError("No projection may replace a converged state")

    monkeypatch.setattr(eos, "_project_equilibrium", no_projection)
    for guess in (8000.0, 15.8e6, 80e6):
        actual = eos.properties_rho_u(target.rho_kg_m3, target.u_j_kg, guess, 280.0)
        assert actual.region == 1
        assert actual.p_pa == pytest.approx(pressure, rel=2e-8, abs=0.005)
        assert actual.T_k == pytest.approx(temperature, abs=2e-8)
        assert actual.v_m3_kg == pytest.approx(target.v_m3_kg, rel=1e-11)
        assert actual.u_j_kg == pytest.approx(target.u_j_kg, rel=1e-11)
        assert actual.energy_mismatch_j_kg == 0.0
        assert actual.volume_mismatch == 0.0


def test_energy_seed_is_used_when_general_root_declines(monkeypatch):
    target = eos.properties_pt(15.6e6, 618.0)

    def fail(*args, **kwargs):
        raise DomainError("general root failed")

    monkeypatch.setattr(eos, "_invert_single_phase", fail)
    result = eos.properties_rho_u(target.rho_kg_m3, target.u_j_kg, 15.8e6, 593.0)
    assert result.p_pa == pytest.approx(target.p_pa, abs=0.005)
    assert result.T_k == pytest.approx(target.T_k, abs=2e-8)


def test_cold_liquid_inverse_resolves_extremely_small_subcooling():
    pressure = 15.65e6
    for subcooling in (1e-5, 1e-7):
        target = eos.properties_pt(
            pressure, eos.saturation_temperature_k(pressure * 1e-6) - subcooling
        )
        for guess in (8000.0, 15.8e6, 80e6):
            result = eos.properties_rho_u(target.rho_kg_m3, target.u_j_kg, guess, 280.0)
            assert result.region == 1
            assert result.p_pa == pytest.approx(pressure, abs=0.005)
            assert result.v_m3_kg == pytest.approx(target.v_m3_kg, rel=1e-11)
            assert result.u_j_kg == pytest.approx(target.u_j_kg, rel=1e-11)


def test_liquid_seed_declines_vapor_and_incompatible_density():
    vapor = eos.properties_pt(0.5e6, 600.0)
    assert eos._invert_liquid_energy_seed(vapor.v_m3_kg, vapor.u_j_kg / 1000, 0.5e6) is None
    assert eos._invert_liquid_energy_seed(10.0, 1600.0, 15.8e6) is None


def test_hot_liquid_checkpoint_restores_exactly_into_new_nominal_session():
    original = create_session()
    target = eos.properties_pt(15.5976954793e6, 617.8110989055)
    snapshot = export_snapshot(original)
    volume = original.network.volumes["hot_leg"].volume_m3
    mass = target.rho_kg_m3 * volume
    snapshot["state"][16:19] = [mass, mass * target.u_j_kg, mass * 0.001]
    restored = create_session()
    restore_snapshot(restored, snapshot)
    assert export_snapshot(restored) == snapshot


@pytest.mark.parametrize("quality", [0.0, 1e-6, 2.5e-5, 1e-4, 1e-3])
def test_near_liquid_saturation_boundary_is_found_from_distant_guesses(quality):
    pressure = 15.65e6
    liquid, vapor = eos.saturation_properties(pressure)
    volume = (1 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg
    energy = (1 - quality) * liquid.u_j_kg + quality * vapor.u_j_kg
    for guess in (8000.0, 15.8e6, 80e6):
        state = eos.properties_rho_u(1 / volume, energy, guess, 280.0)
        assert state.p_pa == pytest.approx(pressure, abs=0.005)
        assert state.quality == pytest.approx(quality, abs=1e-10)
        assert state.v_m3_kg == pytest.approx(volume, rel=1e-11)
        assert state.u_j_kg == pytest.approx(energy, rel=1e-11)


@pytest.mark.parametrize("quality,allowed", [(2.5e-5, True), (1e-3, False)])
def test_plant_distinguishes_resolved_eos_from_its_boiling_limit(quality, allowed):
    plant = create_session()
    liquid, vapor = eos.saturation_properties(15.65e6)
    volume = (1 - quality) * liquid.v_m3_kg + quality * vapor.v_m3_kg
    energy = (1 - quality) * liquid.u_j_kg + quality * vapor.u_j_kg
    core = plant.network.volumes["core"]
    core.mass_kg = core.volume_m3 / volume
    core.internal_energy_j = core.mass_kg * energy
    core.sync()
    assert core.thermo.quality == pytest.approx(quality, abs=1e-10)
    if allowed:
        plant._check_domain()
    else:
        with pytest.raises(DomainError, match="Объёмное кипение"):
            plant._check_domain()
