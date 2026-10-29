"""Десять следующих пунктов: режимы ВВЭР, группы стержней, токи, post-CHF, матрица."""

import json
from pathlib import Path

import pytest

from control.regulation import RodBank, RodBanks
from core.status import DomainError
from physics.hydraulics.closures import heat_flux_w_m2
from physics.neutronics.transport import diffusion_k_sparse, leakage_balance, modal_delayed_step_nd
from scenarios.session import run_scenario


def test_four_vver_scenarios_move_the_requested_quantity():
    ramp = run_scenario(Path("scenarios/vver_load_ramp.yaml"))
    assert ramp["rows"][-1]["electrical_power_w"] < ramp["rows"][0]["electrical_power_w"]
    boron = run_scenario(Path("scenarios/vver_boron_step.yaml"))
    assert boron["rows"][-1]["reactivity"] < boron["rows"][0]["reactivity"]
    assert boron["rows"][-1]["energy_residual_j"] is not None
    pump = run_scenario(Path("scenarios/vver_pump_trip.yaml"))
    assert pump["rows"][-1]["mass_flow_kg_s"] < pump["rows"][0]["mass_flow_kg_s"]
    feed = run_scenario(Path("scenarios/vver_feed_step.yaml"))
    assert feed["rows"][-1]["feed_flow_kg_s"] > feed["rows"][0]["feed_flow_kg_s"]


def test_scram_overrides_every_rod_bank():
    banks = RodBanks(
        [
            RodBank("shutdown", 0.2, 0.4, 0.01, 0.2),
            RodBank("regulating", 0.5, 0.1, 0.004, 0.5),
        ]
    )
    banks.step(command=-1.0, scram=True, dt_s=0.5)
    assert banks.banks[0].insertion == pytest.approx(0.4)
    assert banks.banks[1].insertion == pytest.approx(0.55)
    assert banks.reactivity() < 0.0


def test_leakage_matches_fission_over_k_and_critical_precursors_stay():
    shape = (8, 8)
    lengths = (1.2, 1.2)
    diffusion = 0.012
    sigma = 0.2
    nu_sigma = 0.25
    k_value, phi, ok = diffusion_k_sparse(shape, lengths, diffusion, sigma, nu_sigma)
    assert ok
    removal, source = leakage_balance(shape, lengths, diffusion, sigma, nu_sigma, k_value, phi)
    assert removal == pytest.approx(source, rel=1.0e-6)
    critical = nu_sigma / k_value
    beta = 0.0065
    decay = 0.08
    precursor = beta * critical * phi / decay
    flux_new, precursor_new = modal_delayed_step_nd(
        shape, lengths, diffusion, sigma, critical, 2200.0, beta, decay, phi, precursor, 0.05
    )
    assert flux_new == pytest.approx(phi, rel=1.0e-6, abs=1.0e-8)
    assert precursor_new == pytest.approx(precursor, rel=1.0e-6, abs=1.0e-8)


def test_post_chf_is_refused_and_the_matrix_points_at_real_tests():
    with pytest.raises(DomainError):
        heat_flux_w_m2("post_chf")
    payload = json.loads(Path("validation/phenomenon_matrix.json").read_text(encoding="utf-8"))
    assert payload["validated"] is False
    assert payload["plant_referenced"] is False
    for row in payload["rows"]:
        assert Path(row["test"]).is_file()
    assert len(payload["rows"]) >= 9
