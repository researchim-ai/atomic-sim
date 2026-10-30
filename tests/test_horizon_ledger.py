"""Накопительный баланс короткого горизонта и манифест непринятых режимов."""

import json
import math
from pathlib import Path

import pytest

from plants.ledger import HorizonLedger
from plants.vver import VVERPlant


def test_balanced_source_and_sink_do_not_collapse_the_error_scale():
    class BalancedVolume:
        horizon_ledger = None
        last_energy_ledger = None
        stored = 1000.0

        def closed_inventories(self):
            return {"stored_j": self.stored, "closed_mass_kg": 1.0, "boron_kg": 1.0}

    volume = BalancedVolume()
    ledger = HorizonLedger()
    ledger.attach(volume)
    volume.stored += 10.0
    volume.last_energy_ledger = {
        "supplied_j": {"source": 1e9, "sink": -1e9},
        "residual_j": 10.0,
    }
    ledger.observe(volume)
    report = ledger.report()
    assert report["energy_global_j"] == 10.0
    assert report["energy_scale_j"] == 2e9
    assert report["energy_relative"] == pytest.approx(5e-9)
    assert report["energy_net_relative"] == 10.0


def _hold(rtol: float, atol: float, steps: int) -> dict:
    plant = VVERPlant()
    HorizonLedger().attach(plant)
    for _ in range(steps):
        plant.advance(0.25, rtol=rtol, atol=atol)
    return plant.horizon_ledger.report()


def test_closed_mass_boron_and_energy_drift_agree():
    report = _hold(1.0e-5, 1.0e-4, 2)
    assert report["steps"] == 2
    assert report["full_horizon_accepted"] is False
    assert report["mass_relative"] < 1.0e-6
    assert report["boron_relative"] < 1.0e-6
    assert report["energy_global_j"] == pytest.approx(report["energy_integral_j"], abs=1.0)


def test_tighter_tolerance_reduces_the_energy_drift():
    # На неподвижном номинале остаток уже около ULP полного запаса (~1e12 Дж).
    # Порядок сходимости проверяем на ненулевом тепловом переходе, а не шуме округления.
    errors = []
    for tolerance in (1e-3, 1e-4, 1e-5):
        plant = VVERPlant()
        plant.sg_wall_temperature += 30.0
        plant.sg_wall.temperature_k = plant.sg_wall_temperature
        HorizonLedger().attach(plant)
        plant.advance(0.25, rtol=tolerance, atol=10 * tolerance)
        report = plant.horizon_ledger.report()
        errors.append(report["energy_max_abs_j"])
        assert report["full_horizon_accepted"] is False
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.2 * errors[0]
    assert errors[2] < 1e-5 * plant.fission_power_nom * 0.25


def test_stationary_balance_is_bounded_by_inventory_roundoff():
    plant = VVERPlant()
    roundoff_bound = 16 * math.ulp(plant.closed_inventories()["stored_j"])
    for tolerance in (1e-3, 1e-5):
        report = _hold(tolerance, 10 * tolerance, 1)
        assert report["energy_max_abs_j"] <= roundoff_bound


def test_run_manifest_keeps_long_horizons_unaccepted():
    payload = json.loads(Path("validation/run_manifest.json").read_text(encoding="utf-8"))
    assert payload["validated"] is False
    assert payload["plant_referenced"] is False
    modes = {item["id"]: item for item in payload["modes"]}
    for name in ("nominal_3600s", "fast_transient_60s", "coastdown_1800s", "load_rejection"):
        assert modes[name]["accepted_full_horizon"] is False
    assert modes["natural_circulation_in_vver"]["status"] == "not_in_passport"
