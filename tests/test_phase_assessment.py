"""Independent trace integrity and four-phenomenon numerical verification."""

import json

import pytest

from atomic_model_phase_assessment.assessment import evaluate
from atomic_model_phase_assessment.dataset import load
from atomic_model_phase_assessment.importer import METADATA, PRESSURE, THERMAL, decode
from atomic_model_phase_assessment.vessel import Vessel, cases, qualify


def test_primary_histories_and_admission():
    m, t = load()
    assert m["native_counts"] == dict(pressure=8150000, thermal=975000)
    assert len(t["pressure"]) == 3261 and len(t["thermal"]) == 3251
    assert t["pressure"][-1]["time_s"] != t["thermal"][-1]["time_s"]
    r = evaluate()
    assert r["production_forced_boiling_admitted_rows"] == 0
    assert r["thermal_regime_census"]["negative_heat_flux"] > 0
    assert not r["validated"]


def sample_row():
    r = {k: "" for k in set(METADATA) | set(PRESSURE.values()) | set(THERMAL.values())}
    for k in METADATA:
        r[k] = "1"
    for k in PRESSURE.values():
        r[k] = "1"
    for k in THERMAL.values():
        r[k] = "1"
    return r


@pytest.mark.parametrize("change", [{"Time_Thermocouple (seconds)": ""}, {"Pressure (kPa)": "inf"}])
def test_invalid_native_rows_rejected(change):
    r = sample_row()
    r.update(change)
    with pytest.raises(ValueError):
        decode([r])


def test_four_phenomena_step_refinement_restart():
    result = qualify()
    assert {r["case"] for r in result["cases"]} == {
        "boiling",
        "condensation",
        "flashing",
        "limited_blowdown",
    }
    assert sum(len(r["runs"]) for r in result["cases"]) == 12
    assert not result["validated"]


def test_vessel_checkpoint_json_and_tampering():
    model = Vessel(cases()[0])
    s = model.advance(model.initial, 1.0, 0.4)
    saved = json.loads(json.dumps(model.checkpoint(s)))
    assert model.restore(saved) == s
    saved["payload"]["stocks"][0] += 1.0
    with pytest.raises(ValueError):
        model.restore(saved)


@pytest.mark.parametrize("cap", [0.0, 1e-20, float("inf"), float("nan"), 2.0])
def test_vessel_refuses_unbounded_work_request(cap):
    model = Vessel(cases()[0])
    before = model.initial
    with pytest.raises(ValueError):
        model.advance(before, 1.0, cap)
    assert model.initial == before
