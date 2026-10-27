"""База этапа E01 плана 3.0: что входит в ВВЭР и какие команды недопустимы."""

import json
import math
from pathlib import Path

import pytest

from sdk.api import apply_commands, create_session


def test_baseline_keeps_new_components_out_of_the_vver_passport():
    payload = json.loads(Path("validation/e01_baseline.json").read_text(encoding="utf-8"))
    assert payload["plan_version"] == "3.0"
    assert payload["stage"] == "E01"
    assert payload["stage_status"] != "accepted"
    assert payload["accepted_as_complete_product"] is False
    assert payload["claims"]["validated"] is False
    assert payload["claims"]["plant_referenced"] is False
    assert "vver-lumped-loop" in payload["integrated_model_ids"]
    for model_id in ("dynamic-condenser", "heat-exchanger-wall", "natural-circulation-loop"):
        assert model_id in payload["standalone_model_ids_not_in_vver"]
        assert model_id not in payload["integrated_model_ids"]
    assert payload["vver_condenser"] == "fixed_vacuum_hotwell"


def test_invalid_commands_are_rejected_before_the_physics_step():
    plant = create_session("vver")
    throttle = plant.throttle
    with pytest.raises(ValueError):
        apply_commands(plant, {"not_a_command": 1})
    with pytest.raises(ValueError):
        apply_commands(plant, {"throttle": math.nan})
    with pytest.raises(ValueError):
        apply_commands(plant, {"throttle": 2.0})
    with pytest.raises(ValueError):
        apply_commands(plant, {"rod_command": 4.0})
    with pytest.raises(ValueError):
        apply_commands(plant, {"motor_on": True, "motor_off": True})
    with pytest.raises(ValueError):
        apply_commands(plant, {"scram": "yes"})
    assert plant.throttle == pytest.approx(throttle)
    apply_commands(plant, {"throttle": 0.4, "rod_command": -0.2, "scram": False})
    assert plant.throttle == pytest.approx(0.4)
    assert plant.scram_latched is False
