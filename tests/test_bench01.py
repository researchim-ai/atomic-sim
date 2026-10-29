"""Замороженный допуск BENCH-01. Коэффициенты IF97 под этот файл не подстраиваются."""

import json
from pathlib import Path

import pytest

from physics.materials.iapws import properties_pt


def test_frozen_if97_budget_holds_without_retuning():
    payload = json.loads(Path("validation/bench01_budget.json").read_text(encoding="utf-8"))
    assert payload["parameters_were_not_adjusted_to_this_file"] is True
    assert payload["accepted_as_plant_validation"] is False
    points = {item["id"]: item for item in payload["points"]}
    budget = points["if97-region3-rho-500-t-650"]
    region3 = properties_pt(budget["inputs"]["p_pa"], budget["inputs"]["T_k"])
    assert region3.rho_kg_m3 == pytest.approx(
        budget["reference"]["rho_kg_m3"], rel=budget["relative_budget"]
    )
    assert region3.h_j_kg == pytest.approx(
        budget["reference"]["h_j_kg"], rel=budget["relative_budget"]
    )
    region5 = properties_pt(0.5e6, 1500.0)
    other = points["if97-region5-t-1500-p-0.5mpa"]
    assert region5.v_m3_kg == pytest.approx(
        other["reference"]["v_m3_kg"], rel=other["relative_budget"]
    )
    assert region5.h_j_kg == pytest.approx(
        other["reference"]["h_j_kg"], rel=other["relative_budget"]
    )
