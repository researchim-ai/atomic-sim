"""Replay complete command matrices and reject resealed physical trajectories."""

import json
import shutil

import pytest

from scripts.qualify_controlled_radau import audit, cases, negative_cases, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    first = list(cases())[:1]
    negative = list(negative_cases())[:1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.qualify_controlled_radau.cases", lambda: iter(first))
        patch.setattr("scripts.qualify_controlled_radau.negative_cases", lambda: iter(negative))
        patch.setattr("scripts.qualify_controlled_radau.TIMES", (0.0, 37.5, 38.0))
        folder = tmp_path_factory.mktemp("controlled") / "matrix"
        qualify(folder)
        yield folder


@pytest.fixture
def packet(tmp_path, campaign):
    folder = tmp_path / "matrix"
    shutil.copytree(campaign, folder)
    return folder


def test_audit_recomputes_command_physics_restart_and_refusals(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 2 and result["negative_checks"] == 1
    assert result["restart_values_identical"]


def test_declared_matrix_covers_both_modes_and_warm_state():
    matrix = list(cases())
    assert len(matrix) == 9 and sum(mode == "scheduled" for _, mode, _ in matrix) == 7
    assert any(name == "scheduled_warm_reinitialization" for name, _, _ in matrix)
    assert len(list(negative_cases())) == 3


@pytest.mark.parametrize(
    "damage", ["driver", "budget", "request", "result", "matrix", "negative_matrix", "signature"]
)
def test_tampering_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers/controlled_reduced_engine.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets/ai05_controlled_radau_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "scheduled_heater_pulse" / (damage + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        value = json.loads(path.read_text())
        if damage == "matrix":
            value["cases"] = []
        elif damage == "negative_matrix":
            value["negative_cases"] = []
        else:
            value["signature"] = {}
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_command_energy_is_independently_recomputed(packet):
    path = packet / "scheduled_heater_pulse/result.json"
    value = json.loads(path.read_text())
    value["meshes"][-1]["states"][-1]["engine"]["state"]["integrated_external_energy_j"] += 100.0
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)


def test_one_ulp_step_difference_is_not_an_independent_time_mesh():
    import math

    from scripts.qualify_controlled_radau import require_mesh_separation

    with pytest.raises(ValueError, match="sufficiently distinct"):
        require_mesh_separation([1.0, math.nextafter(1.0, 0.0), 0.5])
