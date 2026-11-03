"""Reject incomplete or resealed online numerical evidence."""

import json

import pytest

from scripts.qualify_online_graph import audit, cases, evaluate, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture
def packet(tmp_path, monkeypatch):
    first = list(cases())[:1]
    monkeypatch.setattr("scripts.qualify_online_graph.cases", lambda: iter(first))
    monkeypatch.setattr("scripts.qualify_online_graph.DURATIONS", (1.0, 1.0, 1.0, 1.0))
    folder = tmp_path / "matrix"
    qualify(folder)
    return folder


def test_audit_recomputes_feedback_and_restart(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 6 and result["adaptive_restart_values_identical"]


def test_declared_matrix_includes_eight_nodes_and_both_control_kinds():
    matrix = dict(cases())
    assert len(matrix) == 4 and max(len(s["volumes_m3"]) for s in matrix.values()) == 8
    assert "pressure_reversal" in matrix and "inlet_feedback" in matrix


@pytest.mark.parametrize("damage", ["driver", "budget", "request", "result", "matrix", "signature"])
def test_tampering_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers/online_thermal_graph.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets/ai05_online_graph_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "thermal_feedback" / (damage + ".json")).write_text("{}")
    else:
        p = packet / "qualification.json"
        value = json.loads(p.read_text())
        if damage == "matrix":
            value["cases"] = []
        else:
            value["signature"] = {}
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_trajectory_is_recomputed(packet):
    path = packet / "thermal_feedback/result.json"
    value = json.loads(path.read_text())
    value["meshes"][-1]["observations"][-1]["model"]["state"]["temperatures_k"]["a"] += 1
    path.write_text(json.dumps(value))
    p = packet / "qualification.json"
    report = json.loads(p.read_text())
    report["cases"][0]["files"]["result.json"] = sha(path)
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)


def test_repeated_actual_meshes_are_rejected(monkeypatch):
    monkeypatch.setattr("scripts.qualify_online_graph.DURATIONS", (1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr("scripts.qualify_online_graph.MESH_STEPS", (0.125, 0.125, 0.125))
    name, spec = next(cases())
    with pytest.raises(ValueError, match="not distinct"):
        evaluate(name, spec)
