"""Measured meshes, independent IVPs and complete source-bound nonlinear replay."""

import json

import pytest

from scripts.nonlinear_graph_references import cases, reference
from scripts.qualify_nonlinear_graph import audit, evaluate, qualify


@pytest.fixture
def packet(tmp_path, monkeypatch):
    first = list(cases())[:1]
    monkeypatch.setattr("scripts.qualify_nonlinear_graph.cases", lambda: iter(first))
    monkeypatch.setattr("scripts.qualify_nonlinear_graph.TIMES", (0.0, 0.5, 1.0))
    folder = tmp_path / "matrix"
    qualify(folder)
    return folder


def test_evidence_recomputes_real_meshes_and_adaptive_restart(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 2 and result["adaptive_restart_identical"]


def test_declared_matrix_has_actual_reversal_and_eight_node_case():
    matrix = list(cases())
    assert len(matrix) == 10
    assert max(len(s["volumes_m3"]) for _, s in matrix) == 8
    assert any(name == "bridge_reversal" for name, _ in matrix)


@pytest.mark.parametrize("damage", ["driver", "budget", "request", "result", "matrix", "metrics"])
def test_evidence_tampering_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers" / "nonlinear_thermal_graph.py").write_text("# altered")
    elif damage == "budget":
        (packet / "budgets" / "ai05_nonlinear_graph_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "pipe_negative" / (damage + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        data = json.loads(path.read_text())
        if damage == "matrix":
            data["cases"] = []
        else:
            data["cases"][0]["metrics"]["adaptive_restart_identical"] = False
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_trajectory_does_not_bypass_numerical_audit(packet):
    from scripts.solve_hydraulic_graph import sha

    path = packet / "pipe_negative" / "result.json"
    data = json.loads(path.read_text())
    data["meshes"][-1]["states"][-1]["temperatures_k"]["a"] += 1
    path.write_text(json.dumps(data))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs"):
        audit(packet)


def test_reference_uses_neither_production_rk_nor_hydraulic_inverse(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent IVP called production")

    monkeypatch.setattr(
        "scripts.nonlinear_thermal_graph.NonlinearThermalGraph._evaluate", forbidden
    )
    monkeypatch.setattr("scripts.solve_hydraulic_graph.edge_flows", forbidden)
    _, spec = next(cases())
    result = reference(spec, "DOP853", (0.0, 1.0))
    assert result["times_s"][-1] == 1


def test_repeated_caps_are_not_counted_as_distinct_measured_meshes(monkeypatch):
    monkeypatch.setattr("scripts.qualify_nonlinear_graph.TIMES", (0.0, 1.0))
    monkeypatch.setattr("scripts.qualify_nonlinear_graph.MESH_STEPS", (0.5, 0.5, 0.5))
    _, spec = next(cases())
    with pytest.raises(ValueError, match="distinct"):
        evaluate(spec)
