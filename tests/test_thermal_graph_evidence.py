"""Independent thermal references and complete, tamper-resistant replay."""

import json

import pytest

from scripts.qualify_thermal_graph import audit, qualify
from scripts.thermal_graph_references import limiting_cases, reference


@pytest.fixture
def packet(tmp_path):
    folder = tmp_path / "matrix"
    qualify(folder)
    return folder


def test_thermal_matrix_is_fully_recomputed(packet):
    result = audit(packet)
    assert result["passed"] and result["cases"] == 21 and result["maximum_nodes"] == 128
    assert not result["validated"]


@pytest.mark.parametrize(
    "target", ["budget", "driver", "request", "solution", "expected", "matrix", "metrics"]
)
def test_tampered_evidence_is_rejected(packet, target):
    if target == "budget":
        (packet / "budgets" / "ai05_thermal_graph_budget_v1.json").write_text("{}")
    elif target == "driver":
        (packet / "drivers" / "solve_thermal_graph.py").write_text("# tampered")
    elif target in ("request", "solution", "expected"):
        (packet / "3_linear" / (target + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        d = json.loads(path.read_text())
        if target == "matrix":
            d["cases"].pop()
        else:
            d["cases"][0]["metrics"]["reference_error_k"] = 1.0
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_numerical_result_still_fails_audit(packet):
    from scripts.solve_hydraulic_graph import sha

    path = packet / "3_linear" / "solution.json"
    solved = json.loads(path.read_text())
    solved["temperatures_k"]["node0"] += 1
    path.write_text(json.dumps(solved))
    report = packet / "qualification.json"
    d = json.loads(report.read_text())
    d["cases"][0]["files"]["solution.json"] = sha(path)
    report.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="differs"):
        audit(packet)


def test_reference_does_not_call_production_hydraulic_or_thermal_solver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Reference called production")

    monkeypatch.setattr("scripts.solve_thermal_graph.solve_thermal_graph", forbidden)
    monkeypatch.setattr("scripts.solve_thermal_graph.prepare", forbidden)
    monkeypatch.setattr("scripts.solve_hydraulic_graph.solve_hydraulic_graph", forbidden)
    for _, spec, exact in limiting_cases():
        assert reference(spec)["temperatures_k"] == pytest.approx(exact["temperatures_k"], abs=1e-7)
