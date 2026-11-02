"""Numerical graph evidence must be complete, source-bound and recomputed."""

import json

import pytest

from scripts.hydraulic_graph_references import cases, reference
from scripts.qualify_hydraulic_graph import audit, qualify


@pytest.fixture
def packet(tmp_path):
    folder = tmp_path / "qualification"
    qualify(folder)
    return folder


def test_full_matrix_is_recomputed_with_reference(packet):
    result = audit(packet)
    assert result["passed"] and result["cases"] == 18 and result["maximum_nodes"] == 128
    assert not result["validated"] and not result["plant_referenced"]


@pytest.mark.parametrize(
    "target", ["budget", "driver", "request", "solution", "expected", "matrix", "metrics"]
)
def test_tampering_cannot_receive_audit_acceptance(packet, target):
    if target == "budget":
        (packet / "budget.json").write_text("{}")
    elif target == "driver":
        (packet / "drivers" / "solve_hydraulic_graph.py").write_text("# changed")
    elif target in ("request", "solution", "expected"):
        (packet / "3_linear" / (target + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        report = json.loads(path.read_text())
        if target == "matrix":
            report["cases"].pop()
        else:
            report["cases"][0]["metrics"]["reference_pa"] = 1.0
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(packet)


def test_reference_uses_no_production_inverse_or_compiler(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent reference used production")

    monkeypatch.setattr("scripts.solve_hydraulic_graph.edge_flows", forbidden)
    monkeypatch.setattr("scripts.solve_hydraulic_graph.compile_graph", forbidden)
    _, spec, exact = next(cases())
    result = reference(spec)
    assert result["pressures_pa"] == pytest.approx(exact["pressures_pa"], abs=1e-6)
    assert result["flows_kg_s"] == pytest.approx(exact["flows_kg_s"], abs=1e-9)


def test_audit_rejects_resealed_solution(packet):
    from scripts.solve_hydraulic_graph import sha

    path = packet / "3_linear" / "solution.json"
    solution = json.loads(path.read_text())
    solution["flows_kg_s"]["edge0"] += 1
    path.write_text(json.dumps(solution))
    report_path = packet / "qualification.json"
    report = json.loads(report_path.read_text())
    report["cases"][0]["files"]["solution.json"] = sha(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="differs"):
        audit(packet)
