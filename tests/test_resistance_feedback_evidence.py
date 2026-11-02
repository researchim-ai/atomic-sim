"""Independent mixed roots and complete source-bound feedback evidence."""

import json

import pytest

from scripts.qualify_resistance_feedback import audit, qualify
from scripts.resistance_feedback_references import cases, reference
from scripts.solve_resistance_feedback import BUDGET


@pytest.fixture
def packet(tmp_path, monkeypatch):
    first = list(cases())[:1]
    monkeypatch.setattr("scripts.qualify_resistance_feedback.cases", lambda: iter(first))
    folder = tmp_path / "qualification"
    qualify(folder)
    return folder


def test_evidence_replays_coupled_solver_and_independent_root(packet):
    result = audit(packet)
    assert result["passed"] and result["cases"] == 1 and not result["global_uniqueness_proven"]


def test_declared_matrix_has_both_feedback_signs_and_internal_zero_flow():
    matrix = list(cases())
    assert len(matrix) == 20
    assert max(len(s["initial_temperatures_k"]) for _, s, _ in matrix) == 32
    assert any(name == "3_near_zero_negative" for name, _, _ in matrix)
    for name, spec, _ in matrix:
        if name.startswith("two_supplies"):
            mixer = next(n for n in spec["steady_input"]["hydraulic"]["nodes"] if n["id"] == "mix")
            assert mixer["injection_kg_s"] == 0 and "pressure_pa" not in mixer


@pytest.mark.parametrize(
    "damage", ["driver", "budget", "request", "expected", "result", "matrix", "metrics"]
)
def test_tampered_feedback_evidence_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers" / "solve_resistance_feedback.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets" / BUDGET.name).write_text("{}")
    elif damage in ("request", "expected", "result"):
        (packet / "3_linear_negative" / (damage + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        value = json.loads(path.read_text())
        if damage == "matrix":
            value["cases"] = []
        else:
            value["cases"][0]["metrics"]["reference_temperatures_k"] = 999
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_solution_is_not_accepted_without_numerical_replay(packet):
    from scripts.solve_hydraulic_graph import sha

    path = packet / "3_linear_negative" / "result.json"
    value = json.loads(path.read_text())
    value["solution"]["solution"]["temperatures_k"]["node0"] += 1
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs"):
        audit(packet)


def test_reference_does_not_use_production_alternation_or_inverse_law(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent reference called production")

    monkeypatch.setattr("scripts.solve_resistance_feedback.effective_input", forbidden)
    monkeypatch.setattr("scripts.solve_resistance_feedback.solve_thermal_graph", forbidden)
    monkeypatch.setattr("scripts.solve_hydraulic_graph.edge_flows", forbidden)
    _, spec, exact = next(cases())
    result = reference(spec)
    assert result["temperatures_k"] == pytest.approx(exact["temperatures_k"], abs=1e-6)
