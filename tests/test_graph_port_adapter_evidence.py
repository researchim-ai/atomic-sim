"""Complete numerical evidence replay, including resealed adapter results."""

import json

import pytest

from scripts.qualify_graph_port_adapter import audit, cases, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture
def packet(tmp_path, monkeypatch):
    first = list(cases())[:1]
    monkeypatch.setattr("scripts.qualify_graph_port_adapter.cases", lambda: iter(first))
    monkeypatch.setattr("scripts.qualify_graph_port_adapter.TIMES", (0.0, 0.25, 0.5))
    folder = tmp_path / "matrix"
    qualify(folder)
    return folder


def test_replay_recomputes_residual_ports_and_checkpoint_lifts(packet):
    result = audit(packet)
    assert result["passed"] and result["aligned_state_comparisons"] == 3
    assert result["checkpoint_lifts"] == 2


@pytest.mark.parametrize("damage", ["driver", "budget", "request", "result", "matrix", "signature"])
def test_tampering_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers/graph_port_adapter.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets/ai05_port_adapter_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "pipe_negative" / (damage + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        value = json.loads(path.read_text())
        if damage == "matrix":
            value["cases"] = []
        else:
            value["signature"] = {}
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_port_value_does_not_bypass_recomputation(packet):
    path = packet / "pipe_negative/result.json"
    value = json.loads(path.read_text())
    ports = value["observations"][0]["initialization"]["ports"]
    ports[next(iter(ports))]["energy_flow_w"] += 100.0
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)


def test_reference_rhs_is_independent_of_adapter_and_production_solver(monkeypatch):
    import numpy as np

    from scripts.nonlinear_graph_references import IndependentSystem

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference called production")

    monkeypatch.setattr("scripts.graph_port_adapter.GraphPortAdapter.residual", forbidden)
    monkeypatch.setattr("scripts.nonlinear_thermal_graph.solve_hydraulic_graph", forbidden)
    _, spec = next(cases())
    system = IndependentSystem(spec)
    assert np.all(np.isfinite(system.rhs(0.0, np.zeros(len(system.names) + 1))))
