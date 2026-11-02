"""Frozen campaigns, independent IVPs, tampering and declared coverage."""

import json

import pytest

from scripts.qualify_transient_graph import audit, evaluate, qualify
from scripts.transient_graph_references import cases, reference


@pytest.fixture
def packet(tmp_path, monkeypatch):
    # Small fixture for integrity tests; full22-case campaign is run separately.
    first = list(cases())[:1]
    monkeypatch.setattr("scripts.qualify_transient_graph.cases", lambda: iter(first))
    folder = tmp_path / "campaign"
    qualify(folder)
    return folder


def test_campaign_recomputes_independent_ivps_and_restart(packet):
    result = audit(packet)
    assert result["passed"] and result["cases"] == 1 and result["independent_ivps"] == 2


def test_declared_matrix_includes_closed_adiabatic_and_large_graphs():
    matrix = list(cases())
    assert len(matrix) == 22
    assert max(len(s["volumes_m3"]) for _, s in matrix) == 128
    assert matrix[-1][0] == "adiabatic_cycle"


@pytest.mark.parametrize("damage", ["driver", "budget", "request", "result", "matrix", "metrics"])
def test_invalid_evidence_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers" / "transient_thermal_graph.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets" / "ai05_transient_graph_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "3_linear" / (damage + ".json")).write_text("{}")
    else:
        path = packet / "qualification.json"
        d = json.loads(path.read_text())
        if damage == "matrix":
            d["cases"] = []
        else:
            d["cases"][0]["metrics"]["reference_energy_error_j"] = 999
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        audit(packet)


def test_resealed_trajectory_is_not_accepted(packet):
    from scripts.solve_hydraulic_graph import sha

    path = packet / "3_linear" / "result.json"
    d = json.loads(path.read_text())
    d["states"][-1]["temperatures_k"]["node0"] += 1
    path.write_text(json.dumps(d))
    manifest = packet / "qualification.json"
    m = json.loads(manifest.read_text())
    m["cases"][0]["files"]["result.json"] = sha(path)
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="differs"):
        audit(packet)


def test_reference_uses_no_production_exponential_or_step(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Independent IVP used production")

    monkeypatch.setattr("scripts.transient_thermal_graph.expm", forbidden)
    monkeypatch.setattr("scripts.transient_thermal_graph.ThermalGraph.advance", forbidden)
    _, spec = next(cases())
    result = reference(spec, "DOP853")
    assert result["times_s"][-1] == 120


def test_large_near_zero_case_has_independent_energy_and_restart_check():
    _, spec = next(c for c in cases() if c[0] == "128_near_zero")
    result = evaluate(spec)
    assert result["metrics"]["maximum_energy_residual_j"] <= 1e-4
    assert result["metrics"]["restart_temperature_error_k"] <= 1e-9
