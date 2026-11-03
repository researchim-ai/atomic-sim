"""Matrix completeness, frozen companions and numerical replay after resealing."""

import json
import shutil

import pytest

from scripts.qualify_reduced_engine import audit, cases, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    first = list(cases())[:1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.qualify_reduced_engine.cases", lambda: iter(first))
        patch.setattr("scripts.qualify_reduced_engine.TIMES", (0.0, 37.5, 38.0))
        folder = tmp_path_factory.mktemp("reduced") / "matrix"
        qualify(folder)
        yield folder


@pytest.fixture
def packet(tmp_path, campaign):
    folder = tmp_path / "matrix"
    shutil.copytree(campaign, folder)
    return folder


def test_audit_recomputes_transients_and_both_handoffs(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 2 and result["bidirectional_handoffs"] == 2
    assert result["adaptive_restart_values_identical"]


def test_declared_matrix_covers_reverse_flow_cycle_and_eight_nodes():
    matrix = dict(cases())
    assert len(matrix) == 6 and max(len(s["volumes_m3"]) for s in matrix.values()) == 8
    assert "pipe_reverse" in matrix and "cycle_cooling" in matrix


@pytest.mark.parametrize("damage", ["driver", "budget", "request", "result", "matrix", "signature"])
def test_tampering_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers/reduced_residual_engine.py").write_text("# altered")
    elif damage == "budget":
        (packet / "budgets/ai05_reduced_engine_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "pipe_positive" / (damage + ".json")).write_text("{}")
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


def test_resealed_handoff_result_is_recomputed(packet):
    path = packet / "pipe_positive/result.json"
    value = json.loads(path.read_text())
    value["legacy_incoming_final"]["state"]["temperatures_k"]["a"] += 1
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)
