"""Root certificates, independent event IVPs and complete cold replay."""

import json
import shutil

import pytest

from scripts.qualify_state_event_radau import audit, cases, negative_cases, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    first = list(cases())[:1]
    negative = list(negative_cases())[:1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.qualify_state_event_radau.cases", lambda: iter(first))
        patch.setattr("scripts.qualify_state_event_radau.negative_cases", lambda: iter(negative))
        patch.setattr("scripts.qualify_state_event_radau.TIMES", (0.0, 3.0, 7.5, 12.0))
        folder = tmp_path_factory.mktemp("state_events") / "matrix"
        qualify(folder)
        yield folder


@pytest.fixture
def packet(tmp_path, campaign):
    folder = tmp_path / "matrix"
    shutil.copytree(campaign, folder)
    return folder


def test_replay_recomputes_roots_modes_and_root_checkpoint(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 2 and result["negative_checks"] == 1
    assert result["restart_values_identical"] and result["root_checkpoint_restore_verified"]


def test_declared_matrix_has_two_event_chain_and_eight_node_graph():
    matrix = dict(cases())
    assert (
        len(matrix) == 9 and max(len(s["model_input"]["volumes_m3"]) for s in matrix.values()) == 8
    )
    assert len(matrix["heat_then_cool"]["guards"]) == 2 and len(list(negative_cases())) == 3


@pytest.mark.parametrize(
    "damage",
    [
        "driver",
        "budget",
        "request",
        "result",
        "matrix",
        "negative_matrix",
        "signature",
        "negative_result",
    ],
)
def test_tampering_is_rejected(packet, damage):
    if damage == "driver":
        (packet / "drivers/state_event_radau.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets/ai05_state_event_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "idle_rising" / (damage + ".json")).write_text("{}")
    elif damage == "negative_result":
        (packet / "negative/coincident_roots/result.json").write_text("{}")
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


def test_resealed_root_certificate_is_numerically_recomputed(packet):
    path = packet / "idle_rising/result.json"
    value = json.loads(path.read_text())
    value["meshes"][-1]["states"][-1]["fired"][0]["time_s"] += 0.01
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)


def test_reference_event_solver_does_not_call_production_root_or_hydraulic_code(monkeypatch):
    from scripts.state_event_references import reference

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent IVP called production")

    monkeypatch.setattr("scripts.state_event_radau._locate", forbidden)
    monkeypatch.setattr("scripts.nonlinear_thermal_graph.solve_hydraulic_graph", forbidden)
    monkeypatch.setattr("scripts.graph_port_adapter.GraphPortAdapter.residual", forbidden)
    _, spec = next(cases())
    result = reference(spec, "DOP853", (0.0, 12.0))
    assert len(result["fired"]) == 1 and abs(result["fired"][0]["time_s"] - 5.0) < 1e-8
