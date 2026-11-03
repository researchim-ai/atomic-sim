"""Cold replay of cyclic mode histories and minimum-residence refusals."""

import json
import shutil

import pytest

from scripts.qualify_hysteretic_radau import audit, cases, negative_cases, qualify
from scripts.solve_hydraulic_graph import sha


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    first = list(cases())[:1]
    negative = list(negative_cases())[:1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.qualify_hysteretic_radau.cases", lambda: iter(first))
        patch.setattr("scripts.qualify_hysteretic_radau.negative_cases", lambda: iter(negative))
        patch.setattr("scripts.qualify_hysteretic_radau.TIMES", (0.0, 2.0, 11.0, 40.0))
        folder = tmp_path_factory.mktemp("hysteresis") / "matrix"
        qualify(folder)
        yield folder


@pytest.fixture
def packet(tmp_path, campaign):
    folder = tmp_path / "matrix"
    shutil.copytree(campaign, folder)
    return folder


def test_audit_recomputes_rearming_mode_path_and_residence(packet):
    result = audit(packet)
    assert result["passed"] and result["production_mesh_trajectories"] == 3
    assert result["independent_ivps"] == 2 and result["negative_checks"] == 1
    assert result["restart_values_identical"] and result["transition_checkpoint_restore_verified"]


def test_declared_matrix_has_recurrent_flow_branching_and_eight_node_cases():
    matrix = dict(cases())
    assert (
        len(matrix) == 9 and max(len(s["model_input"]["volumes_m3"]) for s in matrix.values()) == 8
    )
    assert "pressure_relay" in matrix and "branch_selection" in matrix
    assert len(list(negative_cases())) == 4


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
        (packet / "drivers/hysteretic_radau.py").write_text("# changed")
    elif damage == "budget":
        (packet / "budgets/ai05_hysteresis_budget_v1.json").write_text("{}")
    elif damage in ("request", "result"):
        (packet / "idle_relay" / (damage + ".json")).write_text("{}")
    elif damage == "negative_result":
        (packet / "negative/too_short_residence/result.json").write_text("{}")
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


def test_resealed_transition_history_is_numerically_recomputed(packet):
    path = packet / "idle_relay/result.json"
    value = json.loads(path.read_text())
    value["meshes"][-1]["states"][-1]["history"][0]["target_mode"] = "heat"
    path.write_text(json.dumps(value))
    report = packet / "qualification.json"
    value = json.loads(report.read_text())
    value["cases"][0]["files"]["result.json"] = sha(path)
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="replay differs"):
        audit(packet)


def test_independent_reference_does_not_call_production_controller_or_residual(monkeypatch):
    from scripts.hysteresis_references import reference

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference called production controller")

    monkeypatch.setattr("scripts.hysteretic_radau.HystereticRadauGraph.advance", forbidden)
    monkeypatch.setattr("scripts.state_event_radau._locate", forbidden)
    monkeypatch.setattr("scripts.graph_port_adapter.GraphPortAdapter.residual", forbidden)
    _, spec = next(cases())
    result = reference(spec, "DOP853", (0.0, 20.0))
    assert len(result["history"]) == 4 and result["states"][-1]["mode"] == "heat"
