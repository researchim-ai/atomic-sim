"""Transversal root brackets, mode changes, one-shot memory and transactional restore."""

import json
import math
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.nonlinear_thermal_graph import digest
from scripts.state_event_radau import StateEventError, StateEventGraph
from scripts.state_event_references import cases, guard, idle, negative_cases, wrap


def rise_request():
    return dict(cases())["idle_rising"]


def test_analytic_heater_root_and_post_event_energy():
    model = StateEventGraph(rise_request())
    result = model.advance(20.0)
    root = result["fired"][0]
    assert root["guard_id"] == "rise" and abs(root["time_s"] - 5.0) <= 1e-8
    assert root["bracket_s"][0] <= 5.0 + 1e-12 and root["bracket_s"][1] >= 5.0 - 1e-12
    assert root["bracket_s"][1] - root["bracket_s"][0] <= model.budget["root_time_atol_s"]
    assert root["bracket_values_k"][0] < 0 <= root["bracket_values_k"][1]
    assert abs(root["root_value_k"]) <= model.budget["root_value_atol_k"]
    state = result["model"]["engine"]["state"]
    assert state["temperatures_k"]["a"] == pytest.approx(310.125, abs=1e-8)
    assert state["integrated_external_energy_j"] == pytest.approx(62500.0, abs=0.001)
    assert abs(state["energy_residual_j"]) < 1e-5
    assert result["segments"] and result["evaluations"]["root"] > 0


def test_falling_guard_and_wrong_direction_are_distinct():
    falling = StateEventGraph(dict(cases())["idle_falling"]).advance(12.0)
    assert falling["fired"][0]["direction"] == -1
    assert falling["fired"][0]["rate_bounds_k_s"][1] < 0
    ignored = StateEventGraph(dict(cases())["ignored_direction"]).advance(12.0)
    assert not ignored["fired"] and ignored["pending_guards"] == ["fall_only"]


def test_guard_can_start_on_post_side_and_fire_after_mode_reversal():
    result = StateEventGraph(dict(cases())["heat_then_cool"]).advance(20.0)
    assert [r["guard_id"] for r in result["fired"]] == ["hot", "cool"]
    assert [r["time_s"] for r in result["fired"]] == pytest.approx([5.0, 6.25], abs=3e-8)
    assert result["model"]["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(
        310.0625, abs=1e-8
    )


def test_event_order_depends_on_root_times_not_input_order():
    spec = dict(cases())["ordered_guards"]
    a = StateEventGraph(spec).advance(20.0)
    changed = deepcopy(spec)
    changed["guards"].reverse()
    b = StateEventGraph(changed).advance(20.0)
    assert (
        [r["guard_id"] for r in a["fired"]]
        == [r["guard_id"] for r in b["fired"]]
        == ["earlier", "later"]
    )
    assert [r["time_s"] for r in a["fired"]] == [r["time_s"] for r in b["fired"]]


def test_switch_reinitializes_pressure_without_changing_differential_state():
    result = StateEventGraph(dict(cases())["pressure_reversal"]).advance(10.0)
    trace = result["switches"][0]
    assert (
        trace["left"]["state"]["flows_kg_s"]["pipe"]
        > 0
        > trace["right"]["state"]["flows_kg_s"]["pipe"]
    )
    assert trace["left"]["increments"] == trace["right"]["increments"]
    assert (
        trace["left"]["state"]["integrated_external_energy_j"]
        == trace["right"]["state"]["integrated_external_energy_j"]
    )


def test_root_at_endpoint_is_consumed_once_and_not_replayed():
    spec = wrap(
        idle(125000.0),
        [
            guard(
                "trip",
                "a",
                math.nextafter(310.5, -math.inf),
                1,
                thermal_nodes=dict(a=dict(heat_power_w=0.0)),
            )
        ],
    )
    model = StateEventGraph(spec)
    result = model.advance(2.0)
    assert len(result["fired"]) == 1 and result["fired"][0]["time_s"] <= 2.0
    assert model.advance(2.0)["switches"] == []


@pytest.mark.parametrize("cut", [2.5, 5.25, 5.5, 7.25])
def test_restart_before_or_after_event_matches_in_memory_continuation(cut):
    spec = rise_request()
    a = StateEventGraph(spec)
    a.advance(cut)
    b = StateEventGraph(spec)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(10.0) == b.advance(10.0)


def test_future_guard_definitions_are_bound_by_checkpoint():
    spec = rise_request()
    a = StateEventGraph(spec)
    a.advance(1.0)
    cp = a.checkpoint()
    changed = deepcopy(spec)
    changed["guards"][0]["threshold_k"] += 0.1
    with pytest.raises(NetworkError):
        StateEventGraph(changed).restore(cp)
    with pytest.raises(NetworkError):
        StateEventGraph(spec, max_macro_step_s=1.0).restore(cp)


@pytest.mark.parametrize("name", [name for name, _, _ in negative_cases()])
def test_real_invalid_or_ambiguous_events_roll_back_complete_call(name):
    spec = next(s for n, s, _ in negative_cases() if n == name)
    model = StateEventGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(20.0)
    assert model.checkpoint() == before


def test_previously_committed_mode_survives_later_event_failure():
    spec = wrap(
        idle(),
        [
            guard("first", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=12500.0))),
            guard("bad", "a", 310.25, 1, hydraulic_nodes=dict(a=dict(pressure_pa=1e9))),
        ],
    )
    model = StateEventGraph(spec)
    model.advance(6.0)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(10.0)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "key",
    [
        "maximum_attempts_per_advance",
        "maximum_rhs_evaluations_per_advance",
        "maximum_root_evaluations_per_advance",
        "maximum_root_iterations",
        "maximum_guard_box_refinements",
    ],
)
def test_resource_budget_failure_is_atomic(key):
    model = StateEventGraph(rise_request())
    model.budget[key] = 0
    before = model.checkpoint()
    with pytest.raises(StateEventError):
        model.advance(10.0)
    assert model.checkpoint() == before


def test_unresolved_transversality_is_refused_not_treated_as_no_event(monkeypatch):
    model = StateEventGraph(rise_request())
    before = model.checkpoint()
    kernel = model._runtime._engine.system.adapter._kernel
    # Class-level injection survives detached copies; a box that straddles zero
    # rate at every refinement cannot establish the required crossing proof.
    original = type(kernel)._box

    def unresolved(self, t, hyd, effective):
        lo, hi, rl, rh = original(self, t, hyd, effective)
        rl[:] = -1.0
        rh[:] = 1.0
        return lo, hi, rl, rh

    with monkeypatch.context() as patch:
        patch.setattr(type(kernel), "_box", unresolved)
        with pytest.raises(StateEventError, match="monotone"):
            model.advance(10.0)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "source",
        "input",
        "schema",
        "model",
        "id",
        "time",
        "bracket",
        "bracket_value",
        "root_value",
        "rates",
        "threshold",
        "direction",
        "command",
        "pending",
        "observation",
        "native",
    ],
)
def test_resealed_checkpoint_corruption_cannot_mutate_model(damage):
    model = StateEventGraph(rise_request())
    model.advance(10.0)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    state = p["state"]
    root = state["fired"][0]
    if damage == "digest":
        cp["sha256"] = "0" * 64
    elif damage == "source":
        p["signature"] = {}
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    elif damage == "schema":
        p["schema_version"] = True
    elif damage == "model":
        p["model"] = "other"
    elif damage == "id":
        root["guard_id"] = "unknown"
    elif damage == "time":
        root["time_s"] = -1.0
    elif damage == "bracket":
        root["bracket_s"][0] -= 1.0
    elif damage == "bracket_value":
        root["bracket_values_k"][0] = 1.0
    elif damage == "root_value":
        root["root_value_k"] = 1.0
    elif damage == "rates":
        root["rate_bounds_k_s"][0] = -1.0
    elif damage == "threshold":
        root["threshold_k"] += 1.0
    elif damage == "direction":
        root["direction"] = True
    elif damage == "command":
        root["changes_sha256"] = "0" * 64
    elif damage == "pending":
        state["pending_guards"] = ["rise"]
    elif damage == "observation":
        state["model"]["engine"]["state"]["temperatures_k"]["a"] += 1.0
    else:
        p["runtime_checkpoint"]["payload"]["state"]["engine"]["increments"][0] += 0.1
    if damage != "digest":
        cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "damage",
    [
        "format",
        "schema",
        "node",
        "id",
        "threshold",
        "direction",
        "bool_direction",
        "initial_contact",
        "duplicate",
        "empty_command",
        "topology",
        "too_many",
    ],
)
def test_invalid_guard_contract_rejected(damage):
    request = rise_request()
    g = request["guards"][0]
    if damage == "format":
        request["format"] = "other"
    elif damage == "schema":
        request["schema_version"] = True
    elif damage == "node":
        g["node"] = "unknown"
    elif damage == "id":
        g["id"] = "../bad"
    elif damage == "threshold":
        g["threshold_k"] = 1e9
    elif damage == "direction":
        g["direction"] = 0
    elif damage == "bool_direction":
        g["direction"] = True
    elif damage == "initial_contact":
        g["threshold_k"] = 310.0
    elif damage == "duplicate":
        request["guards"].append(deepcopy(g))
    elif damage == "empty_command":
        g["changes"] = {}
    elif damage == "topology":
        g["changes"] = dict(hydraulic_edges=dict(pipe=dict(linear_pa_s_kg=1.0)))
    else:
        request["guards"] = [deepcopy(g) for _ in range(17)]
    with pytest.raises(NetworkError):
        StateEventGraph(request)


@pytest.mark.parametrize("duration", [0.0, -1.0, True, float("nan"), float("inf"), 10001.0])
def test_bad_duration_never_commits_event(duration):
    model = StateEventGraph(rise_request())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(duration)
    assert model.checkpoint() == before


def test_no_guard_and_no_crossing_cases_keep_history_empty():
    spec = rise_request()
    spec["guards"] = []
    model = StateEventGraph(spec)
    assert not model.advance(10.0)["fired"]
    result = StateEventGraph(dict(cases())["no_crossing"]).advance(10.0)
    assert not result["fired"] and result["pending_guards"] == ["never"]


def test_legacy_time_integrator_is_not_used_for_event_detection(monkeypatch):
    model = StateEventGraph(rise_request())

    def forbidden(*args, **kwargs):
        raise AssertionError("Event detection called legacy DP")

    monkeypatch.setattr("scripts.nonlinear_thermal_graph.NonlinearThermalGraph.advance", forbidden)
    assert len(model.advance(10.0)["fired"]) == 1


@pytest.mark.parametrize("operation", ["advance", "restore"])
def test_final_observation_failure_cannot_commit(operation, monkeypatch):
    model = StateEventGraph(rise_request())
    before = model.checkpoint()

    def fail(*args, **kwargs):
        raise NetworkError("injected final observation failure")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_snapshot", fail)
        with pytest.raises(NetworkError):
            if operation == "advance":
                model.advance(10.0)
            else:
                model.restore(before)
    assert model.checkpoint() == before


def test_exports_are_detached_and_source_drift_is_blocked(monkeypatch):
    request = rise_request()
    original = deepcopy(request)
    model = StateEventGraph(request)
    report = model.advance(10.0)
    before = model.checkpoint()
    report["fired"].clear()
    report["model"]["entry_temperatures_k"]["a"] = 0.0
    assert model.checkpoint() == before and request == original
    monkeypatch.setattr("scripts.state_event_radau.signature", lambda: {})
    with pytest.raises(StateEventError, match="changed"):
        model.advance(1.0)
    with pytest.raises(StateEventError, match="changed"):
        model.restore(before)
    assert model.checkpoint() == before


def test_unresolved_endpoint_contact_is_refused_instead_of_silently_skipped():
    spec = wrap(
        idle(125000.0), [guard("trip", "a", 310.5, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))]
    )
    model = StateEventGraph(spec)
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(2.0)
    assert error.value.report["criterion"] == "endpoint_contact"
    assert model.checkpoint() == before
    assert len(model.advance(2.1)["fired"]) == 1


def test_near_coincident_guards_are_not_epsilon_coalesced():
    request = rise_request()
    request["guards"].append(
        guard("near", "a", 310.125 + 1e-9, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))
    )
    model = StateEventGraph(request)
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(10.0)
    assert error.value.report["criterion"] in ("ambiguous_roots", "zero_time_cascade")
    assert model.checkpoint() == before


def test_stop_at_event_returns_right_state_and_root_checkpoint_resumes_exactly():
    request = dict(cases())["heat_then_cool"]
    model = StateEventGraph(request)
    result = model.advance(20.0, stop_at_event=True)
    assert result["stopped_at_event"] and result["time_s"] < result["requested_time_s"] == 20.0
    assert len(result["fired"]) == 1 and result["time_s"] == result["fired"][0]["time_s"]
    restored = StateEventGraph(request)
    assert restored.restore(model.checkpoint()) == model.snapshot()
    a = model.advance(20.0)
    b = restored.advance(20.0)
    assert a == b and [r["guard_id"] for r in a["fired"]] == ["hot", "cool"]
    assert len(a["switches"]) == 1


def test_stop_at_event_without_crossing_reaches_requested_time():
    result = StateEventGraph(dict(cases())["no_crossing"]).advance(10.0, stop_at_event=True)
    assert not result["stopped_at_event"] and result["time_s"] == result["requested_time_s"] == 10.0
    with pytest.raises(NetworkError):
        StateEventGraph(rise_request()).advance(10.0, stop_at_event=1)


def test_dense_endpoint_root_uses_one_consistent_trajectory_after_split():
    model = StateEventGraph(rise_request())
    model.advance(3.0)
    result = model.advance(4.5)
    assert len(result["fired"]) == 1 and abs(result["fired"][0]["time_s"] - 5.0) <= 1e-8
    assert result["model"]["engine"]["state"]["integrated_external_energy_j"] == pytest.approx(
        62500.0, abs=0.001
    )


def test_rhs_budget_is_shared_across_the_event_and_mode_change():
    request = rise_request()
    probe = StateEventGraph(request)
    first = probe.advance(20.0, stop_at_event=True)
    model = StateEventGraph(request)
    model.budget["maximum_rhs_evaluations_per_advance"] = first["evaluations"]["rhs"]
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(20.0)
    assert error.value.report["criterion"] == "rhs_budget"
    assert model.checkpoint() == before
