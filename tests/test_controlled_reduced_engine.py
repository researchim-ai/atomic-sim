"""Radau command cuts, warm-state reinitialization, global frame and atomicity."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.controlled_reduced_engine import OnlineReducedGraph, ScheduledReducedGraph
from scripts.nonlinear_graph_references import bridge_spec, pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.scheduled_graph_references import event, heater_spec, negative_cases, wrap


def command(power=1000.0):
    return dict(thermal_nodes=dict(a=dict(heat_power_w=power)))


def test_piecewise_heating_matches_analytic_global_energy_and_endpoint_commands():
    model = ScheduledReducedGraph(heater_spec())
    result = model.advance(120.0)
    state = result["engine"]["state"]
    assert state["temperatures_k"] == pytest.approx(dict(a=310.1, b=319.99), abs=1e-10)
    assert state["integrated_external_energy_j"] == pytest.approx(45000.0, abs=1e-6)
    assert abs(state["energy_residual_j"]) < 1e-6
    assert result["active_event_count"] == 5 and len(result["switches"]) == 4
    assert model._active["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] == 1000.0
    assert model.advance(1.0)["switches"] == []


def test_pressure_reversal_preserves_frame_and_has_no_energy_impulse():
    spec = wrap(pipe_spec(), [event(1.0, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0)))])
    result = ScheduledReducedGraph(spec).advance(2.0)
    trace = result["switches"][0]
    assert (
        trace["left"]["state"]["flows_kg_s"]["pipe"]
        > 0
        > trace["right"]["state"]["flows_kg_s"]["pipe"]
    )
    assert trace["left"]["increments"] == trace["right"]["increments"]
    for key in (
        "temperatures_k",
        "temperature_increments_k",
        "integrated_external_energy_j",
        "energy_residual_j",
    ):
        assert trace["left"]["state"][key] == trace["right"]["state"][key]


def test_warm_reinitialization_and_restart_do_not_validate_original_cold_guess():
    spec = bridge_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["max_pressure_pa"] = 196500.0
    request = wrap(spec, [event(60.0, hydraulic_nodes=dict(supply=dict(pressure_pa=201000.0)))])
    model = ScheduledReducedGraph(request)
    model.advance(60.0)
    cold = deepcopy(spec)
    cold["steady_input"]["hydraulic"]["nodes"][0]["pressure_pa"] = 201000.0
    with pytest.raises(NetworkError):
        NonlinearThermalGraph(cold)
    restored = ScheduledReducedGraph(request)
    restored.restore(json.loads(json.dumps(model.checkpoint())))
    assert restored.advance(2.0) == model.advance(2.0)


def test_combined_inlet_command_is_validated_atomically():
    spec = bridge_spec()
    spec["steady_input"]["thermal"]["nodes"]["a"].pop("supply_temperature_k")
    model = OnlineReducedGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(dict(hydraulic_nodes=dict(a=dict(injection_kg_s=0.5))), 1.0, expected_revision=0)
    assert model.checkpoint() == before
    result = model.step(
        dict(
            hydraulic_nodes=dict(a=dict(injection_kg_s=0.5)),
            thermal_nodes=dict(a=dict(supply_temperature_k=350.0)),
        ),
        1.0,
        expected_revision=0,
    )
    assert result["observation"]["revision"] == 1


def test_close_float_commands_are_not_merged():
    second = float(np.nextafter(1.0, math.inf))
    model = ScheduledReducedGraph(
        wrap(pipe_spec(), [event(1.0, **command()), event(second, **command(2000.0))])
    )
    result = model.advance(2.0)
    assert [s["receipt"]["time_s"] for s in result["switches"]] == [1.0, second]


def test_rounding_alignment_preserves_absolute_event_timestamp():
    timestamp = float(np.nextafter(1.0, math.inf))
    model = ScheduledReducedGraph(wrap(pipe_spec(), [event(timestamp, **command())]))
    model.advance(float(np.finfo(float).eps / 2))
    result = model.advance(2.0)
    assert result["switches"][0]["receipt"]["time_s"] == timestamp


@pytest.mark.parametrize("name", [c[0] for c in negative_cases()])
def test_real_negative_scenarios_roll_back_whole_prefix(name):
    spec = next(s for n, s, _ in negative_cases() if n == name)
    model = ScheduledReducedGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(10.0)
    assert model.checkpoint() == before


def test_failure_after_previously_committed_command_preserves_it():
    spec = wrap(pipe_spec(), [event(1.0, **command()), event(3.0, **command(1e12))])
    model = ScheduledReducedGraph(spec)
    model.advance(2.0)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(2.0)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "limit",
    [
        "maximum_segments_per_advance",
        "maximum_rhs_evaluations_per_advance",
        "maximum_attempts_per_advance",
    ],
)
def test_resource_caps_apply_across_event_segments(limit):
    model = ScheduledReducedGraph(
        wrap(pipe_spec(), [event(1.0, **command()), event(2.0, **command(2000.0))])
    )
    model.budget[limit] = 1
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(4.0)
    assert model.checkpoint() == before


def test_restart_at_command_and_unaligned_split_is_exact():
    spec = wrap(
        pipe_spec(),
        [event(1.0, **command()), event(3.0, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0)))],
    )
    for cut in (1.0, 1.375):
        model = ScheduledReducedGraph(spec)
        model.advance(cut)
        copy = ScheduledReducedGraph(spec)
        copy.restore(json.loads(json.dumps(model.checkpoint())))
        assert copy.advance(5.0) == model.advance(5.0)


@pytest.mark.parametrize("with_event", [False, True])
def test_integer_temperature_input_checkpoints_roundtrip(with_event):
    spec = pipe_spec()
    spec["initial_temperatures_k"] = dict(a=310, b=320)
    request = wrap(spec, [event(1.0, **command())] if with_event else [])
    a = ScheduledReducedGraph(request)
    a.advance(2.0)
    b = ScheduledReducedGraph(request)
    assert b.restore(a.checkpoint()) == a.snapshot()


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "cursor",
        "bool_cursor",
        "active",
        "receipts",
        "entry_time",
        "entry_map",
        "clock",
        "temperature",
        "energy",
        "source",
        "input",
        "frame",
    ],
)
def test_resealed_controlled_checkpoint_corruption_is_rejected(damage):
    spec = wrap(pipe_spec(), [event(1.0, **command())])
    model = ScheduledReducedGraph(spec)
    model.advance(2.0)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    state = p["state"]
    if damage == "digest":
        cp["sha256"] = "0" * 64
    elif damage == "cursor":
        state["active_event_count"] = 0
    elif damage == "bool_cursor":
        state["active_event_count"] = True
    elif damage == "active":
        state["active_model_sha256"] = "0" * 64
    elif damage == "receipts":
        state["event_receipts"] = []
    elif damage == "entry_time":
        state["entry_time_s"] = 0.5
    elif damage == "entry_map":
        state["entry_temperatures_k"]["a"] += 1.0
    elif damage == "clock":
        state["time_s"] = -1.0
    elif damage == "temperature":
        state["engine"]["state"]["temperatures_k"]["a"] += 1.0
    elif damage == "energy":
        state["engine"]["increments"][-1] += 1.0
    elif damage == "source":
        p["signature"] = {}
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    else:
        p["engine_checkpoint"]["payload"]["state"]["increments"][0] += 0.1
        p["engine_checkpoint"]["sha256"] = digest(p["engine_checkpoint"]["payload"])
    if damage != "digest":
        cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


def test_future_timeline_change_and_macro_policy_cannot_reuse_checkpoint():
    spec = wrap(pipe_spec(), [event(10.0, **command())])
    a = ScheduledReducedGraph(spec)
    a.advance(1.0)
    cp = a.checkpoint()
    changed = deepcopy(spec)
    changed["timeline"][0]["time_s"] = 11.0
    with pytest.raises(NetworkError):
        ScheduledReducedGraph(changed).restore(cp)
    with pytest.raises(NetworkError):
        ScheduledReducedGraph(spec, max_macro_step_s=1.0).restore(cp)


@pytest.mark.parametrize("revision", [-1, 1, True, 0.0, None, "0"])
def test_stale_online_revision_is_atomic(revision):
    model = OnlineReducedGraph(pipe_spec())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(), 1.0, expected_revision=revision)
    assert model.checkpoint() == before


def test_online_late_failure_can_be_corrected_with_same_revision():
    model = OnlineReducedGraph(pipe_spec())
    model.step(command(), 1.0, expected_revision=0)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(1e12), 2.0, expected_revision=1)
    assert model.checkpoint() == before
    assert model.step(command(0.0), 1.0, expected_revision=1)["observation"]["revision"] == 2


def test_online_restart_preserves_frame_history_and_future_decisions():
    a = OnlineReducedGraph(pipe_spec())
    a.step(command(), 1.375, expected_revision=0)
    b = OnlineReducedGraph(pipe_spec())
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.observation()
    assert a.step(command(2000.0), 2.0, expected_revision=1) == b.step(
        command(2000.0), 2.0, expected_revision=1
    )
    with pytest.raises(AttributeError):
        a.revision = 0


@pytest.mark.parametrize(
    "damage", ["revision", "bool_revision", "history", "source", "input", "future"]
)
def test_online_checkpoint_corruption_is_rejected(damage):
    model = OnlineReducedGraph(pipe_spec())
    model.step(command(), 1.0, expected_revision=0)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if damage == "revision":
        p["revision"] = 0
    elif damage == "bool_revision":
        p["revision"] = True
    elif damage == "history":
        p["history"][0]["changes"] = command(2000.0)
    elif damage == "source":
        p["signature"] = {}
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    else:
        p["history"].append(event(2.0, **command()))
        p["revision"] = 2
    cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "limit", ["maximum_events", "maximum_episode_s", "maximum_rhs_evaluations_per_advance"]
)
def test_online_resource_limits_roll_back_step(limit):
    model = OnlineReducedGraph(pipe_spec())
    model.step(command(), 1.0, expected_revision=0)
    model.budget[limit] = 1
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(), 2.0, expected_revision=1)
    assert model.checkpoint() == before


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 10001.0])
def test_bad_duration_never_commits_online_command(duration):
    model = OnlineReducedGraph(pipe_spec())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(), duration, expected_revision=0)
    assert model.checkpoint() == before


def test_continuation_and_commands_never_use_legacy_time_integration(monkeypatch):
    model = OnlineReducedGraph(pipe_spec())

    def forbidden(*args, **kwargs):
        raise AssertionError("Called legacy DP advance")

    monkeypatch.setattr("scripts.nonlinear_thermal_graph.NonlinearThermalGraph.advance", forbidden)
    model.step(command(), 2.0, expected_revision=0)
    model.step(command(2000.0), 2.0, expected_revision=1)
    assert model.observation()["model"]["time_s"] == 4.0


@pytest.mark.parametrize("operation", ["advance", "restore", "online_step", "online_restore"])
def test_final_observation_failure_cannot_commit(operation, monkeypatch):
    if operation.startswith("online"):
        model = OnlineReducedGraph(pipe_spec())
        before = model.checkpoint()
        actual = model._sources
        calls = 0

        def fail_second():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise NetworkError("injected final source failure")
            actual()

        with monkeypatch.context() as patch:
            patch.setattr(model, "_sources", fail_second)
            with pytest.raises(NetworkError):
                if operation == "online_step":
                    model.step(command(), 1.0, expected_revision=0)
                else:
                    model.restore(before)
    else:
        model = ScheduledReducedGraph(heater_spec())
        before = model.checkpoint()

        def fail(*args, **kwargs):
            raise NetworkError("injected observation failure")

        with monkeypatch.context() as patch:
            patch.setattr(model, "_snapshot", fail)
            with pytest.raises(NetworkError):
                if operation == "advance":
                    model.advance(1.0)
                else:
                    model.restore(before)
    assert model.checkpoint() == before


def test_detached_exports_inputs_and_source_drift(monkeypatch):
    spec = pipe_spec()
    original = deepcopy(spec)
    model = OnlineReducedGraph(spec)
    changes = command()
    result = model.step(changes, 1.0, expected_revision=0)
    before = model.checkpoint()
    result["observation"]["model"]["entry_temperatures_k"]["a"] = 0
    changes["thermal_nodes"]["a"]["heat_power_w"] = 1e12
    assert model.checkpoint() == before and spec == original
    monkeypatch.setattr("scripts.controlled_reduced_engine.signature", lambda: {})
    with pytest.raises(NetworkError, match="drift"):
        model.step(command(), 1.0, expected_revision=1)
    with pytest.raises(NetworkError, match="drift"):
        model.restore(before)
    assert model.checkpoint() == before


def test_rhs_budget_is_shared_after_a_successful_first_event_segment():
    request = wrap(pipe_spec(), [event(1.0, **command())])
    probe = ScheduledReducedGraph(request)
    first = probe.advance(1.0)
    model = ScheduledReducedGraph(request)
    model.budget["maximum_rhs_evaluations_per_advance"] = first["rhs_evaluations"]
    before = model.checkpoint()
    with pytest.raises(NetworkError, match="aggregate"):
        model.advance(2.0)
    assert model.checkpoint() == before


@pytest.mark.parametrize("mode", ["empty_advanced", "late_first", "unheld"])
def test_coherent_scheduled_checkpoint_cannot_bypass_online_hold_contract(mode):
    spec = pipe_spec()
    history = (
        []
        if mode == "empty_advanced"
        else [event(1.0 if mode == "late_first" else 0.0, **command())]
    )
    scheduled = ScheduledReducedGraph(wrap(spec, history))
    if mode != "unheld":
        scheduled.advance(2.0)
    online = OnlineReducedGraph(spec)
    before = online.checkpoint()
    cp = deepcopy(before)
    cp["payload"].update(
        revision=len(history), history=history, scheduled_checkpoint=scheduled.checkpoint()
    )
    cp["sha256"] = digest(cp["payload"])
    with pytest.raises(NetworkError, match="future/unheld/missing"):
        online.restore(cp)
    assert online.checkpoint() == before


@pytest.mark.parametrize(
    "damage", ["format", "schema", "unknown", "duplicate", "empty_batch", "topology"]
)
def test_invalid_controlled_wire_contract_rejected(damage):
    request = wrap(pipe_spec(), [event(1.0, **command())])
    if damage == "format":
        request["format"] = "other"
    elif damage == "schema":
        request["schema_version"] = True
    elif damage == "unknown":
        request["extra"] = 1
    elif damage == "duplicate":
        request["timeline"].append(deepcopy(request["timeline"][0]))
    elif damage == "empty_batch":
        request["timeline"][0]["changes"] = {}
    else:
        request["timeline"][0]["changes"] = dict(
            hydraulic_edges=dict(pipe=dict(linear_pa_s_kg=1.0))
        )
    with pytest.raises(NetworkError):
        ScheduledReducedGraph(request)
