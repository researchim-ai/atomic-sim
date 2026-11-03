"""Calendar semantics, reinitialization, global energy and full-request rollback."""

import json
import math
import sys
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.nonlinear_graph_references import bridge_spec, pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.scheduled_graph_references import event, heater_spec, negative_cases, wrap
from scripts.scheduled_thermal_graph import ScheduledGraphError, ScheduledThermalGraph, main


def test_piecewise_heating_matches_analytic_energy_and_temperature():
    spec = heater_spec()
    model = ScheduledThermalGraph(spec)
    assert model.snapshot()["active_event_count"] == 1
    result = model.advance(120)
    capacity = 500000.0
    assert result["state"]["temperatures_k"] == pytest.approx(
        dict(
            a=310 + (1000 * 10 + 2000 * 20) / capacity, b=320 + (-1000 * 20 + 500 * 30) / capacity
        ),
        abs=1e-9,
    )
    assert result["state"]["integrated_external_energy_j"] == pytest.approx(45000.0, abs=1e-5)
    assert result["active_event_count"] == 5 and len(result["switches"]) == 4
    assert result["state"]["temperature_increments_k"]["b"] == pytest.approx(-0.01, abs=1e-9)
    assert model._active["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] == 1000


def test_endpoint_is_right_continuous_and_never_reapplies_event():
    model = ScheduledThermalGraph(
        wrap(pipe_spec(), [event(1, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0)))])
    )
    result = model.advance(1)
    assert (
        result["state"]["pressures_pa"]["a"] == 198000 and result["state"]["flows_kg_s"]["pipe"] < 0
    )
    assert len(result["switches"]) == 1
    following = model.advance(1)
    assert following["switches"] == [] and following["active_event_count"] == 1


def test_simultaneous_inlet_and_temperature_are_validated_as_one_batch():
    model = bridge_spec()
    model["steady_input"]["thermal"]["nodes"]["a"].pop("supply_temperature_k")
    spec = wrap(
        model,
        [
            event(
                1,
                hydraulic_nodes=dict(a=dict(injection_kg_s=0.5)),
                thermal_nodes=dict(a=dict(supply_temperature_k=350.0)),
            )
        ],
    )
    result = ScheduledThermalGraph(spec).advance(2)
    assert result["active_event_count"] == 1 and result["switches"][0]["right"]["time_s"] == 1


def test_reinitialization_preserves_stocks_energy_and_resets_proposal():
    model = ScheduledThermalGraph(
        wrap(pipe_spec(), [event(1, hydraulic_nodes=dict(a=dict(pressure_pa=201000.0)))])
    )
    result = model.advance(2)
    trace = result["switches"][0]
    left, right = trace["left"], trace["right"]
    for key in (
        "temperatures_k",
        "temperature_increments_k",
        "integrated_external_energy_j",
        "energy_integral_scaled_k",
        "stored_energy_j",
    ):
        assert left[key] == right[key]
    assert left["flows_kg_s"] != right["flows_kg_s"]
    assert right["step_proposal_s"] == model.max_step


def test_reinitialization_uses_warm_state_not_original_temperature_guess():
    model = bridge_spec()
    model["steady_input"]["hydraulic"]["nodes"][1]["max_pressure_pa"] = 196500.0
    spec = wrap(model, [event(60, hydraulic_nodes=dict(supply=dict(pressure_pa=201000.0)))])
    result = ScheduledThermalGraph(spec).advance(60)
    assert (
        result["state"]["pressures_pa"]["a"] < 196500
        and result["state"]["temperatures_k"]["a"] > 300
    )
    hypothetical = deepcopy(model)
    hypothetical["steady_input"]["hydraulic"]["nodes"][0]["pressure_pa"] = 201000
    with pytest.raises(NetworkError):
        NonlinearThermalGraph(hypothetical)


def test_distinct_adjacent_float_events_are_not_epsilon_coalesced():
    second = float(np.nextafter(1.0, math.inf))
    spec = wrap(
        pipe_spec(),
        [
            event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
            event(second, thermal_nodes=dict(a=dict(heat_power_w=2000.0))),
        ],
    )
    result = ScheduledThermalGraph(spec).advance(2)
    assert [t["receipt"]["time_s"] for t in result["switches"]] == [1.0, second]
    assert result["active_event_count"] == 2


def test_absolute_clock_alignment_does_not_move_event_order():
    timestamp = float(np.nextafter(1.0, math.inf))
    model = ScheduledThermalGraph(
        wrap(pipe_spec(), [event(timestamp, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    )
    model.advance(np.finfo(float).eps / 2)
    result = model.advance(2)
    assert result["switches"][0]["left"]["time_s"] == timestamp
    assert result["switches"][0]["right"]["time_s"] == timestamp
    assert all(abs(s["clock_alignment_s"]) <= math.ulp(s["end_s"]) for s in result["segments"])


def test_empty_timeline_matches_unscheduled_model():
    base = pipe_spec()
    expected = NonlinearThermalGraph(base).advance(10)
    result = ScheduledThermalGraph(wrap(base, [])).advance(10)
    for key in (
        "temperatures_k",
        "temperature_increments_k",
        "integrated_external_energy_j",
        "pressures_pa",
        "flows_kg_s",
    ):
        assert result["state"][key] == expected[key]


@pytest.mark.parametrize(
    "name", ["invalid_right_state", "late_domain_exit", "missing_new_inlet_temperature"]
)
def test_failure_after_prefix_work_rolls_back_entire_request(name):
    _, spec, criterion = next(c for c in negative_cases() if c[0] == name)
    model = ScheduledThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(ScheduledGraphError) as error:
        model.advance(10)
    assert error.value.report["criterion"] == criterion
    assert model.checkpoint() == before


def test_failure_after_previously_committed_event_preserves_that_event():
    _, spec, _ = next(c for c in negative_cases() if c[0] == "late_domain_exit")
    model = ScheduledThermalGraph(spec)
    model.advance(5)
    before = model.checkpoint()
    with pytest.raises(ScheduledGraphError):
        model.advance(5)
    assert model.checkpoint() == before and model.snapshot()["active_event_count"] == 2


@pytest.mark.parametrize(
    "budget_key,criterion",
    [
        ("maximum_attempts_per_advance", "attempt_budget"),
        ("maximum_segments_per_advance", "segment_budget"),
    ],
)
def test_resource_budget_applies_to_whole_request_not_each_event(budget_key, criterion):
    spec = wrap(
        pipe_spec(),
        [
            event(0.1, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
            event(0.2, thermal_nodes=dict(a=dict(heat_power_w=2000.0))),
        ],
    )
    model = ScheduledThermalGraph(spec)
    before = model.checkpoint()
    model.budget[budget_key] = 1
    with pytest.raises(ScheduledGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == criterion and model.checkpoint() == before


@pytest.mark.parametrize("operation", ["advance", "restore"])
def test_final_observation_failure_cannot_commit_wrapper_state(operation, monkeypatch):
    spec = wrap(pipe_spec(), [event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    model = ScheduledThermalGraph(spec)
    before = model.checkpoint()
    original = model._snapshot
    other = ScheduledThermalGraph(spec)
    other.advance(2)
    checkpoint = other.checkpoint()

    def fail(kernel, cursor, active, receipts):
        if kernel._time > model._kernel._time:
            raise NetworkError("Injected final observation failure")
        return original(kernel, cursor, active, receipts)

    monkeypatch.setattr(model, "_snapshot", fail)
    with pytest.raises(NetworkError, match="Injected"):
        if operation == "advance":
            model.advance(2)
        else:
            model.restore(checkpoint)
    assert model.checkpoint() == before


def test_unrepresentable_duration_cannot_advance_or_apply_commands():
    model = ScheduledThermalGraph(wrap(pipe_spec(), []))
    model.advance(1)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(1e-30)
    assert model.checkpoint() == before


def test_checkpoint_at_event_resumes_exactly_without_replaying_it():
    spec = wrap(
        pipe_spec(),
        [
            event(1, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0))),
            event(3, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
        ],
    )
    memory = ScheduledThermalGraph(spec)
    memory.advance(1)
    checkpoint = json.loads(json.dumps(memory.checkpoint()))
    fresh = ScheduledThermalGraph(spec)
    assert fresh.restore(checkpoint) == memory.snapshot()
    assert fresh.advance(4) == memory.advance(4)
    assert fresh.snapshot()["active_event_count"] == 2


def test_checkpoint_restores_future_schedule_and_unaligned_path():
    spec = wrap(
        pipe_spec(),
        [
            event(1.3, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
            event(4.7, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0))),
        ],
    )
    memory = ScheduledThermalGraph(spec)
    memory.advance(2.2)
    fresh = ScheduledThermalGraph(spec)
    fresh.restore(memory.checkpoint())
    assert memory.advance(3.8) == fresh.advance(3.8)


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "cursor",
        "clock",
        "active",
        "receipt",
        "source",
        "input",
        "temperature",
        "energy",
        "proposal",
        "flow",
        "nan",
    ],
)
def test_restore_rejects_forgery_without_state_mutation(damage):
    spec = wrap(pipe_spec(), [event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    model = ScheduledThermalGraph(spec)
    model.advance(2)
    before = model.checkpoint()
    bad = deepcopy(before)
    p = bad["payload"]
    s = p["state"]
    if damage == "digest":
        bad["sha256"] = "0" * 64
    elif damage == "cursor":
        s["active_event_count"] = 0
    elif damage == "clock":
        s["state"]["time_s"] = 3
    elif damage == "active":
        s["active_model_sha256"] = "0" * 64
    elif damage == "receipt":
        s["event_receipts"][0]["time_s"] = 0.5
    elif damage == "source":
        p["signature"]["python"] = "wrong"
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    elif damage == "temperature":
        s["state"]["temperatures_k"]["a"] += 1
    elif damage == "energy":
        s["state"]["integrated_external_energy_j"] += 1
    elif damage == "proposal":
        s["state"]["step_proposal_s"] = 99
    elif damage == "flow":
        s["state"]["flows_kg_s"]["pipe"] += 1
    elif damage == "nan":
        s["time_s"] = float("nan")
    if damage not in ("digest", "nan"):
        bad["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(bad)
    assert model.checkpoint() == before


def test_changed_future_timeline_cannot_reuse_checkpoint():
    spec = wrap(pipe_spec(), [event(100, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    model = ScheduledThermalGraph(spec)
    checkpoint = model.checkpoint()
    changed = deepcopy(spec)
    changed["timeline"][0]["changes"]["thermal_nodes"]["a"]["heat_power_w"] = 2000
    with pytest.raises(NetworkError):
        ScheduledThermalGraph(changed).restore(checkpoint)


def test_observations_and_exports_are_detached():
    spec = wrap(pipe_spec(), [event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    copy = deepcopy(spec)
    model = ScheduledThermalGraph(spec)
    model.advance(2)
    before = model.checkpoint()
    model.rates()
    observation = model.snapshot()
    observation["event_receipts"][0]["event_index"] = 99
    export = model.checkpoint()
    export["payload"]["signature"]["python"] = "changed"
    assert model.checkpoint() == before and spec == copy


@pytest.mark.parametrize(
    "damage",
    [
        "order",
        "duplicate",
        "negative_time",
        "bool_time",
        "nan_time",
        "unknown_group",
        "unknown_node",
        "unknown_edge",
        "role",
        "topology",
        "empty",
        "bool_value",
        "supply",
        "schema",
        "event_limit",
    ],
)
def test_invalid_schedule_contract(damage):
    spec = wrap(pipe_spec(), [event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0)))])
    if damage == "order":
        spec["timeline"] += [event(0.5, thermal_nodes=dict(a=dict(heat_power_w=2000.0)))]
    elif damage == "duplicate":
        spec["timeline"] *= 2
    elif damage == "negative_time":
        spec["timeline"][0]["time_s"] = -1
    elif damage == "bool_time":
        spec["timeline"][0]["time_s"] = True
    elif damage == "nan_time":
        spec["timeline"][0]["time_s"] = float("nan")
    elif damage == "unknown_group":
        spec["timeline"][0]["changes"] = {"materials": {"cp": 1}}
    elif damage == "unknown_node":
        spec["timeline"][0]["changes"] = {"thermal_nodes": {"missing": {"heat_power_w": 1}}}
    elif damage == "unknown_edge":
        spec["timeline"][0]["changes"] = {"hydraulic_edges": {"missing": {"pressure_gain_pa": 1}}}
    elif damage == "role":
        spec["timeline"][0]["changes"] = {"hydraulic_nodes": {"a": {"injection_kg_s": 1}}}
    elif damage == "topology":
        spec["timeline"][0]["changes"] = {"hydraulic_edges": {"pipe": {"from": "b"}}}
    elif damage == "empty":
        spec["timeline"][0]["changes"] = {}
    elif damage == "bool_value":
        spec["timeline"][0]["changes"]["thermal_nodes"]["a"]["heat_power_w"] = True
    elif damage == "supply":
        spec["timeline"][0]["changes"]["thermal_nodes"]["a"] = {"supply_temperature_k": 600}
    elif damage == "schema":
        spec["schema_version"] = True
    elif damage == "event_limit":
        spec["timeline"] = [
            event(i, thermal_nodes=dict(a=dict(heat_power_w=1.0))) for i in range(129)
        ]
    with pytest.raises(NetworkError):
        ScheduledThermalGraph(spec)


def test_source_drift_is_rejected_before_work(monkeypatch):
    model = ScheduledThermalGraph(wrap(pipe_spec(), []))
    before = model.checkpoint()
    monkeypatch.setattr("scripts.scheduled_thermal_graph.signature", lambda: {})
    with pytest.raises(ScheduledGraphError) as error:
        model.advance(1)
    assert error.value.report["criterion"] == "source_drift" and model.checkpoint() == before


def test_cli_resume_and_refuse_overwrite(tmp_path, monkeypatch):
    spec = wrap(pipe_spec(), [event(1, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0)))])
    source = tmp_path / "input.json"
    source.write_text(json.dumps(spec))
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setattr(
        sys, "argv", ["run", str(source), "--duration-s", "1", "--output", str(first)]
    )
    assert main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(source),
            "--duration-s",
            "1",
            "--resume",
            str(first / "checkpoint.json"),
            "--output",
            str(second),
        ],
    )
    assert main() == 0
    report = json.loads((second / "report.json").read_text())
    assert report["final"]["time_s"] == 2 and report["final"]["switches"] == []
    with pytest.raises(FileExistsError):
        main()


def test_cli_failed_event_never_writes_partial_checkpoint(tmp_path, monkeypatch):
    _, spec, _ = next(c for c in negative_cases() if c[0] == "invalid_right_state")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(spec))
    out = tmp_path / "result"
    monkeypatch.setattr(
        sys, "argv", ["run", str(source), "--duration-s", "10", "--output", str(out)]
    )
    assert main() == 2 and not (out / "checkpoint.json").exists()
