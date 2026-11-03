"""Online command lifecycle, failure atomicity, restart and observation isolation."""

import json
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.nonlinear_graph_references import bridge_spec, pipe_spec
from scripts.nonlinear_thermal_graph import digest
from scripts.online_thermal_graph import OnlineThermalGraph
from scripts.scheduled_thermal_graph import ScheduledThermalGraph


def command(power=1000):
    return dict(thermal_nodes=dict(a=dict(heat_power_w=power)))


def test_commands_equal_predetermined_schedule_and_preserve_global_energy():
    spec = pipe_spec()
    online = OnlineThermalGraph(spec)
    actions = [command(1000), command(-500), command(2000)]
    for i, changes in enumerate(actions):
        result = online.step(changes, 2.5, expected_revision=i)
        assert result["observation"]["revision"] == i + 1
        trace = result["command_switch"]
        for key in ["temperatures_k", "integrated_external_energy_j", "stored_energy_j"]:
            assert trace["left"][key] == trace["right"][key]
    scheduled = ScheduledThermalGraph(
        dict(
            format="atomic-scheduled-thermal-graph",
            schema_version=1,
            model_input=spec,
            timeline=[dict(time_s=i * 2.5, changes=c) for i, c in enumerate(actions)],
        )
    )
    # Matching external partitions make adaptive proposals/computations identical.
    for _ in actions:
        scheduled.advance(2.5)
    assert online.observation()["model"] == scheduled.snapshot()


@pytest.mark.parametrize("revision", [-1, 1, True, 0.0, None, "0"])
def test_stale_or_invalid_revision_has_no_effect(revision):
    model = OnlineThermalGraph(pipe_spec())
    before = model.checkpoint()
    with pytest.raises(NetworkError, match="revision"):
        model.step(command(), 1, expected_revision=revision)
    assert model.checkpoint() == before


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True, 10001])
def test_invalid_duration_rolls_back_command(duration):
    model = OnlineThermalGraph(pipe_spec())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(), duration, expected_revision=0)
    assert model.checkpoint() == before


def test_late_integration_failure_does_not_commit_command_or_revision():
    model = OnlineThermalGraph(pipe_spec())
    model.step(command(), 1, expected_revision=0)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(1e12), 10, expected_revision=1)
    assert model.checkpoint() == before
    model.step(command(0), 1, expected_revision=1)
    assert model.revision == 2


def test_combined_new_inlet_and_temperature_are_atomic():
    spec = bridge_spec()
    spec["steady_input"]["thermal"]["nodes"]["a"].pop("supply_temperature_k")
    model = OnlineThermalGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(dict(hydraulic_nodes=dict(a=dict(injection_kg_s=0.5))), 1, expected_revision=0)
    assert model.checkpoint() == before
    result = model.step(
        dict(
            hydraulic_nodes=dict(a=dict(injection_kg_s=0.5)),
            thermal_nodes=dict(a=dict(supply_temperature_k=350)),
        ),
        1,
        expected_revision=0,
    )
    assert result["observation"]["revision"] == 1


def test_restart_resumes_adaptive_path_and_rejects_stale_action():
    spec = pipe_spec()
    model = OnlineThermalGraph(spec)
    model.step(command(), 3.75, expected_revision=0)
    checkpoint = json.loads(json.dumps(model.checkpoint()))
    restored = OnlineThermalGraph(spec)
    assert restored.restore(checkpoint) == model.observation()
    changes = dict(hydraulic_nodes=dict(a=dict(pressure_pa=198000)))
    assert restored.step(changes, 2.3, expected_revision=1) == model.step(
        changes, 2.3, expected_revision=1
    )
    before = restored.checkpoint()
    with pytest.raises(NetworkError):
        restored.step(command(), 1, expected_revision=1)
    assert restored.checkpoint() == before


def test_initial_checkpoint_roundtrip():
    a = OnlineThermalGraph(pipe_spec())
    b = OnlineThermalGraph(pipe_spec())
    assert b.restore(a.checkpoint()) == a.observation()


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "revision",
        "bool_revision",
        "history",
        "source",
        "input",
        "model",
        "schema",
        "future",
    ],
)
def test_corrupt_checkpoints_are_rejected_atomically(damage):
    model = OnlineThermalGraph(pipe_spec())
    model.step(command(), 1, expected_revision=0)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if damage == "digest":
        cp["sha256"] = "0" * 64
    elif damage == "revision":
        p["revision"] = 2
    elif damage == "bool_revision":
        p["revision"] = True
    elif damage == "history":
        p["history"][0]["changes"]["thermal_nodes"]["a"]["heat_power_w"] = 2000
    elif damage == "source":
        p["signature"] = {}
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    elif damage == "model":
        p["model"] = "other"
    elif damage == "schema":
        p["schema_version"] = True
    else:
        p["history"].append(dict(time_s=2, changes=command()))
        p["revision"] = 2
    if damage != "digest":
        cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


@pytest.mark.parametrize("limit", ["maximum_commands", "maximum_episode_s"])
def test_episode_and_command_limits_do_not_commit_partial_step(limit):
    model = OnlineThermalGraph(pipe_spec())
    model.step(command(), 1, expected_revision=0)
    model.budget[limit] = 1
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step(command(), 1, expected_revision=1)
    assert model.checkpoint() == before


def test_exports_input_and_command_are_detached():
    spec = pipe_spec()
    original = deepcopy(spec)
    model = OnlineThermalGraph(spec)
    changes = command()
    result = model.step(changes, 1, expected_revision=0)
    before = model.checkpoint()
    result["observation"]["model"]["state"]["temperatures_k"]["a"] = 0
    changes["thermal_nodes"]["a"]["heat_power_w"] = 1e12
    exported = model.checkpoint()
    exported["payload"]["history"].clear()
    assert model.checkpoint() == before and spec == original


def test_source_drift_prevents_advance_and_restore(monkeypatch):
    model = OnlineThermalGraph(pipe_spec())
    cp = model.checkpoint()
    monkeypatch.setattr("scripts.online_thermal_graph.signature", lambda: {})
    with pytest.raises(NetworkError, match="drift"):
        model.step(command(), 1, expected_revision=0)
    with pytest.raises(NetworkError, match="drift"):
        model.restore(cp)
    assert model.checkpoint() == cp


@pytest.mark.parametrize("operation", ["step", "restore"])
def test_final_source_check_failure_cannot_commit(operation, monkeypatch):
    model = OnlineThermalGraph(pipe_spec())
    before = model.checkpoint()
    calls = 0

    def fail_second():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NetworkError("injected final source failure")

    monkeypatch.setattr(model, "_sources", fail_second)
    with pytest.raises(NetworkError):
        if operation == "step":
            model.step(command(), 1, expected_revision=0)
        else:
            model.restore(before)
    assert model.checkpoint() == before


@pytest.mark.parametrize("mode", ["unheld", "late_first", "empty_advanced"])
def test_resealed_valid_scheduled_state_cannot_bypass_online_hold_contract(mode):
    from scripts.online_thermal_graph import wrap

    spec = pipe_spec()
    model = OnlineThermalGraph(spec)
    history = (
        []
        if mode == "empty_advanced"
        else [dict(time_s=1.0 if mode == "late_first" else 0.0, changes=command())]
    )
    scheduled = ScheduledThermalGraph(wrap(spec, history))
    if mode != "unheld":
        scheduled.advance(2.0)
    cp = model.checkpoint()
    cp["payload"].update(
        history=history, revision=len(history), scheduled_checkpoint=scheduled.checkpoint()
    )
    cp["sha256"] = digest(cp["payload"])
    before = model.checkpoint()
    with pytest.raises(NetworkError, match="future, unheld or missing"):
        model.restore(cp)
    assert model.checkpoint() == before


def test_revision_is_read_only():
    model = OnlineThermalGraph(pipe_spec())
    with pytest.raises(AttributeError):
        model.revision = 12
