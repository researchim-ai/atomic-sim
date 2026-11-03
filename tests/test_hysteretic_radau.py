"""Repeatable mode guards, residence, bounded history and exact cyclic restart."""

import json
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.hysteresis_references import cases, negative_cases, relay
from scripts.hysteretic_radau import HystereticRadauGraph
from scripts.nonlinear_thermal_graph import digest
from scripts.state_event_radau import StateEventError


def test_relay_rearms_and_preserves_analytic_hysteresis_band_and_period():
    model = HystereticRadauGraph(relay())
    result = model.advance(30.0)
    assert [r["guard_id"] for r in result["history"]] == [
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
    ]
    assert [r["time_s"] for r in result["history"]] == pytest.approx(
        [5.5, 8.0, 13.0, 15.5, 20.5, 23.0, 28.0], abs=2e-9
    )
    assert result["mode"] == "cool" and result["transition_count"] == 7
    assert result["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(310.0375, abs=1e-9)
    assert abs(result["engine"]["state"]["energy_residual_j"]) < 1e-6
    for trace in result["switches"]:
        assert trace["left"]["increments"] == trace["right"]["increments"]
        assert (
            trace["left"]["state"]["integrated_external_energy_j"]
            == trace["right"]["state"]["integrated_external_energy_j"]
        )


def test_mode_residence_is_measured_from_entry_not_last_advance():
    request = relay(residence=6.0)
    model = HystereticRadauGraph(request)
    model.advance(3.0)
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(3.0)
    assert error.value.report["criterion"] == "minimum_residence"
    assert model.checkpoint() == before


def test_minimum_residence_never_delays_or_suppresses_a_crossing():
    model = HystereticRadauGraph(relay(residence=10.0))
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(20.0)
    assert error.value.report["criterion"] == "minimum_residence" and model.checkpoint() == before


def test_checkpoint_at_transition_rearms_next_mode_and_resumes_exactly():
    request = relay()
    a = HystereticRadauGraph(request)
    stopped = a.advance(30.0, stop_at_transition=True)
    assert (
        stopped["stopped_at_transition"]
        and stopped["mode"] == "cool"
        and len(stopped["history"]) == 1
    )
    b = HystereticRadauGraph(request)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(20.0) == b.advance(20.0)
    assert a.snapshot()["transition_count"] > 3


@pytest.mark.parametrize("cut", [3.0, 9.0, 17.25])
def test_mid_cycle_checkpoint_retains_mode_entry_clock_and_reference_frame(cut):
    request = relay()
    a = HystereticRadauGraph(request)
    a.advance(cut)
    before = a.snapshot()
    b = HystereticRadauGraph(request)
    assert b.restore(a.checkpoint()) == before
    assert a.advance(12.0) == b.advance(12.0)


def test_repeated_history_beyond_calendar_limit_is_supported_across_calls():
    request = relay(high=310.001, low=310.0, heat=12500.0, cool=-12500.0, residence=0.001)
    a = HystereticRadauGraph(request)
    for _ in range(3):
        a.advance(2.01)
    assert a.snapshot()["transition_count"] > 128
    b = HystereticRadauGraph(request)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(0.17) == b.advance(0.17)


def test_pressure_relay_returns_explicit_commands_without_hidden_path_state():
    request = dict(cases())["pressure_relay"]
    a = HystereticRadauGraph(request)
    first = a.advance(20.0, stop_at_transition=True)
    second = a.advance(20.0, stop_at_transition=True)
    assert first["mode"] == "cool" and second["mode"] == "heat"
    assert (
        first["engine"]["state"]["flows_kg_s"]["pipe"]
        < 0
        < second["engine"]["state"]["flows_kg_s"]["pipe"]
    )
    assert first["engine"]["state"]["pressures_pa"]["a"] == 198000.0
    assert second["engine"]["state"]["pressures_pa"]["a"] == 200000.0


def test_terminal_mode_does_not_rearm_or_invent_transitions():
    a = HystereticRadauGraph(dict(cases())["terminal_mode"])
    result = a.advance(20.0)
    assert (
        result["mode"] == "hold"
        and result["active_transitions"] == []
        and result["transition_count"] == 1
    )
    assert a.advance(20.0)["switches"] == []
    standby = HystereticRadauGraph(dict(cases())["standby"]).advance(10.0, stop_at_transition=True)
    assert not standby["stopped_at_transition"] and standby["time_s"] == 10.0


def test_initial_mode_commands_apply_at_zero_without_energy_reset():
    request = relay()
    request["model_input"]["steady_input"]["thermal"]["nodes"]["a"]["heat_power_w"] = 0.0
    model = HystereticRadauGraph(request)
    initial = model.snapshot()
    assert initial["transition_count"] == 0 and initial["mode_entry_time_s"] == 0.0
    assert initial["engine"]["state"]["integrated_external_energy_j"] == 0.0
    assert len(model.advance(6.0)["history"]) == 1


@pytest.mark.parametrize("name", [name for name, _, _ in negative_cases()])
def test_real_unsupported_transitions_roll_back_full_call(name):
    spec = next(s for n, s, _ in negative_cases() if n == name)
    model = HystereticRadauGraph(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(20.0)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "key",
    [
        "maximum_switches_per_advance",
        "maximum_history_entries",
        "maximum_rhs_evaluations_per_advance",
        "maximum_root_evaluations_per_advance",
        "maximum_attempts_per_advance",
    ],
)
def test_shared_resource_caps_preserve_previously_accepted_history(key):
    model = HystereticRadauGraph(relay())
    first = model.advance(30.0, stop_at_transition=True)
    model.budget[key] = (
        1 if key in ("maximum_switches_per_advance", "maximum_history_entries") else 0
    )
    before = model.checkpoint()
    with pytest.raises(StateEventError):
        model.advance(20.0)
    assert model.checkpoint() == before and len(model.snapshot()["history"]) == len(
        first["history"]
    )


def test_rhs_budget_is_shared_across_repeated_mode_entries():
    request = relay()
    probe = HystereticRadauGraph(request)
    first = probe.advance(30.0, stop_at_transition=True)
    model = HystereticRadauGraph(request)
    model.budget["maximum_rhs_evaluations_per_advance"] = first["evaluations"]["rhs"]
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(30.0)
    assert error.value.report["criterion"] == "rhs_budget" and model.checkpoint() == before


@pytest.mark.parametrize(
    "damage",
    [
        "mode",
        "mask",
        "unknown_target",
        "self_loop",
        "unreachable",
        "duplicate_id",
        "duplicate_guard",
        "node",
        "direction",
        "bool_direction",
        "threshold",
        "residence",
        "initial_contact",
        "format",
        "schema",
        "too_many_modes",
    ],
)
def test_invalid_mode_contract_is_rejected(damage):
    spec = relay()
    g = spec["modes"][0]["transitions"][0]
    if damage == "mode":
        spec["initial_mode"] = "absent"
    elif damage == "mask":
        spec["modes"][1]["commands"]["thermal_nodes"]["a"]["ambient_temperature_k"] = 300.0
    elif damage == "unknown_target":
        g["to"] = "absent"
    elif damage == "self_loop":
        g["to"] = "heat"
    elif damage == "unreachable":
        extra = deepcopy(spec["modes"][1])
        extra["id"] = "unreachable"
        extra["transitions"] = []
        spec["modes"].append(extra)
    elif damage == "duplicate_id":
        spec["modes"][1]["id"] = "heat"
    elif damage == "duplicate_guard":
        spec["modes"][1]["transitions"][0]["id"] = "hot"
    elif damage == "node":
        g["node"] = "absent"
    elif damage == "direction":
        g["direction"] = 0
    elif damage == "bool_direction":
        g["direction"] = True
    elif damage == "threshold":
        g["threshold_k"] = 1e9
    elif damage == "residence":
        spec["minimum_residence_s"] = 1e-12
    elif damage == "initial_contact":
        g["threshold_k"] = 310.0
    elif damage == "format":
        spec["format"] = "other"
    elif damage == "schema":
        spec["schema_version"] = True
    else:
        spec["modes"] = [deepcopy(spec["modes"][0]) for _ in range(17)]
    with pytest.raises(NetworkError):
        HystereticRadauGraph(spec)


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "source",
        "input",
        "model",
        "schema",
        "count",
        "bool_count",
        "mode",
        "entry_time",
        "entry_map",
        "sequence",
        "source_mode",
        "target_mode",
        "guard",
        "bracket",
        "root_value",
        "commands",
        "active",
        "observation",
        "native",
    ],
)
def test_resealed_mode_checkpoint_corruption_is_rejected_atomically(damage):
    model = HystereticRadauGraph(relay())
    model.advance(20.0)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    s = p["state"]
    r = s["history"][0]
    if damage == "digest":
        cp["sha256"] = "0" * 64
    elif damage == "source":
        p["signature"] = {}
    elif damage == "input":
        p["input_sha256"] = "0" * 64
    elif damage == "model":
        p["model"] = "other"
    elif damage == "schema":
        p["schema_version"] = True
    elif damage == "count":
        s["transition_count"] += 1
    elif damage == "bool_count":
        s["transition_count"] = True
    elif damage == "mode":
        s["mode"] = "cool" if s["mode"] == "heat" else "heat"
    elif damage == "entry_time":
        s["mode_entry_time_s"] = 0.0
    elif damage == "entry_map":
        s["mode_entry_temperatures_k"]["a"] += 0.01
    elif damage == "sequence":
        r["sequence"] = 2
    elif damage == "source_mode":
        r["source_mode"] = "cool"
    elif damage == "target_mode":
        r["target_mode"] = "heat"
    elif damage == "guard":
        r["guard_id"] = "absent"
    elif damage == "bracket":
        r["bracket_s"][0] = 0.0
    elif damage == "root_value":
        r["root_value_k"] = 1.0
    elif damage == "commands":
        r["target_commands_sha256"] = "0" * 64
    elif damage == "active":
        s["active_transitions"] = []
    elif damage == "observation":
        s["engine"]["state"]["temperatures_k"]["a"] += 0.1
    else:
        p["native_checkpoint"]["sha256"] = "0" * 64
    if damage != "digest":
        cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


def test_future_modes_and_macro_policy_are_bound_by_checkpoint():
    request = relay()
    model = HystereticRadauGraph(request)
    cp = model.checkpoint()
    changed = deepcopy(request)
    changed["modes"][1]["commands"]["thermal_nodes"]["a"]["heat_power_w"] = -30000.0
    with pytest.raises(NetworkError):
        HystereticRadauGraph(changed).restore(cp)
    with pytest.raises(NetworkError):
        HystereticRadauGraph(request, max_macro_step_s=1.0).restore(cp)


@pytest.mark.parametrize("duration", [0.0, -1.0, True, float("nan"), float("inf"), 10001.0])
def test_invalid_duration_does_not_commit_mode_or_history(duration):
    model = HystereticRadauGraph(relay())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(duration)
    assert model.checkpoint() == before


@pytest.mark.parametrize("operation", ["advance", "restore"])
def test_final_observation_failure_does_not_commit(operation, monkeypatch):
    model = HystereticRadauGraph(relay())
    before = model.checkpoint()

    def fail(*args, **kwargs):
        raise NetworkError("injected final observation failure")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_snapshot", fail)
        with pytest.raises(NetworkError):
            if operation == "advance":
                model.advance(20.0)
            else:
                model.restore(before)
    assert model.checkpoint() == before


def test_exports_inputs_and_runtime_drift_are_checked(monkeypatch):
    request = relay()
    original = deepcopy(request)
    model = HystereticRadauGraph(request)
    report = model.advance(10.0)
    before = model.checkpoint()
    report["history"].clear()
    report["engine"]["increments"][0] = 0.0
    assert model.checkpoint() == before and request == original
    monkeypatch.setattr("scripts.hysteretic_radau.signature", lambda: {})
    with pytest.raises(StateEventError, match="changed"):
        model.advance(1.0)
    with pytest.raises(StateEventError, match="changed"):
        model.restore(before)
    assert model.checkpoint() == before


def test_relay_cycles_do_not_call_legacy_time_integrator(monkeypatch):
    model = HystereticRadauGraph(relay())

    def forbidden(*args, **kwargs):
        raise AssertionError("Called legacy DP advance")

    monkeypatch.setattr("scripts.nonlinear_thermal_graph.NonlinearThermalGraph.advance", forbidden)
    assert model.advance(20.0)["transition_count"] >= 4
