"""Supervisory transaction boundaries, causal receipts and restart contracts."""

import json
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.hysteresis_references import commands, relay
from scripts.nonlinear_thermal_graph import digest
from scripts.state_event_radau import StateEventError
from scripts.supervised_radau import SupervisedRadauGraph


def request(schedule=(), *, residence=1.0):
    controller = relay(residence=residence)
    controller["modes"].append(dict(id="hold", commands=commands(0), transitions=[]))
    return dict(
        format="atomic-supervised-mode-graph",
        schema_version=1,
        controller=controller,
        schedule=list(schedule),
        online_modes=["heat", "cool", "hold"],
    )


def calendar(time, mode, name="scheduled"):
    return dict(id=name, time_s=time, to=mode)


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def test_calendar_pause_keeps_stocks_and_expected_analytic_roots():
    model = SupervisedRadauGraph(request([calendar(20, "hold"), calendar(30, "cool", "resume")]))
    result = model.advance(40)
    assert result["calendar_cursor"] == 2 and result["revision"] == 1
    forced = [e for e in result["events"] if e["event"]["kind"] == "calendar"]
    assert [e["event"]["time_s"] for e in forced] == [20, 30]
    assert result["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(310.01875, abs=1e-9)
    for event in result["events"]:
        assert event["left"]["increments"] == event["right"]["increments"]
        assert (
            event["left"]["state"]["integrated_external_energy_j"]
            == event["right"]["state"]["integrated_external_energy_j"]
        )
    roots = [r["time_s"] for r in result["history"] if r["kind"] == "state"]
    assert roots == pytest.approx([5.5, 8, 13, 15.5, 32.25, 37.25, 39.75], abs=2e-9)


def test_external_only_terminal_modes_need_no_synthetic_transition():
    req = request([calendar(10, "hold")])
    req["controller"]["modes"] = [
        dict(id="heat", commands=commands(1000), transitions=[]),
        dict(id="hold", commands=commands(0), transitions=[]),
    ]
    req["online_modes"] = []
    model = SupervisedRadauGraph(req)
    result = model.advance(20)
    assert result["mode"] == "hold" and result["mode_transition_count"] == 1
    assert result["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(310.02, abs=1e-10)


@pytest.mark.parametrize("online", [False, True])
def test_noop_acknowledgement_preserves_mode_age(online):
    req = request([] if online else [calendar(3, "heat")], residence=4.5)
    model = SupervisedRadauGraph(req)
    initial = model.snapshot()["mode_entry_temperatures_k"]
    if online:
        model.advance(3)
        result = model.step("heat", 3, expected_revision=1, command_id="same:mode-1")
    else:
        result = model.advance(6)
    assert result["history"][0]["changed"] is False
    assert result["history"][0]["entry_frame_sha256"] == digest(initial)
    assert result["history"][1]["time_s"] == pytest.approx(5.5, abs=2e-9)
    assert result["mode_transition_count"] == 1


@pytest.mark.parametrize(
    "time,mode,criterion",
    [
        (5.5, "hold", "ambiguous_event_order"),
        (0.1, "cool", "minimum_residence"),
        (1.1, "cool", "minimum_residence"),
    ],
)
def test_late_calendar_or_state_failure_rolls_back_entire_prefix(time, mode, criterion):
    model = SupervisedRadauGraph(request([calendar(time, mode)]))
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.advance(15)
    assert error.value.report["criterion"] == criterion
    assert model.checkpoint() == before


def test_online_receipt_revision_duplicate_and_retry_are_transactional():
    model = SupervisedRadauGraph(request())
    model.advance(2)
    before = model.checkpoint()
    with pytest.raises(StateEventError):
        model.step("cool", 3, expected_revision=0, command_id="stale")
    assert model.checkpoint() == before
    result = model.step("hold", 10, expected_revision=1, command_id="pause-1")
    assert result["revision"] == 2 and result["online_command_count"] == 1
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.step("heat", 10, expected_revision=2, command_id="pause-1")
    assert error.value.report["criterion"] == "duplicate_command"
    assert model.checkpoint() == before
    assert model.step("heat", 10, expected_revision=2, command_id="resume-1")["revision"] == 3


def test_failed_online_hold_does_not_consume_id_or_revision():
    model = SupervisedRadauGraph(request())
    before = model.checkpoint()
    with pytest.raises(StateEventError):
        model.step("hold", 3, expected_revision=0, command_id="retry")
    assert model.checkpoint() == before
    result = model.step("heat", 2, expected_revision=0, command_id="retry")
    assert result["revision"] == 1


@pytest.mark.parametrize("cid", ["", "a b", "кириллица", "a/1", "a" * 129, 1, None])
def test_bad_command_id_is_atomic(cid):
    model = SupervisedRadauGraph(request())
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.step("heat", 2, expected_revision=0, command_id=cid)
    assert model.checkpoint() == before


@pytest.mark.parametrize("rev", [True, -1, 1, 0.0, None])
def test_revision_is_strict_integer_and_current(rev):
    model = SupervisedRadauGraph(request())
    with pytest.raises(StateEventError):
        model.step("heat", 2, expected_revision=rev, command_id="id")
    assert model.revision == 0


def test_online_permission_not_all_declared_modes():
    req = request()
    req["online_modes"] = ["hold"]
    model = SupervisedRadauGraph(req)
    with pytest.raises(StateEventError) as error:
        model.step("cool", 3, expected_revision=0, command_id="bad")
    assert error.value.report["criterion"] == "online_mode"


@pytest.mark.parametrize("stop_time,mode", [(2, "hold"), (10, "cool")])
def test_stop_consumes_calendar_or_root_but_not_noop(stop_time, mode):
    req = request([calendar(1, "heat"), calendar(stop_time, mode, "stop")])
    model = SupervisedRadauGraph(req)
    result = model.advance(20, stop_at_transition=True)
    expected = 2 if stop_time == 2 else 5.5
    assert result["time_s"] == pytest.approx(expected, abs=2e-9)
    assert result["stopped_at_transition"]
    assert result["history"][0]["changed"] is False


@pytest.fixture(scope="module")
def mixed():
    req = request([calendar(20, "hold"), calendar(30, "cool", "resume")])
    model = SupervisedRadauGraph(req)
    model.step("heat", 2, expected_revision=0, command_id="noop-1")
    model.advance(38)
    model.step("heat", 2, expected_revision=2, command_id="noop-2")
    return req, model.checkpoint()


def test_json_mixed_checkpoint_and_exact_continuation(mixed):
    req, checkpoint = mixed
    a, b = SupervisedRadauGraph(req), SupervisedRadauGraph(req)
    assert a.restore(json.loads(json.dumps(checkpoint))) == checkpoint["payload"]["state"]
    b.restore(checkpoint)
    assert a.advance(12) == b.advance(12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", True),
        ("event_count", 0),
        ("calendar_cursor", 0),
        ("mode_transition_count", 0),
        ("online_command_count", 0),
        ("mode", "hold"),
        ("mode_entry_time_s", 0),
        ("active_transitions", []),
        ("active_commands_sha256", "bad"),
    ],
)
def test_resealed_checkpoint_observation_corruption_is_atomic(mixed, field, value):
    req, checkpoint = mixed
    model = SupervisedRadauGraph(req)
    before = model.checkpoint()
    cp = deepcopy(checkpoint)
    cp["payload"]["state"][field] = value
    with pytest.raises(NetworkError):
        model.restore(reseal(cp))
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "other"),
        ("commit_revision", 0),
        ("observed_revision", True),
        ("command_id", "bad id"),
        ("changed", True),
        ("source_mode", "hold"),
        ("target_mode", "hold"),
        ("entry_frame_sha256", "bad"),
        ("sequence", False),
        ("residence_lower_bound_s", 1),
    ],
)
def test_resealed_online_receipt_corruption_is_atomic(mixed, field, value):
    req, checkpoint = mixed
    model = SupervisedRadauGraph(req)
    cp = deepcopy(checkpoint)
    cp["payload"]["state"]["history"][0][field] = value
    with pytest.raises(NetworkError):
        model.restore(reseal(cp))
    assert model.revision == 0


@pytest.mark.parametrize("mutation", ["future", "permissions", "macro"])
def test_checkpoint_binds_future_schedule_permissions_and_mesh(mixed, mutation):
    req, checkpoint = mixed
    req = deepcopy(req)
    if mutation == "future":
        req["schedule"].append(calendar(100, "hold", "future"))
    if mutation == "permissions":
        req["online_modes"] = ["hold"]
    model = SupervisedRadauGraph(req, max_macro_step_s=1 if mutation == "macro" else 2)
    with pytest.raises(NetworkError):
        model.restore(checkpoint)


@pytest.mark.parametrize("kind", ["events", "switches", "rhs", "root", "attempts", "history"])
def test_shared_resource_limit_rolls_back(kind):
    model = SupervisedRadauGraph(request([calendar(2, "hold")]))
    keys = dict(
        events="maximum_events_per_advance",
        switches="maximum_switches_per_advance",
        rhs="maximum_rhs_evaluations_per_advance",
        root="maximum_root_evaluations_per_advance",
        attempts="maximum_attempts_per_advance",
        history="maximum_history_entries",
    )
    model.budget[keys[kind]] = 0
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(10)
    assert model.checkpoint() == before


@pytest.mark.parametrize("bad", ["zero", "unordered", "duplicate", "unknown", "unused", "mask"])
def test_constructor_rejects_invalid_program(bad):
    req = request([calendar(2, "hold")])
    if bad == "zero":
        req["schedule"][0]["time_s"] = 0
    if bad == "unordered":
        req["schedule"].append(calendar(1, "heat", "later"))
    if bad == "duplicate":
        req["schedule"].append(calendar(3, "heat"))
    if bad == "unknown":
        req["online_modes"].append("unknown")
    if bad == "unused":
        req["online_modes"] = []
        req["schedule"] = []
    if bad == "mask":
        req["controller"]["modes"][-1]["commands"]["thermal_nodes"]["a"][
            "ambient_temperature_k"
        ] = 300
    with pytest.raises(NetworkError):
        SupervisedRadauGraph(req)


def test_online_acknowledgements_can_share_boundary_timestamp_with_calendar():
    req = request([calendar(2, "heat")])
    model = SupervisedRadauGraph(req)
    model.advance(2)
    model.step("heat", 1, expected_revision=1, command_id="after-calendar")
    restored = SupervisedRadauGraph(req)
    assert restored.restore(model.checkpoint()) == model.snapshot()
    assert [r["time_s"] for r in model.snapshot()["history"]] == [2, 2]


def test_online_success_prefix_rolls_back_if_later_calendar_fails():
    model = SupervisedRadauGraph(request([calendar(2.1, "cool")]))
    model.advance(2)
    before = model.checkpoint()
    with pytest.raises(StateEventError):
        model.step("hold", 2, expected_revision=1, command_id="retry-after-failure")
    assert model.checkpoint() == before


def test_final_observation_failure_is_transactional(monkeypatch):
    model = SupervisedRadauGraph(request())
    before = model.checkpoint()
    original = SupervisedRadauGraph._snapshot

    def fail_after_progress(self):
        if self._engine._time > 0:
            raise NetworkError("injected observation failure")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(SupervisedRadauGraph, "_snapshot", fail_after_progress)
        with pytest.raises(NetworkError):
            model.advance(10)
    assert model.checkpoint() == before


def test_final_source_failure_is_transactional(monkeypatch):
    model = SupervisedRadauGraph(request())
    before = model.checkpoint()
    import scripts.supervised_radau as module

    original = module.signature
    calls = []

    def drifting():
        value = original()
        calls.append(True)
        if len(calls) > 1:
            value["injected"] = True
        return value

    with monkeypatch.context() as patch:
        patch.setattr(module, "signature", drifting)
        with pytest.raises(StateEventError):
            model.advance(10)
    assert model.checkpoint() == before


def test_inputs_observations_checkpoints_are_detached():
    req = request([calendar(2, "hold")])
    model = SupervisedRadauGraph(req)
    req["schedule"][0]["to"] = "cool"
    result = model.advance(3)
    result["history"].clear()
    cp = model.checkpoint()
    cp["payload"]["state"]["mode_entry_temperatures_k"].clear()
    assert model.snapshot()["mode"] == "hold" and model.snapshot()["event_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("calendar_index", 3), ("calendar_id", "other"), ("time_s", 21), ("target_mode", "heat")],
)
def test_resealed_calendar_corruption_rejected(mixed, field, value):
    req, checkpoint = mixed
    cp = deepcopy(checkpoint)
    record = next(r for r in cp["payload"]["state"]["history"] if r["kind"] == "calendar")
    record[field] = value
    model = SupervisedRadauGraph(req)
    with pytest.raises(NetworkError):
        model.restore(reseal(cp))
    assert model.revision == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("guard_id", "unknown"),
        ("bracket_s", [0, 1]),
        ("bracket_values_k", [1, 2]),
        ("direction", True),
        ("rate_bounds_k_s", [-1, 1]),
    ],
)
def test_resealed_root_corruption_rejected(mixed, field, value):
    req, checkpoint = mixed
    cp = deepcopy(checkpoint)
    record = next(r for r in cp["payload"]["state"]["history"] if r["kind"] == "state")
    record[field] = value
    model = SupervisedRadauGraph(req)
    with pytest.raises(NetworkError):
        model.restore(reseal(cp))
    assert model.revision == 0
