"""Hybrid dispatch, warm reinitialization, coupled transaction and replay checks."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts import hybrid_thermal_graph as module
from scripts.hybrid_graph_references import calendar, cases, hold, hybrid
from scripts.hybrid_thermal_graph import HybridThermalGraph
from scripts.nonlinear_thermal_graph import digest
from scripts.state_event_radau import StateEventError


def rehash(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def test_analytic_idle_relay_and_no_switch_impulses():
    m = HybridThermalGraph(hybrid())
    result = m.advance(30.0)
    assert [e["winner"] for e in result["history"]] == [
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
        "cold",
        "hot",
    ]
    assert [e["time_s"] for e in result["history"]] == pytest.approx(
        [5.5, 8.0, 13.0, 15.5, 20.5, 23.0, 28.0], abs=2e-9
    )
    assert result["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(310.0375, abs=1e-9)
    for e in result["history"]:
        for key in [
            "temperatures_k",
            "temperature_increments_k",
            "integrated_external_energy_j",
            "energy_residual_j",
        ]:
            assert e["left_state"][key] == e["right_state"][key]
        assert e["sha256"] == digest({k: v for k, v in e.items() if k != "sha256"})
    assert abs(result["engine"]["state"]["energy_residual_j"]) < 1e-6


def test_calendar_is_exact_and_consumed_once_at_output_boundary():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("stop", 2.0, "hold")]
    m = HybridThermalGraph(spec)
    first = m.advance(2.0)
    assert first["mode"] == "hold" and first["calendar_cursor"] == 1
    assert first["history"][0]["time_s"] == 2.0
    assert len(m.advance(3.0)["history"]) == 1
    assert first["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(310.05)


def test_calendar_zero_is_explicit_pending_initial_input():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("stop", 0.0, "hold")]
    m = HybridThermalGraph(spec)
    assert m.snapshot()["calendar_cursor"] == 0
    r = m.advance(2.0)
    assert r["mode"] == "hold" and r["history"][0]["time_s"] == 0.0
    assert r["engine"]["state"]["temperatures_k"]["a"] == 310.0


def test_calendar_priority_consumes_all_and_tie_rolls_back():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("low", 2.0, "cool", 1), calendar("high", 2.0, "hold", 20)]
    m = HybridThermalGraph(spec)
    r = m.advance(3.0)
    assert r["mode"] == "hold" and r["calendar_cursor"] == 2
    assert r["history"][0]["winner"] == "high"
    assert len(r["history"][0]["contenders"]) == 2
    spec["calendar"][1]["priority"] = 1
    m = HybridThermalGraph(spec)
    before = m.checkpoint()
    with pytest.raises(StateEventError, match="unique highest priority"):
        m.advance(3.0)
    assert m.checkpoint() == before


def test_online_competes_with_calendar_and_controller_state_is_transactional():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("calendar", 0.0, "cool", 10)]
    m = HybridThermalGraph(spec)
    r = m.step("hold", 2.0, expected_revision=0, controller_state={"tick": 1}, priority=20)
    assert r["mode"] == "hold" and r["controller_state"] == {"tick": 1}
    assert r["history"][0]["winner"] == "online_0"
    before = m.checkpoint()
    with pytest.raises(NetworkError, match="Stale"):
        m.step("heat", 2.0, expected_revision=0, controller_state={"tick": 2})
    assert m.checkpoint() == before


def test_same_physical_guard_has_one_root_and_explicit_priority():
    spec = hold(hybrid())
    g = deepcopy(spec["modes"][0]["transitions"][0])
    g.update(id="low", to="hold", priority=1)
    spec["modes"][0]["transitions"].append(g)
    r = HybridThermalGraph(spec).advance(6.0)
    assert r["history"][0]["kind"] == "guard_root"
    assert r["history"][0]["winner"] == "hot"
    assert len(r["history"][0]["contenders"]) == 2
    assert r["mode"] == "cool"


def test_arming_delay_is_level_at_release_and_preserved_across_calls():
    spec = hybrid()
    spec["modes"][0]["transitions"][0]["arm_delay_s"] = 10.0
    m = HybridThermalGraph(spec)
    m.advance(7.0)
    assert m.snapshot()["mode"] == "heat"
    cp = m.checkpoint()
    other = HybridThermalGraph(spec)
    other.restore(cp)
    r = m.advance(4.0)
    assert other.advance(4.0) == r
    assert r["history"][0]["time_s"] == 10.0
    assert r["history"][0]["kind"] == "guard_level" and r["history"][0]["root"] is None
    assert r["history"][0]["left_state"]["temperatures_k"]["a"] == pytest.approx(310.25)


@pytest.mark.parametrize("cut", [2.0, 6.0, 9.0, 17.25])
def test_full_replay_restart_exact(cut):
    spec = hybrid()
    a = HybridThermalGraph(spec)
    a.advance(cut)
    cp = json.loads(json.dumps(a.checkpoint()))
    b = HybridThermalGraph(spec)
    assert b.restore(cp) == a.snapshot()
    assert a.advance(11.125) == b.advance(11.125)


@pytest.mark.parametrize("fault", ["clock", "history", "native", "calls", "controller", "extra"])
def test_rehashed_checkpoint_tamper_cannot_bypass_replay(fault):
    m = HybridThermalGraph(hybrid())
    m.step("heat", 9.0, expected_revision=0, controller_state={"i": 1})
    before = m.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    if fault == "clock":
        p["observation"]["time_s"] += 1
    elif fault == "history":
        p["observation"]["history"][0]["winner"] = "forged"
    elif fault == "native":
        p["native_checkpoint"]["sha256"] = "0" * 64
    elif fault == "calls":
        p["calls"][0]["duration_s"] += 1
    elif fault == "controller":
        p["observation"]["controller_state"]["i"] = 2
    else:
        p["extra"] = True
    with pytest.raises((NetworkError, ValueError)):
        m.restore(rehash(cp))
    assert m.checkpoint() == before


def test_input_mismatch_and_source_drift_refused(monkeypatch):
    spec = hybrid()
    m = HybridThermalGraph(spec)
    m.advance(2.0)
    cp = m.checkpoint()
    other = deepcopy(spec)
    other["modes"][0]["commands"]["thermal_nodes"]["a"]["heat_power_w"] *= 2
    with pytest.raises(NetworkError):
        HybridThermalGraph(other).restore(cp)
    monkeypatch.setattr(module, "signature", lambda: {"changed": True})
    with pytest.raises(StateEventError):
        m.advance(1.0)


@pytest.mark.parametrize(
    "state",
    [{"x": float("nan")}, {"x": float("inf")}, {1: "bad"}, {"x": (1, 2)}, {"x": "a" * 40000}],
)
def test_invalid_controller_state_never_committed(state):
    m = HybridThermalGraph(hybrid())
    before = m.checkpoint()
    with pytest.raises((NetworkError, ValueError, TypeError)):
        m.step("heat", 2.0, expected_revision=0, controller_state=state)
    assert m.checkpoint() == before


def test_late_switch_failure_rolls_back_physics_and_controller(monkeypatch):
    m = HybridThermalGraph(hybrid())
    before = m.checkpoint()
    original = m._build
    seen = []

    def fail(mode, engine):
        seen.append(mode)
        if len(seen) >= 3:
            raise NetworkError("injected late projection failure")
        return original(mode, engine)

    monkeypatch.setattr(m, "_build", fail)
    with pytest.raises(NetworkError):
        m.step("heat", 15.0, expected_revision=0, controller_state={"tick": 1})
    assert len(seen) == 3 and m.checkpoint() == before


def test_late_source_failure_rolls_back(monkeypatch):
    m = HybridThermalGraph(hybrid())
    before = m.checkpoint()
    original = module.signature
    calls = []

    def signature():
        calls.append(1)
        return original() if len(calls) == 1 else {"changed": True}

    monkeypatch.setattr(module, "signature", signature)
    with pytest.raises(StateEventError):
        m.advance(9.0)
    monkeypatch.setattr(module, "signature", original)
    assert m.checkpoint() == before


def test_zero_time_cascade_rejected_atomically():
    spec = hybrid()
    spec["modes"][0]["transitions"][0]["threshold_k"] = 309.0
    spec["modes"][1]["transitions"][0]["threshold_k"] = 311.0
    m = HybridThermalGraph(spec)
    before = m.checkpoint()
    with pytest.raises(StateEventError, match="budget"):
        m.advance(1.0)
    assert m.checkpoint() == before


def test_concurrent_same_revision_only_one_commit():
    m = HybridThermalGraph(hybrid())

    def command(i):
        try:
            return m.step("heat", 1.0, expected_revision=0, controller_state={"owner": i})[
                "revision"
            ]
        except NetworkError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(command, [1, 2]))
    assert sorted(map(str, results)) == ["1", "stale"]
    assert m.snapshot()["time_s"] == 1.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_calls", 0),
        ("maximum_switches_per_advance", 0),
        ("maximum_history_entries", 0),
        ("maximum_checkpoint_bytes", 100),
    ],
)
def test_resource_refusal_rolls_back(key, value):
    m = HybridThermalGraph(hybrid())
    before = m.checkpoint()
    old = m.budget[key]
    m.budget[key] = value
    with pytest.raises((NetworkError, ValueError)):
        m.advance(9.0)
    m.budget[key] = old
    assert m.checkpoint() == before


@pytest.mark.parametrize("bad", ["priority", "delay", "target", "mask", "reserved", "schema"])
def test_invalid_program_rejected(bad):
    spec = hybrid()
    if bad == "priority":
        spec["modes"][0]["transitions"][0]["priority"] = True
    elif bad == "delay":
        spec["modes"][0]["transitions"][0]["arm_delay_s"] = -1
    elif bad == "target":
        spec["modes"][0]["transitions"][0]["to"] = "missing"
    elif bad == "mask":
        spec["modes"][1]["commands"]["hydraulic_nodes"] = {"a": {"pressure_pa": 198000.0}}
    elif bad == "reserved":
        spec["modes"][0]["transitions"][0]["id"] = "online_0"
    else:
        spec["schema_version"] = True
    with pytest.raises((NetworkError, ValueError)):
        HybridThermalGraph(spec)


def test_flow_reversal_reprojects_algebraic_state_without_energy_impulse():
    spec = next(s for n, s, c in cases() if n == "flow_reversal")
    result = HybridThermalGraph(spec).advance(14.0)
    switched = [r for r in result["history"] if r["source_mode"] != r["target_mode"]]
    assert switched
    for r in switched:
        assert r["left_state"]["flows_kg_s"]["pipe"] * r["right_state"]["flows_kg_s"]["pipe"] < 0
        assert r["left_state"]["temperatures_k"] == r["right_state"]["temperatures_k"]
        assert (
            r["left_state"]["integrated_external_energy_j"]
            == r["right_state"]["integrated_external_energy_j"]
        )


def test_identical_guard_priority_tie_rejected_after_root_with_rollback():
    spec = hold(hybrid())
    other = deepcopy(spec["modes"][0]["transitions"][0])
    other.update(id="tied", to="hold")
    spec["modes"][0]["transitions"].append(other)
    m = HybridThermalGraph(spec)
    before = m.checkpoint()
    with pytest.raises(StateEventError) as error:
        m.advance(6.0)
    assert error.value.report["criterion"] == "priority_tie"
    assert m.checkpoint() == before


def test_distinct_unresolved_roots_not_ordered_by_priority():
    spec = hold(hybrid())
    other = deepcopy(spec["modes"][0]["transitions"][0])
    other.update(id="nearly_same", to="hold", threshold_k=310.1375 + 1e-13, priority=99)
    spec["modes"][0]["transitions"].append(other)
    m = HybridThermalGraph(spec)
    before = m.checkpoint()
    with pytest.raises(StateEventError) as error:
        m.advance(6.0)
    assert error.value.report["criterion"] in ["ambiguous_roots", "zero_time_cascade"]
    assert m.checkpoint() == before


def test_root_at_calendar_boundary_refused_without_epsilon_merging():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("same_boundary", 5.5, "hold", 99)]
    m = HybridThermalGraph(spec)
    before = m.checkpoint()
    with pytest.raises(StateEventError) as error:
        m.advance(6.0)
    assert error.value.report["criterion"] in ["boundary_root_overlap", "endpoint_contact"]
    assert m.checkpoint() == before


def test_sub_ulp_arm_delay_at_mode_entry_refused():
    spec = hold(hybrid())
    spec["initial_mode"] = "hold"
    spec["modes"][0]["transitions"][0]["arm_delay_s"] = 1e-300
    m = HybridThermalGraph(spec)
    m.advance(1.0)
    before = m.checkpoint()
    with pytest.raises(NetworkError, match="Unrepresentable guard arm"):
        m.step("heat", 1.0, expected_revision=1, controller_state={"attempt": 1})
    assert m.checkpoint() == before


def test_losing_online_action_still_commits_explicit_controller_state():
    spec = hold(hybrid())
    spec["calendar"] = [calendar("higher", 0.0, "heat", 99)]
    m = HybridThermalGraph(spec)
    result = m.step("hold", 2.0, expected_revision=0, controller_state={"attempt": 1}, priority=1)
    assert result["mode"] == "heat" and result["controller_state"] == {"attempt": 1}
    assert result["history"][0]["winner"] == "higher"
    assert {c["id"] for c in result["history"][0]["contenders"]} == {"online_0", "higher"}
