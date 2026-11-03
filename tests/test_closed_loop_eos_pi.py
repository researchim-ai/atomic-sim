"""Sampled PI recurrence, coupled plant, full rollback and restart."""

import json
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.closed_loop_eos_pi import ClosedLoopEOSPI, pi_sample
from scripts.controlled_eos_assembly import ControlledEOSAssembly
from scripts.eos_pi_references import IndependentPI, cases, request
from scripts.inertial_graph import InertialGraphError, digest


def test_exact_unsaturated_discrete_integral():
    p = request()["policy"]
    p.update(setpoint_k=301.0, kp_w_k=2.0, ki_w_k_s=10.0, sample_period_s=0.1)
    state = dict(filtered_k=300.0, integral_w=0.0, output_w=0.0)
    for k in range(1, 4):
        state, decision = pi_sample(p, state, 300.0)
        assert state["integral_w"] == pytest.approx(k)
        assert state["output_w"] == pytest.approx(2 + k)
        assert decision["integral_held"] is False


def test_filter_and_conditional_integrator_unwind():
    p = request()["policy"]
    p.update(
        setpoint_k=300.0,
        kp_w_k=1.0,
        ki_w_k_s=10.0,
        sample_period_s=0.1,
        output_min_w=0.0,
        output_max_w=100.0,
    )
    memory = dict(filtered_k=290.0, integral_w=100.0, output_w=100.0)
    after, d = pi_sample(p, memory, 290.0)
    assert after["integral_w"] == 100 and after["output_w"] == 100 and d["integral_held"]
    after, d = pi_sample(p, after, 310.0)
    assert after["integral_w"] == 90 and after["output_w"] == 80 and not d["integral_held"]
    p["filter_tau_s"] = 0.2
    memory = dict(filtered_k=300.0, integral_w=0.0, output_w=0.0)
    after, _ = pi_sample(p, memory, 310.0)
    assert after["filtered_k"] == pytest.approx(300 + 10 * (-np.expm1(-0.5)))
    assert memory["filtered_k"] == 300.0


@pytest.mark.parametrize("name,spec", list(cases()))
def test_independent_sampled_policy_and_absolute_energy_reference(name, spec):
    e = ClosedLoopEOSPI(spec)
    s = e.run(3, expected_revision=0)
    for method in ["DOP853", "Radau"]:
        ref = IndependentPI(spec).integrate(3, method)
        for actual, target in zip(s["history"], ref):
            assert actual["output_w"] == pytest.approx(target["output_w"], abs=0.01)
            assert actual["integral_after_w"] == pytest.approx(target["integral_w"], abs=0.01)
            assert actual["integral_held"] == target["integral_held"]
        for port, T in ref[-1]["port_temperatures_k"].items():
            assert s["plant"]["port_temperatures_k"][port] == pytest.approx(T, abs=1e-5)


@pytest.mark.parametrize("mode,expected", [("upper", 1000.0), ("lower", -1000.0)])
def test_saturation_does_not_accumulate_integral(mode, expected):
    e = ClosedLoopEOSPI(request(mode=mode))
    s = e.run(5, expected_revision=0)
    assert s["controller"]["integral_w"] == 0.0
    assert all(r["integral_held"] for r in s["history"])
    assert all(r["output_w"] == expected for r in s["history"])


def test_slew_limit_and_sampling_clock():
    e = ClosedLoopEOSPI(request(mode="slew"))
    s = e.run(5, expected_revision=0)
    assert [r["output_w"] for r in s["history"]] == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert s["next_sample_time_s"] == 0.05 and s["plant"]["time_s"] == 0.05
    assert s["plant"]["revision"] == 5 and s["revision"] == 1
    assert all(r["commit_revision"] == 1 for r in s["history"])


def test_all_tick_partitionings_have_identical_physical_and_controller_state():
    spec = request(mode="filtered")
    batch = ClosedLoopEOSPI(spec)
    split = ClosedLoopEOSPI(spec)
    batch.run(6, expected_revision=0)
    for _ in range(6):
        split.step(expected_revision=split.revision)
    a = batch.snapshot()
    b = split.snapshot()
    assert (
        a["plant"] == b["plant"]
        and a["controller"] == b["controller"]
        and a["tick_count"] == b["tick_count"]
    )
    assert [r["output_w"] for r in a["history"]] == [r["output_w"] for r in b["history"]]
    for obj in [batch, split]:
        restored = ClosedLoopEOSPI(spec)
        restored.restore(obj.checkpoint())
        assert restored.checkpoint() == obj.checkpoint()


def test_json_restart_initial_and_filtered_in_saturation_episode():
    spec = request(mode="filtered")
    original = ClosedLoopEOSPI(spec)
    other = ClosedLoopEOSPI(spec)
    other.restore(json.loads(json.dumps(original.checkpoint())))
    original.run(3, expected_revision=0)
    cp = json.loads(json.dumps(original.checkpoint()))
    other.restore(cp)
    original.run(2, expected_revision=original.revision)
    other.run(2, expected_revision=other.revision)
    assert original.checkpoint() == other.checkpoint()


@pytest.mark.parametrize(
    "defect",
    [
        "integral",
        "filter",
        "output",
        "recurrence",
        "truth",
        "hash_chain",
        "plant_receipt",
        "clock",
        "layout",
        "digest",
        "identity",
        "commit",
    ],
)
def test_corrupt_checkpoint_is_atomic(defect):
    e = ClosedLoopEOSPI(request())
    e.run(3, expected_revision=0)
    before = e.checkpoint()
    bad = deepcopy(before)
    s = bad["payload"]["state"]
    h = s["history"][0]
    if defect == "integral":
        s["controller"]["integral_w"] += 1
    elif defect == "filter":
        s["controller"]["filtered_k"] += 1
    elif defect == "output":
        s["controller"]["output_w"] += 1
    elif defect == "recurrence":
        h["integral_after_w"] += 1
    elif defect == "truth":
        h["truth_k"] += 1
    elif defect == "hash_chain":
        h["plant_after_sha256"] = "a" * 64
    elif defect == "plant_receipt":
        plant = bad["payload"]["plant_checkpoint"]
        plant["payload"]["snapshot"]["history"][0]["command_id"] = "different"
        plant["sha256"] = digest(plant["payload"])
    elif defect == "clock":
        s["tick_count"] = True
    elif defect == "layout":
        s["history"].pop()
    elif defect == "identity":
        bad["payload"]["input_sha256"] = "bad"
    elif defect == "commit":
        h["commit_revision"] = 2
    bad["sha256"] = digest(bad["payload"]) if defect != "digest" else "bad"
    with pytest.raises((NetworkError, InertialGraphError)):
        e.restore(bad)
    assert e.checkpoint() == before


@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "bool_revision",
        "ticks",
        "period",
        "gain",
        "filter",
        "bounds",
        "initial_output",
        "sensor",
        "actuator",
        "setpoint",
    ],
)
def test_invalid_request_and_call(defect):
    spec = request()
    if defect in ["stale", "bool_revision", "ticks"]:
        e = ClosedLoopEOSPI(spec)
        before = e.checkpoint()
        with pytest.raises((NetworkError, InertialGraphError)):
            if defect == "stale":
                e.run(1, expected_revision=1)
            elif defect == "bool_revision":
                e.run(1, expected_revision=False)
            else:
                e.run(33, expected_revision=0)
        assert e.checkpoint() == before
        return
    p = spec["policy"]
    if defect == "period":
        p["sample_period_s"] = 0
    elif defect == "gain":
        p["kp_w_k"] = -1
    elif defect == "filter":
        p["filter_tau_s"] = -1
    elif defect == "bounds":
        p["output_min_w"] = p["output_max_w"]
    elif defect == "initial_output":
        p["initial_output_w"] = 1
    elif defect == "sensor":
        p["sensor"] = "unknown"
    elif defect == "actuator":
        p["actuator"] = "unknown"
    elif defect == "setpoint":
        p["setpoint_k"] = 10000
    with pytest.raises(NetworkError):
        ClosedLoopEOSPI(spec)


def test_failure_on_second_tick_rolls_back_sensor_controller_and_plant(monkeypatch):
    e = ClosedLoopEOSPI(request(mode="filtered"))
    before = e.checkpoint()
    original = ControlledEOSAssembly.step
    calls = []

    def fail(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise InertialGraphError("injected", "Second tick failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ControlledEOSAssembly, "step", fail)
    with pytest.raises(InertialGraphError):
        e.run(3, expected_revision=0)
    monkeypatch.setattr(ControlledEOSAssembly, "step", original)
    assert e.checkpoint() == before
    e.run(3, expected_revision=0)
    assert e.snapshot()["tick_count"] == 3


def test_shared_batch_work_budget_and_source_drift_are_atomic(monkeypatch):
    import scripts.closed_loop_eos_pi as module

    e = ClosedLoopEOSPI(request())
    before = e.checkpoint()
    e.budget["maximum_batch_rhs_evaluations"] = 0
    with pytest.raises(InertialGraphError):
        e.run(2, expected_revision=0)
    assert e.checkpoint() == before
    e.budget["maximum_batch_rhs_evaluations"] = 300000
    original = module.signature
    monkeypatch.setattr(module, "signature", lambda: {"changed": True})
    with pytest.raises(InertialGraphError):
        e.run(1, expected_revision=0)
    monkeypatch.setattr(module, "signature", original)
    assert e.checkpoint() == before


def test_parallel_controller_calls_cannot_both_commit_same_revision():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    e = ClosedLoopEOSPI(request())
    barrier = Barrier(2)

    def call(_):
        barrier.wait(timeout=10)
        try:
            e.run(1, expected_revision=0)
            return True
        except InertialGraphError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call, range(2)))
    assert sum(results) == 1 and e.snapshot()["tick_count"] == 1


def test_closed_loop_reduces_error_in_isothermal_small_step_case():
    from scripts.controlled_assembly_references import request as assembly_request

    spec = request()
    spec["plant"] = assembly_request(uniform=True)
    for r in spec["plant"]["circuits"].values():
        r["channel"]["initial"]["temperatures_k"] = [300.0] * 4
        for c in r["chambers"]:
            c["initial_temperature_k"] = 300.0
    spec["policy"].update(setpoint_k=300.1, kp_w_k=2000.0, ki_w_k_s=100.0)
    e = ClosedLoopEOSPI(spec)
    s = e.run(10, expected_revision=0)
    assert abs(s["plant"]["port_temperatures_k"]["wall.w0"] - 300.1) < 0.08
    assert abs(s["plant"]["energy_residual_j"]) < 1e-7


def test_episode_tick_limit_is_shared_across_transactions():
    e = ClosedLoopEOSPI(request())
    e.budget["maximum_episode_ticks"] = 2
    e.run(2, expected_revision=0)
    before = e.checkpoint()
    with pytest.raises(NetworkError):
        e.step(expected_revision=e.revision)
    assert e.checkpoint() == before


def test_nonfinite_policy_arithmetic_rolls_back_entire_episode():
    spec = request()
    spec["policy"].update(setpoint_k=310.0, kp_w_k=1e308)
    e = ClosedLoopEOSPI(spec)
    before = e.checkpoint()
    with pytest.raises(NetworkError):
        e.run(1, expected_revision=0)
    assert e.checkpoint() == before


def test_zero_remaining_work_does_not_start_a_plant_transaction(monkeypatch):
    e = ClosedLoopEOSPI(request())
    before = e.checkpoint()
    e.budget["maximum_batch_rhs_evaluations"] = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Plant must not start without allowance")

    monkeypatch.setattr(ControlledEOSAssembly, "step", forbidden)
    with pytest.raises(InertialGraphError):
        e.run(1, expected_revision=0)
    assert e.checkpoint() == before


def test_remaining_batch_work_is_enforced_inside_second_tick(monkeypatch):
    e = ClosedLoopEOSPI(request())
    before = e.checkpoint()
    e.budget["maximum_batch_rhs_evaluations"] = 90
    original = ControlledEOSAssembly.step
    allowances = []

    def bounded(self, *args, **kwargs):
        allowances.append(self.budget["maximum_rhs_evaluations"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ControlledEOSAssembly, "step", bounded)
    with pytest.raises(InertialGraphError):
        e.run(2, expected_revision=0)
    assert len(allowances) == 2 and allowances[0] == 90 and 0 < allowances[1] < 90
    assert e.checkpoint() == before
