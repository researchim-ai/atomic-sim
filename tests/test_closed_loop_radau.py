"""Joint controller/plant/RNG rollback, causal receipts and exact restart."""

import json
import random
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.closed_loop_radau import ClosedLoopRadau
from scripts.hysteresis_references import commands, relay
from scripts.nonlinear_thermal_graph import digest
from scripts.state_event_radau import StateEventError
from scripts.supervised_radau import SupervisedRadauGraph


def request(noise=0.01, seed=42, period=1.0, schedule=()):
    controller = relay(residence=0.1)
    for mode in controller["modes"]:
        mode["transitions"] = []
    controller["modes"].append(dict(id="hold", commands=commands(0), transitions=[]))
    return dict(
        format="atomic-closed-loop-radau",
        schema_version=1,
        plant=dict(
            format="atomic-supervised-mode-graph",
            schema_version=1,
            controller=controller,
            schedule=list(schedule),
            online_modes=["heat", "cool", "hold"],
        ),
        policy=dict(
            kind="sampled_schmitt_v1",
            sensor_node="a",
            low_k=310.04,
            high_k=310.1,
            heat_mode="heat",
            cool_mode="cool",
            initial_decision="heat",
            sample_period_s=period,
            noise_amplitude_k=noise,
            seed=seed,
        ),
    )


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def test_seeded_episode_restore_continues_entire_feedback_trajectory():
    req = request()
    a = ClosedLoopRadau(req)
    a.run(12, expected_revision=0)
    cp = json.loads(json.dumps(a.checkpoint()))
    b = ClosedLoopRadau(req)
    assert b.restore(cp) == a.snapshot()
    assert a.run(20, expected_revision=1) == b.run(20, expected_revision=1)
    assert len({r["decision"] for r in a.snapshot()["history"]}) == 2
    assert a.snapshot()["plant"]["time_s"] == 32


def test_controller_memory_holds_inside_band_and_changes_only_at_sample():
    a = ClosedLoopRadau(request(noise=0))
    result = a.run(9, expected_revision=0)
    decisions = [r["decision"] for r in result["history"]]
    assert decisions == ["heat"] * 4 + ["cool"] * 2 + ["heat"] * 3
    assert result["plant"]["engine"]["state"]["temperatures_k"]["a"] == pytest.approx(
        310.075, abs=1e-9
    )


def test_zero_noise_consumes_local_draw_but_never_changes_global_rng():
    before = random.getstate()
    a = ClosedLoopRadau(request(noise=0))
    a.run(3, expected_revision=0)
    rng = random.Random(42)
    assert [r["uniform_draw"] for r in a.snapshot()["history"]] == [rng.random() for _ in range(3)]
    assert random.getstate() == before


def test_failure_after_multiple_ticks_rolls_back_memory_rng_clock_and_receipts():
    req = request(schedule=[dict(id="bad", time_s=2.05, to="cool")])
    req["plant"]["controller"]["minimum_residence_s"] = 1
    a = ClosedLoopRadau(req)
    before = a.checkpoint()
    with pytest.raises(StateEventError):
        a.run(5, expected_revision=0)
    assert a.checkpoint() == before
    assert a.run(2, expected_revision=0)["tick_count"] == 2


@pytest.mark.parametrize("revision", [True, 1, -1, 0.0, None])
def test_stale_or_noninteger_revision_never_consumes_rng(revision):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    with pytest.raises(StateEventError):
        a.step(expected_revision=revision)
    assert a.checkpoint() == before


@pytest.mark.parametrize("ticks", [True, 0, -1, 65, 1.0, None])
def test_invalid_batch_is_atomic(ticks):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.run(ticks, expected_revision=0)
    assert a.checkpoint() == before


def test_partition_preserves_physics_rng_and_decisions():
    req = request()
    a, b = ClosedLoopRadau(req), ClosedLoopRadau(req)
    a.run(12, expected_revision=0)
    for i in range(3):
        b.run(4, expected_revision=i)
    sa, sb = a.snapshot(), b.snapshot()
    assert sa["plant"] == sb["plant"] and sa["rng_sha256"] == sb["rng_sha256"]
    assert [r["decision"] for r in sa["history"]] == [r["decision"] for r in sb["history"]]
    assert sa["revision"] == 1 and sb["revision"] == 3


@pytest.fixture(scope="module")
def checkpoint():
    a = ClosedLoopRadau(request())
    a.run(4, expected_revision=0)
    a.run(8, expected_revision=1)
    return a.checkpoint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tick_count", True),
        ("revision", 0),
        ("decision", "hold"),
        ("next_sample_time_s", 13),
        ("rng_sha256", "bad"),
        ("history", []),
    ],
)
def test_resealed_state_corruption_is_atomic(checkpoint, field, value):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    cp = deepcopy(checkpoint)
    cp["payload"]["state"][field] = value
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("uniform_draw", 0),
        ("measured_temperature_k", 400),
        ("previous_decision", "cool"),
        ("decision", "hold"),
        ("time_s", 1),
        ("end_time_s", 2),
        ("sequence", True),
        ("command_id", "other"),
        ("observed_plant_revision", True),
        ("plant_before_sha256", "bad"),
        ("plant_after_sha256", "bad"),
        ("commit_revision", 2),
    ],
)
def test_resealed_controller_ledger_corruption_rejected(checkpoint, field, value):
    cp = deepcopy(checkpoint)
    cp["payload"]["state"]["history"][0][field] = value
    a = ClosedLoopRadau(request())
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.revision == 0


def test_resealed_rng_state_rejected_even_with_matching_rng_hash(checkpoint):
    cp = deepcopy(checkpoint)
    cp["payload"]["rng_state"][1][0] ^= 1
    cp["payload"]["state"]["rng_sha256"] = digest(cp["payload"]["rng_state"])
    with pytest.raises(NetworkError):
        ClosedLoopRadau(request()).restore(reseal(cp))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("noise_amplitude_k", 0.02),
        ("sample_period_s", 2),
        ("low_k", 310.03),
        ("initial_decision", "cool"),
    ],
)
def test_checkpoint_binds_policy_future_noise_and_clock(checkpoint, field, value):
    req = request()
    req["policy"][field] = value
    with pytest.raises(NetworkError):
        ClosedLoopRadau(req).restore(checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 2**64),
        ("noise_amplitude_k", -1),
        ("noise_amplitude_k", 2),
        ("sample_period_s", 0),
        ("sample_period_s", 1e-8),
        ("high_k", 300),
        ("sensor_node", "unknown"),
        ("heat_mode", "unknown"),
        ("cool_mode", "heat"),
        ("initial_decision", "hold"),
        ("kind", "arbitrary_code"),
    ],
)
def test_policy_validation(field, value):
    req = request()
    req["policy"][field] = value
    with pytest.raises(NetworkError):
        ClosedLoopRadau(req)


def test_late_observation_failure_does_not_install_candidate(monkeypatch):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    original = ClosedLoopRadau.snapshot

    def fail(self):
        if self._ticks > 0:
            raise NetworkError("observation fault")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(ClosedLoopRadau, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.run(3, expected_revision=0)
    assert a.checkpoint() == before


def test_controller_failure_after_draw_is_atomic(monkeypatch):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()

    def fail(*args):
        raise NetworkError("policy fault")

    with monkeypatch.context() as patch:
        patch.setattr(ClosedLoopRadau, "_choose", fail)
        with pytest.raises(NetworkError):
            a.step(expected_revision=0)
    assert a.checkpoint() == before


def test_plant_failure_after_command_is_atomic(monkeypatch):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    original = SupervisedRadauGraph.step

    def fail(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise NetworkError("post command fault")

    with monkeypatch.context() as patch:
        patch.setattr(SupervisedRadauGraph, "step", fail)
        with pytest.raises(NetworkError):
            a.step(expected_revision=0)
    assert a.checkpoint() == before


def test_restore_initial_checkpoint_and_detached_results():
    req = request()
    a = ClosedLoopRadau(req)
    cp = a.checkpoint()
    a.run(2, expected_revision=0)
    assert a.restore(cp) == cp["payload"]["state"]
    result = a.run(2, expected_revision=0)
    result["history"].clear()
    req["policy"]["seed"] = 99
    assert len(a.snapshot()["history"]) == 2


def test_decimal_sampling_clock_is_anchored_to_tick_index():
    req = request(period=0.1)
    a = ClosedLoopRadau(req)
    a.run(17, expected_revision=0)
    b = ClosedLoopRadau(req)
    b.restore(a.checkpoint())
    assert a.run(9, expected_revision=1) == b.run(9, expected_revision=1)
    assert a.snapshot()["next_sample_time_s"] == 26 * 0.1


def test_episode_limit_is_checked_before_sampling():
    a = ClosedLoopRadau(request())
    a.budget["maximum_episode_ticks"] = 3
    a.run(2, expected_revision=0)
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.run(2, expected_revision=1)
    assert a.checkpoint() == before


@pytest.mark.parametrize("position,value", [(0, 0), (0, True), (5, 1), (4, 3), (-1, 1)])
def test_resealed_transaction_partition_corruption_rejected(checkpoint, position, value):
    cp = deepcopy(checkpoint)
    cp["payload"]["state"]["history"][position]["commit_revision"] = value
    with pytest.raises(NetworkError):
        ClosedLoopRadau(request()).restore(reseal(cp))


def test_final_source_drift_rolls_back_rng_and_plant(monkeypatch):
    import scripts.closed_loop_radau as module

    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    original = module.signature
    calls = []

    def drift():
        value = original()
        calls.append(True)
        if len(calls) > 1:
            value["injected"] = True
        return value

    with monkeypatch.context() as patch:
        patch.setattr(module, "signature", drift)
        with pytest.raises(StateEventError):
            a.run(2, expected_revision=0)
    assert a.checkpoint() == before


def test_restore_failure_in_final_observation_is_atomic(checkpoint, monkeypatch):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    original = ClosedLoopRadau.snapshot

    def fail(self):
        if self._ticks:
            raise NetworkError("restore observation fault")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(ClosedLoopRadau, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.restore(checkpoint)
    assert a.checkpoint() == before


@pytest.mark.parametrize("resource", ["rhs", "root", "attempts", "events", "switches"])
def test_shared_batch_work_budget_rolls_back_all_ticks(monkeypatch, resource):
    a = ClosedLoopRadau(request())
    before = a.checkpoint()
    original = ClosedLoopRadau._tick

    def costly(self, revision):
        used = original(self, revision)
        used[resource] = 200000
        return used

    with monkeypatch.context() as patch:
        patch.setattr(ClosedLoopRadau, "_tick", costly)
        with pytest.raises(StateEventError) as error:
            a.run(3, expected_revision=0)
    assert error.value.report["criterion"] == "controller_work_budget"
    assert a.checkpoint() == before


def test_sampled_policy_conflict_with_autonomous_guard_is_refused_not_delayed():
    req = request()
    req["plant"]["controller"] = relay(residence=0.1)
    req["plant"]["online_modes"] = ["heat", "cool"]
    req["policy"]["low_k"] = 310.03
    req["policy"]["high_k"] = 310.2
    model = ClosedLoopRadau(req)
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.run(40, expected_revision=0)
    assert error.value.report["criterion"] == "ambiguous_sample_order"
    assert model.checkpoint() == before


def test_guard_at_measurement_boundary_has_no_hidden_sampling_priority():
    req = request(noise=0.001)
    req["plant"]["controller"] = relay(residence=0.1)
    req["plant"]["controller"]["modes"][1]["transitions"] = []
    req["plant"]["online_modes"] = ["heat", "cool"]
    req["policy"]["low_k"] = 310.03
    req["policy"]["high_k"] = 310.13
    model = ClosedLoopRadau(req)
    before = model.checkpoint()
    with pytest.raises(StateEventError) as error:
        model.run(10, expected_revision=0)
    assert error.value.report["criterion"] == "ambiguous_sample_order"
    assert model.checkpoint() == before
