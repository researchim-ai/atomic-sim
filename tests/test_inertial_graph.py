"""Physical inertial graph limits, dimensional residuals and transactional restart."""

import json
import math
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.inertial_graph import InertialGraph, InertialGraphError, digest
from scripts.inertial_graph_references import boundary, cases, graph, junction, pipe


def single():
    return deepcopy(dict((n, s) for n, s, _ in cases())["single_pipe"])


def reseal(cp):
    cp["sha256"] = digest(cp["payload"])
    return cp


def test_single_pipe_matches_linear_transient_and_kinetic_work_heat_analytically():
    a = InertialGraph(single())
    r = a.advance(20)["state"]
    e = math.exp(-2)
    assert r["flows_kg_s"]["ab"] == pytest.approx(2 * (1 - e), abs=1e-9)
    assert r["kinetic_energy_j"] == pytest.approx(20 * (1 - e) ** 2, abs=1e-8)
    assert r["pressure_work_j"] == pytest.approx(4 * (20 - 10 * (1 - e)), abs=1e-8)
    assert r["exported_heat_j"] == pytest.approx(
        4 * (20 - 20 * (1 - e) + 5 * (1 - e * e)), abs=1e-8
    )
    assert abs(r["energy_residual_j"]) < 1e-8
    assert r["gravity_work_j"] == 0


def test_cycle_decay_has_no_boundary_power_and_exports_exact_kinetic_loss():
    spec = next(s for n, s, _ in cases() if n == "free_cycle_decay")
    a = InertialGraph(spec)
    r = a.advance(20)["state"]
    for flow in r["flows_kg_s"].values():
        assert flow == pytest.approx(2 * math.exp(-2), abs=1e-9)
    assert r["pressure_work_j"] == pytest.approx(0, abs=1e-9)
    assert r["kinetic_energy_j"] == pytest.approx(60 * math.exp(-4), abs=1e-8)
    assert r["exported_heat_j"] == pytest.approx(60 * (1 - math.exp(-4)), abs=1e-8)


def test_series_pressure_is_consistent_at_start_without_steady_flow_prehistory():
    spec = next(s for n, s, _ in cases() if n == "series_junction")
    a = InertialGraph(spec)
    r = a.snapshot()["state"]
    assert r["flows_kg_s"] == dict(aj=0.0, jb=0.0)
    assert r["pressures_pa"]["j"] == pytest.approx(202000)
    assert r["flow_rates_kg_s2"] == pytest.approx(dict(aj=0.1, jb=0.1))
    assert a.system.dimension == 1


def test_zero_degree_injection_tree_still_accounts_boundary_work_and_heat():
    spec = next(s for n, s, _ in cases() if n == "injection_tree")
    a = InertialGraph(spec)
    r = a.advance(12)["state"]
    assert a.system.dimension == 0 and r["flows_kg_s"] == dict(aj=-1.0)
    assert r["pressures_pa"]["j"] == 201000 and r["kinetic_energy_j"] == 5
    assert r["pressure_work_j"] == pytest.approx(12) and r["exported_heat_j"] == pytest.approx(12)


def test_hydrostatic_column_is_motionless_and_gravity_work_is_zero():
    a = InertialGraph(next(s for n, s, _ in cases() if n == "hydrostatic_column"))
    r = a.advance(40)["state"]
    assert r["flows_kg_s"] == dict(aj=0.0, jb=0.0)
    assert r["pressures_pa"]["j"] == 249050
    assert r["gravity_work_j"] == r["kinetic_energy_j"] == r["exported_heat_j"] == 0


def test_gravity_throughflow_uses_explicit_potential_work_not_pump_head():
    a = InertialGraph(next(s for n, s, _ in cases() if n == "gravity_throughflow"))
    r = a.advance(20)["state"]
    assert r["flows_kg_s"]["aj"] > 0 and r["gravity_work_j"] > 0
    assert abs(r["pressure_work_j"]) < 1e-7
    assert r["kinetic_energy_j"] + r["exported_heat_j"] == pytest.approx(
        r["gravity_work_j"], abs=1e-6
    )


def test_boundary_pressure_jump_preserves_flows_kinetic_and_integrated_stocks(monkeypatch):
    a = InertialGraph(single())
    a.advance(5)
    before = a.snapshot()["state"]
    seen = []
    original = InertialGraph._observe

    def observe(self, state, time):
        result = original(self, state, time)
        if time == 5 and self._boundaries["a"] == 198000:
            seen.append(result)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(InertialGraph, "_observe", observe)
        a.step(dict(a=198000, b=200000), 10, expected_revision=1, command_id="reverse")
    assert seen
    for key in (
        "flows_kg_s",
        "kinetic_energy_j",
        "pressure_work_j",
        "gravity_work_j",
        "exported_heat_j",
    ):
        assert seen[0][key] == before[key]
    assert a.snapshot()["state"]["flows_kg_s"]["ab"] < 0


@pytest.mark.parametrize("name", [n for n, _, _ in cases()])
def test_json_checkpoint_exact_continuation_for_every_topology(name):
    spec = next(s for n, s, _ in cases() if n == name)
    a, b = InertialGraph(spec), InertialGraph(spec)
    a.advance(7.3)
    cp = json.loads(json.dumps(a.checkpoint()))
    assert b.restore(cp) == a.snapshot()
    assert a.advance(3.7) == b.advance(3.7)


def test_full_residual_is_pure_and_zero_at_consistent_differential_rate():
    a = InertialGraph(next(s for n, s, _ in cases() if n == "parallel_loop"))
    a.advance(3)
    s = a.system
    r = a.snapshot()["state"]
    before = a.checkpoint()
    y = np.r_[
        [r["flows_kg_s"][e] for e in s.edges],
        [r["pressures_pa"][n] for n in s.names],
        [r[k] for k in ("pressure_work_j", "gravity_work_j", "exported_heat_j")],
    ]
    dy = np.r_[
        [r["flow_rates_kg_s2"][e] for e in s.edges],
        np.zeros(len(s.names)),
        [r[k] for k in ("pressure_power_w", "gravity_power_w", "dissipation_w")],
    ]
    y0, dy0 = y.copy(), dy.copy()
    first = a.residual(y, dy)
    second = a.residual(y, dy)
    assert np.max(abs(first)) < 1e-8 and np.array_equal(first, second)
    assert np.array_equal(y, y0) and np.array_equal(dy, dy0) and a.checkpoint() == before
    dy[0] += 1
    assert a.residual(y, dy)[0] == pytest.approx(s.L[0], abs=1e-8)


def test_residual_uses_active_boundary_after_command():
    a = InertialGraph(single())
    a.step(dict(a=198000, b=200000), 1, expected_revision=0, command_id="reverse")
    r = a.snapshot()["state"]
    s = a.system
    y = np.r_[
        list(r["flows_kg_s"].values()),
        list(r["pressures_pa"].values()),
        [r[k] for k in ("pressure_work_j", "gravity_work_j", "exported_heat_j")],
    ]
    dy = np.r_[
        list(r["flow_rates_kg_s2"].values()),
        [0, 0],
        [r[k] for k in ("pressure_power_w", "gravity_power_w", "dissipation_w")],
    ]
    assert max(abs(a.residual(y, dy))) < 1e-8
    assert max(abs(s.residual(y, dy))) > 1000


@pytest.mark.parametrize(
    "field,value",
    [
        ("inertance_pa_s2_kg", 0),
        ("inertance_pa_s2_kg", -1),
        ("linear_pa_s_kg", -1),
        ("quadratic_pa_s2_kg2", -1),
        ("kind", "pump"),
        ("from", "missing"),
        ("to", "a"),
    ],
)
def test_invalid_or_unregistered_pipe_rejected(field, value):
    spec = single()
    spec["edges"][0][field] = value
    with pytest.raises(NetworkError):
        InertialGraph(spec)


@pytest.mark.parametrize(
    "case",
    [
        "unanchored",
        "disconnected",
        "bad_initial_mass",
        "duplicate_node",
        "duplicate_edge",
        "extra_head",
        "bad_density",
        "bad_domain",
        "unknown_node_kind",
    ],
)
def test_invalid_topology_and_initialization_rejected(case):
    spec = next(s for n, s, _ in cases() if n == "series_junction")
    if case == "unanchored":
        spec["nodes"] = [junction(n["id"]) for n in spec["nodes"]]
    if case == "disconnected":
        spec["nodes"].append(junction("isolated"))
    if case == "bad_initial_mass":
        spec["initial_flows_kg_s"]["aj"] = 1
    if case == "duplicate_node":
        spec["nodes"][1]["id"] = "a"
    if case == "duplicate_edge":
        spec["edges"][1]["id"] = "aj"
    if case == "extra_head":
        spec["edges"][0]["pressure_gain_pa"] = 1000
    if case == "bad_density":
        spec["density_kg_m3"] = 0
    if case == "bad_domain":
        spec["min_pressure_pa"] = spec["max_pressure_pa"]
    if case == "unknown_node_kind":
        spec["nodes"][1]["kind"] = "compressible_volume"
    with pytest.raises(NetworkError):
        InertialGraph(spec)


def test_inconsistent_initial_flows_are_not_projected_or_repaired():
    spec = next(s for n, s, _ in cases() if n == "series_junction")
    spec["initial_flows_kg_s"]["aj"] = 1
    original = deepcopy(spec)
    with pytest.raises(InertialGraphError) as error:
        InertialGraph(spec)
    assert error.value.report["criterion"] == "initial_mass" and spec == original


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf"), 1001])
def test_invalid_duration_is_atomic(duration):
    a = InertialGraph(single())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.advance(duration)
    assert a.checkpoint() == before


@pytest.mark.parametrize("revision", [True, 1, -1, 0.0, None])
def test_stale_revision_is_atomic(revision):
    a = InertialGraph(single())
    before = a.checkpoint()
    with pytest.raises(InertialGraphError):
        a.step(dict(a=198000, b=200000), 1, expected_revision=revision, command_id="id")
    assert a.checkpoint() == before


def test_failed_pressure_command_does_not_consume_id_and_retry_succeeds():
    a = InertialGraph(single())
    a.advance(2)
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=3e6, b=200000), 1, expected_revision=1, command_id="retry")
    assert a.checkpoint() == before
    a.step(dict(a=198000, b=200000), 1, expected_revision=1, command_id="retry")
    with pytest.raises(NetworkError):
        a.step(dict(a=202000, b=200000), 1, expected_revision=2, command_id="retry")


def test_trial_domain_failure_after_command_leaves_no_partial_history():
    spec = single()
    spec["max_abs_flow_kg_s"] = 3
    a = InertialGraph(spec)
    a.advance(2)
    before = a.checkpoint()
    with pytest.raises(InertialGraphError):
        a.step(dict(a=1000000, b=200000), 40, expected_revision=1, command_id="overload")
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "resource", ["maximum_rhs_evaluations", "maximum_internal_steps", "maximum_commands"]
)
def test_resource_failure_rolls_back(resource):
    a = InertialGraph(single())
    a.budget[resource] = 0
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=198000, b=200000), 1, expected_revision=0, command_id="limited")
    assert a.checkpoint() == before


@pytest.fixture(scope="module")
def checkpoint():
    a = InertialGraph(single())
    a.advance(2)
    a.step(dict(a=198000, b=200000), 3, expected_revision=1, command_id="reverse")
    return a.checkpoint()


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", True),
        ("revision", 0),
        ("history", []),
        ("active_pressures_pa", dict(a=202000, b=200000)),
        ("contract", {}),
    ],
)
def test_resealed_checkpoint_state_corruption_is_atomic(checkpoint, field, value):
    cp = deepcopy(checkpoint)
    cp["payload"]["snapshot"][field] = value
    a = InertialGraph(single())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.restore(reseal(cp))
    assert a.checkpoint() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", True),
        ("command_id", "bad id"),
        ("time_s", 5),
        ("time_s", 0),
        ("commit_revision", True),
        ("commit_revision", 3),
        ("pressures_pa", dict(a=3e6, b=200000)),
    ],
)
def test_resealed_command_corruption_rejected(checkpoint, field, value):
    cp = deepcopy(checkpoint)
    cp["payload"]["snapshot"]["history"][0][field] = value
    with pytest.raises(NetworkError):
        InertialGraph(single()).restore(reseal(cp))


@pytest.mark.parametrize("change", ["coordinate", "energy", "basis", "input", "signature"])
def test_resealed_checkpoint_native_or_identity_corruption_rejected(checkpoint, change):
    cp = deepcopy(checkpoint)
    p = cp["payload"]
    if change == "coordinate":
        p["coordinates"][0] += 0.1
    if change == "energy":
        p["coordinates"][-1] = -100
    if change == "basis":
        p["snapshot"]["contract"]["basis_sha256"] = "bad"
    if change == "input":
        p["input_sha256"] = "bad"
    if change == "signature":
        p["signature"]["python"] = "foreign"
    with pytest.raises(NetworkError):
        InertialGraph(single()).restore(reseal(cp))


def test_inputs_outputs_and_checkpoints_are_detached():
    spec = single()
    a = InertialGraph(spec)
    spec["edges"][0]["linear_pa_s_kg"] = 99999
    r = a.advance(2)
    r["state"]["flows_kg_s"].clear()
    cp = a.checkpoint()
    cp["payload"]["coordinates"].clear()
    assert a.snapshot()["state"]["flows_kg_s"]["ab"] > 0


def test_observation_failure_and_source_drift_are_atomic(monkeypatch):
    a = InertialGraph(single())
    before = a.checkpoint()
    original = InertialGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("observation failed")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(InertialGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.advance(2)
    assert a.checkpoint() == before
    import scripts.inertial_graph as module

    original_sig = module.signature
    calls = []

    def drift():
        v = original_sig()
        calls.append(True)
        if len(calls) > 1:
            v["drift"] = True
        return v

    with monkeypatch.context() as patch:
        patch.setattr(module, "signature", drift)
        with pytest.raises(NetworkError):
            a.advance(2)
    assert a.checkpoint() == before


def test_eight_node_multicycle_graph_preserves_all_junction_constraints():
    spec = next(s for n, s, _ in cases() if n == "eight_node_chain")
    spec["edges"] += [
        pipe("bypass1", "n0", "n3", 7000, 3000, 100),
        pipe("bypass2", "n2", "n5", 8000, 4000, 100),
        pipe("bypass3", "n4", "n7", 9000, 5000, 200),
    ]
    spec["initial_flows_kg_s"] = {e["id"]: 0.0 for e in spec["edges"]}
    a = InertialGraph(spec)
    r = a.advance(15)["state"]
    assert a.system.dimension == 4
    q = np.array([r["flows_kg_s"][e] for e in a.system.edges])
    assert max(abs(a.system.Af @ q)) < 1e-12
    assert abs(r["energy_residual_j"]) < 1e-6


def test_two_boundaries_and_no_free_nodes_keep_independent_pipe_momenta():
    spec = graph(
        [boundary("a", 202000), boundary("b")],
        [pipe("p1", "a", "b"), pipe("p2", "a", "b", 20000, 1000)],
    )
    a = InertialGraph(spec)
    r = a.advance(10)["state"]
    assert a.system.dimension == 2
    assert r["flows_kg_s"]["p1"] == pytest.approx(2 * (1 - math.exp(-1)), abs=1e-9)
    assert r["flows_kg_s"]["p2"] == pytest.approx(2 * (1 - math.exp(-0.5)), abs=1e-9)


@pytest.mark.parametrize("missing", ["a", "b"])
def test_command_requires_complete_boundary_mask(missing):
    a = InertialGraph(single())
    before = a.checkpoint()
    values = dict(a=202000, b=200000)
    values.pop(missing)
    with pytest.raises(NetworkError):
        a.step(values, 1, expected_revision=0, command_id="incomplete")
    assert a.checkpoint() == before


@pytest.mark.parametrize("value", ["", "bad id", "неascii", None, "x" * 129])
def test_command_id_contract_is_enforced(value):
    a = InertialGraph(single())
    before = a.checkpoint()
    with pytest.raises(NetworkError):
        a.step(dict(a=202000, b=200000), 1, expected_revision=0, command_id=value)
    assert a.checkpoint() == before


def test_checkpoint_binds_original_flow_frame_and_maximum_step(checkpoint):
    spec = single()
    spec["initial_flows_kg_s"]["ab"] = 0.1
    with pytest.raises(NetworkError):
        InertialGraph(spec).restore(checkpoint)
    with pytest.raises(NetworkError):
        InertialGraph(single(), max_step_s=0.5).restore(checkpoint)


def test_history_with_same_boundary_command_value_still_has_unique_receipts():
    a = InertialGraph(single())
    a.step(dict(a=202000, b=200000), 1, expected_revision=0, command_id="same-1")
    a.step(dict(a=202000, b=200000), 1, expected_revision=1, command_id="same-2")
    b = InertialGraph(single())
    assert b.restore(a.checkpoint()) == a.snapshot()
    cp = a.checkpoint()
    cp["payload"]["snapshot"]["history"][1]["time_s"] = 0
    with pytest.raises(NetworkError):
        b.restore(reseal(cp))


def test_restoration_failure_after_observation_does_not_install_candidate(checkpoint, monkeypatch):
    a = InertialGraph(single())
    before = a.checkpoint()
    original = InertialGraph.snapshot

    def fail(self):
        if self._time > 0:
            raise NetworkError("restore observation fault")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(InertialGraph, "snapshot", fail)
        with pytest.raises(NetworkError):
            a.restore(checkpoint)
    assert a.checkpoint() == before


def test_ill_conditioned_pressure_projection_is_refused_without_regularization():
    spec = graph(
        [boundary("a"), junction("j"), junction("k")],
        [pipe("aj", "a", "j", 1e20), pipe("jk", "j", "k", 1)],
    )
    with pytest.raises(NetworkError):
        InertialGraph(spec)


def test_kinetic_overflow_is_rejected_instead_of_accepting_nan_balance():
    spec = single()
    spec["edges"][0]["inertance_pa_s2_kg"] = 1e308
    spec["initial_flows_kg_s"]["ab"] = 2
    with pytest.raises(NetworkError), np.errstate(over="ignore", invalid="ignore"):
        InertialGraph(spec)


def test_strided_svd_basis_never_changes_observation_after_copy_or_json_restore():
    spec = next(s for n, s, _ in cases() if n == "eight_node_chain")
    a = InertialGraph(spec, max_step_s=0.05)
    for duration in (2, 8, 0.5, 9.5):
        a.advance(duration)
    assert deepcopy(a).snapshot() == a.snapshot()
    b = InertialGraph(spec, max_step_s=0.05)
    assert b.restore(json.loads(json.dumps(a.checkpoint()))) == a.snapshot()
    assert a.advance(1) == b.advance(1)


def test_lossless_pipe_acceleration_converts_pressure_work_only_into_kinetic_energy():
    spec = single()
    spec["edges"][0]["linear_pa_s_kg"] = 0
    a = InertialGraph(spec)
    r = a.advance(4)["state"]
    assert r["flows_kg_s"]["ab"] == pytest.approx(0.8, abs=1e-12)
    assert r["kinetic_energy_j"] == pytest.approx(3.2, abs=1e-11)
    assert r["pressure_work_j"] == pytest.approx(3.2, abs=1e-11)
    assert r["exported_heat_j"] == 0


def test_lossless_closed_cycle_retains_initial_kinetic_energy():
    spec = next(s for n, s, _ in cases() if n == "free_cycle_decay")
    for edge in spec["edges"]:
        edge["linear_pa_s_kg"] = 0
    a = InertialGraph(spec)
    r = a.advance(20)["state"]
    assert r["flows_kg_s"] == dict(ab=2.0, bc=2.0, ca=2.0)
    assert r["kinetic_energy_j"] == 60 and r["exported_heat_j"] == 0
    assert r["pressure_work_j"] == 0
