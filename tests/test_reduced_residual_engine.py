"""Registered reduced integration, rollback, checkpoints and bidirectional handoff."""

from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.nonlinear_graph_references import pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph, digest
from scripts.reduced_residual_engine import ReducedResidualEngine
from scripts.registered_graph_system import SYSTEM_TYPES, create_system, graph_request


def engine(spec=None, cap=2.0):
    return ReducedResidualEngine(
        graph_request(pipe_spec() if spec is None else spec), max_macro_step_s=cap
    )


def test_registered_factory_is_explicit_and_does_not_import_wire_code():
    request = graph_request(pipe_spec())
    before = deepcopy(request)
    system = create_system(request)
    assert system.contract["general_component_backend"] is False and request == before
    assert list(SYSTEM_TYPES) == ["constant_rho_thermal_graph_v1"]
    with pytest.raises(TypeError):
        SYSTEM_TYPES["other"] = object


@pytest.mark.parametrize("damage", ["kind", "schema", "extra", "format"])
def test_invalid_factory_request_rejected(damage):
    request = graph_request(pipe_spec())
    if damage == "kind":
        request["kind"] = "arbitrary.module.Factory"
    elif damage == "schema":
        request["schema_version"] = True
    elif damage == "extra":
        request["import"] = "os"
    else:
        request["format"] = "unknown"
    with pytest.raises(NetworkError):
        create_system(request)


def test_analytic_idle_heater_preserves_global_energy():
    spec = pipe_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    spec["steady_input"]["thermal"]["nodes"]["a"]["heat_power_w"] = 1000.0
    model = engine(spec)
    result = model.advance(10)
    assert result["state"]["temperatures_k"]["a"] == pytest.approx(310.02, abs=1e-10)
    assert result["state"]["integrated_external_energy_j"] == pytest.approx(10000.0, abs=1e-7)
    assert abs(result["state"]["energy_residual_j"]) < 1e-7
    assert result["segments"] and result["rhs_evaluations"] > 0


def test_continuation_never_calls_legacy_hydraulic_or_time_solver(monkeypatch):
    model = engine()

    def forbidden(*args, **kwargs):
        raise AssertionError("Reduced continuation called legacy solver")

    monkeypatch.setattr("scripts.nonlinear_thermal_graph.solve_hydraulic_graph", forbidden)
    monkeypatch.setattr("scripts.nonlinear_thermal_graph.NonlinearThermalGraph.advance", forbidden)
    result = model.advance(4)
    assert result["time_s"] == 4 and result["state"]["flows_kg_s"]["pipe"] > 0


def test_legacy_handoff_preserves_increment_frame_energy_and_time():
    spec = pipe_spec()
    legacy = NonlinearThermalGraph(spec)
    legacy.advance(3.75)
    cp = legacy.checkpoint()
    model = engine(spec)
    result = model.import_legacy_checkpoint(cp)
    assert result["increments"] == legacy._state.tolist()
    assert result["time_s"] == 3.75
    model.advance(2.25)
    exported = model.export_legacy_checkpoint()
    receiver = NonlinearThermalGraph(spec)
    receiver.restore(exported["checkpoint"])
    assert receiver._state.tolist() == model.snapshot()["increments"]
    assert receiver.snapshot()["time_s"] == 6
    assert exported["native_dp_proposal_reset"] and not exported["collocation_history_transferred"]
    assert receiver.snapshot()["step_proposal_s"] == spec["max_step_s"]
    receiver.advance(1)
    assert receiver.snapshot()["time_s"] == 7


def test_restart_preserves_algebraic_warm_seed_and_exact_continuation():
    a = engine()
    a.advance(3.75)
    cp = deepcopy(a.checkpoint())
    b = engine()
    assert b.restore(cp) == a.snapshot()
    assert a.advance(4.7) == b.advance(4.7)


@pytest.mark.parametrize("duration", [0.0, -1.0, True, float("nan"), float("inf"), 10001.0])
def test_invalid_duration_is_atomic(duration):
    model = engine()
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(duration)
    assert model.checkpoint() == before


@pytest.mark.parametrize("cap", [0.0, -1.0, True, float("nan"), 3.0])
def test_invalid_macro_step_rejected(cap):
    with pytest.raises(NetworkError):
        engine(cap=cap)


def test_late_segment_failure_rolls_back_earlier_work(monkeypatch):
    model = engine()
    before = model.checkpoint()
    original = model.system.admit
    calls = 0

    def failing(full, requested):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise NetworkError("injected later admission failure")
        return original(full, requested)

    monkeypatch.setattr(model.system, "admit", failing)
    with pytest.raises(NetworkError):
        model.advance(4)
    assert calls == 2 and model.checkpoint() == before


@pytest.mark.parametrize(
    "key", ["maximum_attempts", "maximum_rhs_evaluations_per_advance", "maximum_newton_iterations"]
)
def test_shared_resource_budgets_roll_back(key):
    model = engine()
    model.budget[key] = 1
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(5)
    assert model.checkpoint() == before


@pytest.mark.parametrize(
    "damage",
    [
        "digest",
        "source",
        "input",
        "model",
        "schema",
        "increments",
        "energy",
        "temperature",
        "pressure",
        "clock",
        "policy",
        "observation",
    ],
)
def test_resealed_checkpoint_corruption_is_rejected_without_mutation(damage):
    model = engine()
    model.advance(1)
    before = model.checkpoint()
    cp = deepcopy(before)
    p = cp["payload"]
    state = p["state"]
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
    elif damage == "increments":
        state["increments"].pop()
    elif damage == "energy":
        state["increments"][-1] += 1.0
    elif damage == "temperature":
        state["full_state"][0] += 1.0
    elif damage == "pressure":
        state["full_state"][2] += 100.0
    elif damage == "clock":
        state["time_s"] = -1.0
    elif damage == "policy":
        state["macro_restart_policy"] = "native_radau_history"
    else:
        state["state"]["temperatures_k"]["a"] += 1.0
    if damage != "digest":
        cp["sha256"] = digest(p)
    with pytest.raises(NetworkError):
        model.restore(cp)
    assert model.checkpoint() == before


def test_input_and_macro_policy_mismatches_cannot_restore():
    a = engine()
    a.advance(1)
    cp = a.checkpoint()
    with pytest.raises(NetworkError):
        engine(cap=1.0).restore(cp)
    spec = pipe_spec()
    spec["volumes_m3"]["a"] = 0.6
    with pytest.raises(NetworkError):
        engine(spec).import_legacy_checkpoint(NonlinearThermalGraph(pipe_spec()).checkpoint())


def test_detached_exports_do_not_change_state():
    model = engine()
    model.advance(1)
    before = model.checkpoint()
    exported = model.snapshot()
    exported["increments"][0] = 0
    cp = model.checkpoint()
    cp["payload"]["state"]["full_state"][0] = 0
    assert model.checkpoint() == before


@pytest.mark.parametrize("operation", ["advance", "restore", "import"])
def test_final_observation_failure_cannot_commit(operation, monkeypatch):
    model = engine()
    before = model.checkpoint()

    def fail(*args, **kwargs):
        raise NetworkError("injected observation failure")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_snapshot", fail)
        with pytest.raises(NetworkError):
            if operation == "advance":
                model.advance(1)
            elif operation == "restore":
                model.restore(before)
            else:
                model.import_legacy_checkpoint(NonlinearThermalGraph(pipe_spec()).checkpoint())
    assert model.checkpoint() == before


def test_source_drift_blocks_advance_restore_and_handoff(monkeypatch):
    model = engine()
    before = model.checkpoint()
    monkeypatch.setattr("scripts.reduced_residual_engine.signature", lambda: {})
    for call in [
        lambda: model.advance(1),
        lambda: model.restore(before),
        lambda: model.import_legacy_checkpoint({}),
        model.export_legacy_checkpoint,
    ]:
        with pytest.raises(NetworkError, match="drift"):
            call()
    assert model.checkpoint() == before


def test_real_domain_exit_rolls_back_full_advance():
    spec = pipe_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    spec["steady_input"]["thermal"]["nodes"]["a"]["heat_power_w"] = 1e9
    model = engine(spec)
    before = model.checkpoint()
    with pytest.raises(NetworkError):
        model.advance(1)
    assert model.checkpoint() == before


def test_newton_projection_satisfies_all_dimensional_algebraic_rows():
    model = engine()
    state = model.snapshot()
    z = np.array(state["increments"])
    z[0] = 0.25
    projected = model._project(z, np.array(state["full_state"]))
    rows = model.system.algebraic
    r = model.system.residual(projected, np.zeros(len(projected)))
    for i in rows:
        assert (
            abs(r[i])
            <= model.system.contract["rows"][i]["tolerance"]
            * model.budget["algebraic_tolerance_factor"]
        )
