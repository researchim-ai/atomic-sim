"""Typed residual orientation, dimensions, algebraic rank and exact state lifting."""

from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError, PortSpec
from scripts.graph_port_adapter import GraphPortAdapter
from scripts.nonlinear_graph_references import cases, pipe_spec
from scripts.nonlinear_thermal_graph import NonlinearThermalGraph


@pytest.mark.parametrize("name,spec", list(cases()))
def test_initialization_is_dimensionally_consistent_without_warmup(name, spec):
    before = deepcopy(spec)
    a = GraphPortAdapter(spec)
    result = a.initialize()
    c = a.contract()
    assert spec == before and result["warmup_steps"] == 0 and not result["commands_changed"]
    assert len(c["variables"]) == len(c["rows"]) == len(result["state"])
    assert all(abs(v) <= row["tolerance"] for v, row in zip(result["residual"], c["rows"]))
    assert np.linalg.matrix_rank(a.algebraic_jacobian(result["state"])) == c["algebraic_count"]
    assert sum(v["differential"] for v in c["variables"]) == len(spec["volumes_m3"]) + 1
    assert result["state"][: len(spec["volumes_m3"])] == list(
        spec["initial_temperatures_k"].values()
    )


def test_ports_reconstruct_existing_typed_contract_and_cancel_pairwise():
    a = GraphPortAdapter(pipe_spec(reverse=True))
    initial = a.initialize()
    contract = a.contract()
    for name, spec in contract["ports"].items():
        port = PortSpec(**spec)
        assert port.name == name.rsplit(".", 1)[-1]
        assert port.energy_convention == "specific_enthalpy_transport"
    for link in contract["links"]:
        left, right = (initial["ports"][link[k]] for k in ("left", "right"))
        assert left["pressure_pa"] == right["pressure_pa"]
        assert left["mass_flow_kg_s"] + right["mass_flow_kg_s"] == 0
        assert left["energy_flow_w"] + right["energy_flow_w"] == 0


def test_zero_flow_does_not_invent_transported_temperature():
    spec = pipe_spec()
    spec["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    initial = GraphPortAdapter(spec).initialize()
    assert all(
        p["mass_flow_kg_s"] == 0 and p["stream_temperature_k"] is None
        for p in initial["ports"].values()
    )


def test_residual_and_ports_do_not_solve_hydraulics_or_mutate_state(monkeypatch):
    a = GraphPortAdapter(pipe_spec())
    initial = a.initialize()
    before = a.contract()
    y = np.array(initial["state"])
    dy = np.array(initial["rate"])
    copy_y = y.copy()
    copy_dy = dy.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Residual attempted an implicit solve")

    monkeypatch.setattr("scripts.nonlinear_thermal_graph.solve_hydraulic_graph", forbidden)
    monkeypatch.setattr(a._kernel, "rates", forbidden)
    for _ in range(2):
        assert a.residual(y, dy).tolist() == initial["residual"]
        assert a.ports(y) == initial["ports"]
    assert np.array_equal(y, copy_y) and np.array_equal(dy, copy_dy) and a.contract() == before


def test_algebraic_derivatives_are_ignored_and_mass_matrix_is_exact():
    a = GraphPortAdapter(pipe_spec())
    initial = a.initialize()
    y = np.array(initial["state"])
    dy = np.array(initial["rate"])
    perturb = np.arange(len(y), dtype=float)
    observed = a.residual(y, dy + perturb) - a.residual(y, dy)
    assert observed == pytest.approx(a.mass_matrix() @ perturb, abs=1e-8)
    altered = dy.copy()
    altered[2:-1] = 1e20
    assert a.residual(y, altered) == pytest.approx(a.residual(y, dy), abs=0)


def test_off_shell_momentum_mass_and_global_energy_identity():
    from scripts.nonlinear_graph_references import bridge_spec

    a = GraphPortAdapter(bridge_spec())
    initial = a.initialize()
    y = np.array(initial["state"])
    n = len(a._names)
    y[n + 1] += 10.0
    y[2 * n + 1] += 0.01
    residual = a.residual(y, np.zeros(len(y)))
    assert max(abs(residual[n:-1])) > 0
    rhs = -residual
    q = y[2 * n : -1]
    momentum = residual[2 * n : -1]
    assert rhs[:n].sum() - rhs[-1] == pytest.approx(-q @ momentum / a._kernel.rho, abs=1e-7)


@pytest.mark.parametrize("name,spec", list(cases()))
def test_analytic_algebraic_jacobian_matches_finite_differences(name, spec):
    a = GraphPortAdapter(spec)
    initial = a.initialize()
    y = np.array(initial["state"])
    n = len(spec["volumes_m3"])
    numerical = np.zeros_like(a.algebraic_jacobian(y))
    for j in range(n, len(y) - 1):
        h = 0.01 if j < 2 * n else 1e-6
        plus = y.copy()
        minus = y.copy()
        plus[j] += h
        minus[j] -= h
        numerical[:, j - n] = (
            a.residual(plus, initial["rate"])[n:-1] - a.residual(minus, initial["rate"])[n:-1]
        ) / (2 * h)
    assert numerical == pytest.approx(a.algebraic_jacobian(y), rel=1e-5, abs=1e-5)


def test_checkpoint_lift_preserves_temperatures_energy_clock_and_original_model():
    spec = pipe_spec()
    model = NonlinearThermalGraph(spec)
    model.advance(7.25)
    cp = model.checkpoint()
    before = deepcopy(cp)
    a = GraphPortAdapter(spec)
    initial = a.initialize()
    lifted = a.lift_checkpoint(cp)
    assert lifted["time_s"] == 7.25
    assert lifted["state"][:2] == list(model.snapshot()["temperatures_k"].values())
    assert lifted["state"][-1] == model.snapshot()["integrated_external_energy_j"]
    assert cp == before and a.initialize() == initial


def test_changed_input_and_corrupt_checkpoint_cannot_be_lifted():
    spec = pipe_spec()
    model = NonlinearThermalGraph(spec)
    cp = model.checkpoint()
    other = deepcopy(spec)
    other["volumes_m3"]["a"] = 0.6
    with pytest.raises(NetworkError):
        GraphPortAdapter(other).lift_checkpoint(cp)
    cp["sha256"] = "0" * 64
    with pytest.raises(NetworkError):
        GraphPortAdapter(spec).lift_checkpoint(cp)


@pytest.mark.parametrize("target", ["state", "rate"])
@pytest.mark.parametrize("damage", ["shape", "nan", "bool", "string"])
def test_invalid_vectors_rejected(target, damage):
    a = GraphPortAdapter(pipe_spec())
    initial = a.initialize()
    y = initial["state"]
    dy = initial["rate"]
    value = y if target == "state" else dy
    if damage == "shape":
        value.pop()
    else:
        value[0] = {"nan": float("nan"), "bool": True, "string": "310"}[damage]
    with pytest.raises(NetworkError):
        a.residual(y, dy)


def test_exports_are_detached_and_source_drift_blocks_initialization(monkeypatch):
    a = GraphPortAdapter(pipe_spec())
    before = a.contract()
    contract = a.contract()
    contract["ports"].clear()
    matrix = a.mass_matrix()
    matrix[:] = 0
    assert a.contract() == before and a.mass_matrix()[0, 0] > 0
    monkeypatch.setattr("scripts.graph_port_adapter.signature", lambda: {})
    with pytest.raises(NetworkError, match="drift"):
        a.initialize()
    with pytest.raises(NetworkError, match="drift"):
        a.lift_checkpoint({})


@pytest.mark.parametrize(
    "field,value", [("temperature", 249.0), ("pressure", 0.0), ("flow", 1e200)]
)
def test_invalid_physical_candidate_is_rejected_without_changing_adapter(field, value):
    a = GraphPortAdapter(pipe_spec())
    initial = a.initialize()
    state = initial["state"].copy()
    state[{"temperature": 0, "pressure": 2, "flow": 4}[field]] = value
    with pytest.raises(NetworkError):
        a.residual(state, initial["rate"])
    assert a.initialize() == initial


def test_no_pressure_boundary_is_invented_for_a_gauge_free_network():
    spec = pipe_spec()
    for node in spec["steady_input"]["hydraulic"]["nodes"]:
        node.pop("pressure_pa")
        node["injection_kg_s"] = 0.0
    with pytest.raises(NetworkError):
        GraphPortAdapter(spec)


def test_custom_initial_temperature_is_an_initial_condition_not_a_warmup():
    a = GraphPortAdapter(pipe_spec())
    result = a.initialize(dict(a=325.0, b=330.0))
    assert result["state"][:2] == [325.0, 330.0] and result["state"][-1] == 0.0
    assert result["time_s"] == 0.0 and result["warmup_steps"] == 0
    assert a.initialize()["state"][:2] == [310.0, 320.0]


def test_final_initialization_failure_does_not_mutate_adapter(monkeypatch):
    a = GraphPortAdapter(pipe_spec())
    before = a.initialize()
    actual = a.captured
    calls = 0

    def drift():
        nonlocal calls
        calls += 1
        return actual if calls == 1 else {}

    with monkeypatch.context() as patch:
        patch.setattr("scripts.graph_port_adapter.signature", drift)
        with pytest.raises(NetworkError, match="changed"):
            a.initialize()
    assert a.initialize() == before
