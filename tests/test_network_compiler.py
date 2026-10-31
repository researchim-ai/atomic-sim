"""Typed topology, structural diagnostics and constant thermal-network execution."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from components.linear_thermal_network import LinearThermalNetwork, thermal_catalog
from core.network import (
    LinearComponentType,
    NetworkError,
    PortSpec,
    RowSpec,
    StateSpec,
    compile_network,
)
from core.status import ModelPassport, ModelStatus
from sdk.network_experiment import run_network_experiment


def example():
    return json.loads(Path("scenarios/ai05_thermal_two_capacity.json").read_text())


def test_compiler_initializes_dimensional_ports_and_common_flow_once():
    spec = example()
    compiled = compile_network(spec["topology"], thermal_catalog())
    fixed = {"hot.state.temperature_k": 330, "cold.state.temperature_k": 300}
    state, rate = compiled.initialize(fixed)
    assert state[compiled.index["hot.port.thermal.heat_flow_w"]] == pytest.approx(-600)
    assert rate[compiled.index["hot.state.temperature_k"]] == pytest.approx(-0.57)
    assert state[compiled.index["cold.port.thermal.heat_flow_w"]] == pytest.approx(600)
    assert rate[compiled.index["cold.state.temperature_k"]] == pytest.approx(0.3)
    assert max(abs(compiled.residual(state, rate))) < 1e-7
    for link, _, _ in compiled.links:
        routed = compiled.route_flux(link, {"heat_flow_w": 17.3})
        assert sum(v["heat_flow_w"] for v in routed.values()) == 0
    spec["topology"]["nodes"][0]["parameters"]["heat_capacity_j_k"] = 99
    assert compiled.specification()["nodes"][0]["parameters"]["heat_capacity_j_k"] == 1000
    matrices = compiled.matrices()
    matrices[0][:] = 0
    assert max(abs(compiled.residual(state, rate))) < 1e-7
    before = state.copy()
    compiled.residual(state, rate)
    assert np.array_equal(state, before)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_node",
        "duplicate_node",
        "unknown_port",
        "duplicate_link",
        "version",
        "extra_field",
        "missing_port",
        "fanout",
        "self_link",
        "unknown_model",
        "bad_parameter",
    ],
)
def test_invalid_topology_rejected_before_execution(tmp_path, fault):
    spec = example()
    topology = spec["topology"]
    if fault == "unknown_node":
        topology["connections"][0]["from"] = "missing.thermal"
    elif fault == "duplicate_node":
        topology["nodes"].append(deepcopy(topology["nodes"][0]))
    elif fault == "unknown_port":
        topology["connections"][0]["to"] = "link.missing"
    elif fault == "duplicate_link":
        topology["connections"].append(deepcopy(topology["connections"][0]))
    elif fault == "version":
        topology["nodes"][0]["model_version"] = "99.0.0"
    elif fault == "extra_field":
        topology["nodes"][0]["silent_override"] = True
    elif fault == "missing_port":
        topology["connections"].pop()
    elif fault == "fanout":
        topology["connections"][1]["from"] = "hot.thermal"
    elif fault == "self_link":
        topology["connections"][0]["to"] = "hot.thermal"
    elif fault == "unknown_model":
        topology["nodes"][0]["model_id"] = "unqualified-pump"
    elif fault == "bad_parameter":
        topology["nodes"][0]["parameters"]["heat_capacity_j_k"] = float("nan")
    with pytest.raises(NetworkError):
        run_network_experiment(spec, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_disconnected_closed_islands_rejected():
    topology = example()["topology"]
    extra = deepcopy(topology)
    for n in extra["nodes"]:
        n["id"] = "other_" + n["id"]
    for link in extra["connections"]:
        link["id"] = "other_" + link["id"]
        link["from"] = "other_" + link["from"]
        link["to"] = "other_" + link["to"]
    topology["nodes"] += extra["nodes"]
    topology["connections"] += extra["connections"]
    with pytest.raises(NetworkError, match="islands"):
        compile_network(topology, thermal_catalog())


def passport(name):
    return ModelPassport(
        model_id=name,
        version="1.0.0",
        status=ModelStatus.IMPLEMENTED,
        phenomenon="Test-only declared linear contract",
        approximations=("synthetic",),
        applicability="Tests only",
        data_sources=("Test equation",),
    )


def signal_catalog(unit="K"):
    source = LinearComponentType(
        passport("source"),
        (PortSpec("out", "signal", "output", signal_unit="K"),),
        (),
        (RowSpec("drive", "K", 300, 1e-9),),
        (),
        lambda _: (np.zeros((1, 1)), np.ones((1, 1)), np.array([310])),
    )
    sink = LinearComponentType(
        passport("sink"),
        (PortSpec("inlet", "signal", "input", signal_unit=unit),),
        (),
        (),
        (),
        lambda _: (np.zeros((0, 1)), np.zeros((0, 1)), np.zeros(0)),
    )
    return {"source": source, "sink": sink}


def signal_spec():
    return dict(
        schema_version=1,
        nodes=[
            dict(id=name, model_id=model, model_version="1.0.0", parameters={})
            for name, model in [("driver", "source"), ("a", "sink"), ("b", "sink")]
        ],
        connections=[
            dict(id=name, **{"from": "driver.out", "to": name + ".inlet"}) for name in ("a", "b")
        ],
    )


def test_signal_direction_unit_and_fanout():
    spec = signal_spec()
    compiled = compile_network(spec, signal_catalog())
    state, rate = compiled.initialize({})
    assert state == pytest.approx([310, 310, 310])
    assert rate == pytest.approx([0, 0, 0])
    with pytest.raises(NetworkError, match="unit"):
        compile_network(spec, signal_catalog("degC"))
    bad = deepcopy(spec)
    bad["connections"][0]["from"], bad["connections"][0]["to"] = (
        bad["connections"][0]["to"],
        bad["connections"][0]["from"],
    )
    with pytest.raises(NetworkError, match="output to input"):
        compile_network(bad, signal_catalog())
    with pytest.raises(NetworkError):
        compiled.route_flux("a", {"value": 10})


@pytest.mark.parametrize("change", ["kind", "medium", "energy"])
def test_hydraulic_mass_energy_compatibility_is_not_inferred(change):
    a = PortSpec(
        "out", "hydraulic", medium="water-if97-region1", energy_convention="total_enthalpy"
    )
    b = replace(a, name="inlet")
    if change == "medium":
        b = replace(b, medium="sodium")
    elif change == "energy":
        b = replace(b, energy_convention="specific_enthalpy_transport")
    else:
        b = PortSpec("inlet", "thermal")

    def fixed(port, model):
        size = len(port.fields)
        return LinearComponentType(
            passport(model),
            (port,),
            (),
            tuple(RowSpec("r" + str(i), "1", 1, 1e-9) for i in range(size)),
            (),
            lambda _: (np.zeros((size, size)), np.eye(size), np.ones(size)),
        )

    catalog = {"source": fixed(a, "source"), "sink": fixed(b, "sink")}
    spec = signal_spec()
    spec["nodes"] = spec["nodes"][:2]
    spec["connections"] = spec["connections"][:1]
    with pytest.raises(NetworkError, match="mismatch"):
        compile_network(spec, catalog)


@pytest.mark.parametrize(
    "matrix,diagnostic", [([[1, 0], [2, 0]], "Structurally"), ([[1, 1], [2, 2]], "Numerically")]
)
def test_structural_and_numerical_singularities_are_distinguished(matrix, diagnostic):
    contract = LinearComponentType(
        passport("singular"),
        (),
        (StateSpec("a", "1", False, 1), StateSpec("b", "1", False, 1)),
        (RowSpec("a", "1", 1, 1e-9), RowSpec("b", "1", 1, 1e-9)),
        (),
        lambda _: (np.zeros((2, 2)), np.array(matrix), np.ones(2)),
    )
    spec = dict(
        schema_version=1,
        nodes=[dict(id="node", model_id="singular", model_version="1.0.0", parameters={})],
        connections=[],
    )
    with pytest.raises(NetworkError, match=diagnostic):
        compile_network(spec, {"singular": contract})


@pytest.mark.parametrize("power,reverse", [(0, False), (0, True), (30, False), (30, True)])
def test_thermal_network_matches_independent_two_capacity_solution(power, reverse):
    spec = example()
    spec["topology"]["nodes"][0]["parameters"]["source_w"] = power
    if reverse:
        spec["initial_temperature_k"] = {"hot": 300, "cold": 330}
    model = LinearThermalNetwork(spec["topology"], spec["initial_temperature_k"])
    initial = spec["initial_temperature_k"]
    lam = 20 * (1 / 1000 + 1 / 2000)
    equilibrium = (power / 1000) / lam
    for end in (1.0, 10.0, 100.0, 1000.0):
        obs = model.advance(end - model.time_s)
        gap = equilibrium + (initial["hot"] - initial["cold"] - equilibrium) * np.exp(-lam * end)
        energy = 1000 * initial["hot"] + 2000 * initial["cold"] + power * end
        expected = dict(hot=(energy + 2000 * gap) / 3000, cold=(energy - 1000 * gap) / 3000)
        assert obs["temperature_k"] == pytest.approx(expected, abs=1e-9)
        assert abs(obs["energy_residual_j"]) < 1e-5
    restored = LinearThermalNetwork.from_snapshot(model.snapshot())
    assert restored.snapshot() == model.snapshot()
    assert restored.advance(10)["temperature_k"] == pytest.approx(
        model.advance(10)["temperature_k"], abs=1e-10
    )


def test_snapshot_detached_and_failed_step_atomic(monkeypatch):
    spec = example()
    model = LinearThermalNetwork(spec["topology"], spec["initial_temperature_k"])
    before = model.snapshot()
    copied = model.snapshot()
    copied["topology"]["nodes"][0]["parameters"]["heat_capacity_j_k"] = 17
    assert model.snapshot() == before

    def fail(*args):
        raise NetworkError("observer failure")

    monkeypatch.setattr(model, "_observe_at", fail)
    with pytest.raises(NetworkError):
        model.advance(10)
    assert model.snapshot() == before


@pytest.mark.parametrize("fault", ["model", "layout", "temperature", "time", "unknown"])
def test_bad_checkpoint_rejected(fault):
    spec = example()
    model = LinearThermalNetwork(spec["topology"], spec["initial_temperature_k"])
    payload = model.snapshot()
    if fault == "model":
        payload["model_version"] = "2.0.0"
    elif fault == "layout":
        payload["assembly_sha256"] = "wrong"
    elif fault == "temperature":
        payload["temperature_k"]["hot"] = -1
    elif fault == "time":
        payload["time_s"] = -1
    else:
        payload["extra"] = True
    with pytest.raises(NetworkError):
        LinearThermalNetwork.from_snapshot(payload)


def test_real_sdk_bundle_and_output_is_exclusive(tmp_path):
    spec = example()
    report = run_network_experiment(spec, tmp_path / "run")
    assert report["numerical_acceptance"] and report["completed_time_s"] == 1000
    with pytest.raises(FileExistsError):
        run_network_experiment(spec, tmp_path / "run")
    payload = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert LinearThermalNetwork.from_snapshot(payload).observation()["time_s"] == 1000


def star():
    nodes = []
    connections = []
    for name in ("a", "b", "c"):
        nodes.append(
            dict(
                id=name,
                model_id="linear-thermal-capacity",
                model_version="1.0.0",
                parameters=dict(heat_capacity_j_k=1000, source_w=0),
            )
        )
        nodes.append(
            dict(
                id="link_" + name,
                model_id="linear-thermal-conductance",
                model_version="1.0.0",
                parameters=dict(ua_w_k=10),
            )
        )
        connections.extend(
            [
                dict(
                    id="capacity_" + name,
                    **{"from": name + ".thermal", "to": "link_" + name + ".left"},
                ),
                dict(
                    id="junction_" + name,
                    **{"from": "link_" + name + ".right", "to": "junction." + name},
                ),
            ]
        )
    nodes.append(
        dict(
            id="junction", model_id="linear-thermal-junction3", model_version="1.0.0", parameters={}
        )
    )
    return dict(schema_version=1, nodes=nodes, connections=connections)


def test_branched_thermal_junction_matches_independent_antisymmetric_mode():
    model = LinearThermalNetwork(star(), {"a": 310, "b": 300, "c": 290})
    assert len(model.compiled.variables) == 27
    for t in (1, 10, 100, 1000):
        obs = model.advance(t - model.time_s)
        amplitude = 10 * np.exp(-0.01 * t)
        assert obs["temperature_k"] == pytest.approx(
            dict(a=300 + amplitude, b=300, c=300 - amplitude), abs=1e-9
        )
        assert obs["variables"]["junction.port.a.temperature_k"] == pytest.approx(300, abs=1e-9)
        assert abs(obs["energy_residual_j"]) < 1e-5


def test_experiment_failure_keeps_last_consistent_prefix(tmp_path, monkeypatch):
    original = LinearThermalNetwork._observe_at

    def limited(self, temperature, time_s):
        if time_s > 150:
            raise NetworkError("synthetic observation failure")
        return original(self, temperature, time_s)

    monkeypatch.setattr(LinearThermalNetwork, "_observe_at", limited)
    report = run_network_experiment(example(), tmp_path / "failed")
    assert report["status"] == "failed" and not report["numerical_acceptance"]
    assert report["completed_time_s"] == 100
    checkpoint = json.loads((tmp_path / "failed/checkpoint.json").read_text())
    assert checkpoint["time_s"] == 100
    restored = LinearThermalNetwork.from_snapshot(checkpoint)
    assert restored.observation()["time_s"] == 100


def test_residual_mismatch_is_dimensionally_diagnosed_and_fixed_state_required():
    compiled = compile_network(example()["topology"], thermal_catalog())
    with pytest.raises(NetworkError):
        compiled.initialize({"hot.state.temperature_k": 330})
    state, rate = compiled.initialize(
        {"hot.state.temperature_k": 330, "cold.state.temperature_k": 300}
    )
    rate[compiled.index["hot.state.temperature_k"]] += 1
    with pytest.raises(NetworkError, match="hot.eq.energy"):
        compiled.require_residual(state, rate)


@pytest.mark.parametrize("kind", ["hydraulic", "mechanical"])
def test_conserved_mass_energy_and_shaft_connector_contracts(kind):
    options = (
        dict(medium="water-if97-region1", energy_convention="total_enthalpy")
        if kind == "hydraulic"
        else {}
    )
    left = PortSpec("out", kind, **options)
    right = replace(left, name="inlet")
    size = len(left.fields)
    source = LinearComponentType(
        passport("source"),
        (left,),
        (),
        tuple(RowSpec("r" + str(i), "1", 1, 1e-9) for i in range(size)),
        (),
        lambda _: (np.zeros((size, size)), np.eye(size), np.arange(1, size + 1, dtype=float)),
    )
    sink = LinearComponentType(
        passport("sink"),
        (right,),
        (),
        (),
        (),
        lambda _: (np.zeros((0, size)), np.zeros((0, size)), np.zeros(0)),
    )
    spec = signal_spec()
    spec["nodes"] = spec["nodes"][:2]
    spec["connections"] = spec["connections"][:1]
    compiled = compile_network(spec, {"source": source, "sink": sink})
    state, rate = compiled.initialize({})
    assert max(abs(compiled.residual(state, rate))) < 1e-9
    flows = {f: 3.0 for f, _, flow, _ in left.fields if flow}
    routed = compiled.route_flux("a", flows)
    for f in flows:
        assert sum(v[f] for v in routed.values()) == 0
    if kind == "mechanical":
        i = compiled.index
        assert (
            state[i["driver.port.out.omega_rad_s"]] * state[i["driver.port.out.torque_nm"]]
            + state[i["a.port.inlet.omega_rad_s"]] * state[i["a.port.inlet.torque_nm"]]
            == 0
        )
    with pytest.raises(NetworkError):
        compiled.route_flux("a", {})
    with pytest.raises(NetworkError):
        compiled.route_flux("a", {f: float("nan") for f in flows})


def test_unbalanced_equation_count_and_undeclared_derivative_rejected():
    base = LinearComponentType(
        passport("bad"),
        (),
        (StateSpec("x", "1", False, 1),),
        (RowSpec("one", "1", 1, 1e-9), RowSpec("two", "1", 1, 1e-9)),
        (),
        lambda _: (np.zeros((2, 1)), np.ones((2, 1)), np.ones(2)),
    )
    spec = dict(
        schema_version=1,
        nodes=[dict(id="node", model_id="bad", model_version="1.0.0", parameters={})],
        connections=[],
    )
    with pytest.raises(NetworkError, match="not square"):
        compile_network(spec, {"bad": base})
    base = replace(
        base, rows=(base.rows[0],), build=lambda _: (np.ones((1, 1)), np.ones((1, 1)), np.ones(1))
    )
    with pytest.raises(NetworkError, match="undeclared derivatives"):
        compile_network(spec, {"bad": base})


def test_regular_ode_is_not_rejected_by_cancelled_mass_plus_stiffness():
    contract = LinearComponentType(
        passport("growth"),
        (),
        (StateSpec("x", "1", True, 1),),
        (RowSpec("evolution", "1/s", 1, 1e-9),),
        (),
        lambda _: (np.ones((1, 1)), -np.ones((1, 1)), np.zeros(1)),
    )
    spec = dict(
        schema_version=1,
        nodes=[dict(id="node", model_id="growth", model_version="1.0.0", parameters={})],
        connections=[],
    )
    compiled = compile_network(spec, {"growth": contract})
    state, rate = compiled.initialize({"node.state.x": 1})
    assert state == pytest.approx([1]) and rate == pytest.approx([1])


def test_unrepresentable_time_step_is_rejected_atomically():
    spec = example()
    model = LinearThermalNetwork(spec["topology"], spec["initial_temperature_k"])
    model.advance(1)
    before = model.snapshot()
    with pytest.raises(NetworkError):
        model.advance(1e-30)
    assert model.snapshot() == before


def test_component_and_connection_equation_namespace_collision_rejected():
    catalog = signal_catalog()
    catalog["source"] = replace(catalog["source"], rows=(RowSpec("value", "K", 300, 1e-9),))
    spec = dict(
        schema_version=1,
        nodes=[
            dict(id="connection", model_id="source", model_version="1.0.0", parameters={}),
            dict(id="sink", model_id="sink", model_version="1.0.0", parameters={}),
        ],
        connections=[dict(id="eq", **{"from": "connection.out", "to": "sink.inlet"})],
    )
    with pytest.raises(NetworkError, match="Duplicate global equation"):
        compile_network(spec, catalog)
