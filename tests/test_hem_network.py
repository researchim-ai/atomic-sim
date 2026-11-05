"""Conservation and transactional guards for supported finite-plenum graphs."""

import json
from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem_network.cases import baseline, matrix
from atomic_model_hem_network.model import GraphHEMNetwork
from atomic_model_workspace.common import digest
from physics.materials import iapws


def test_branched_rest_preserves_stocks_with_supported_pressure_forces():
    m = GraphHEMNetwork(baseline())
    before = m.snapshot()
    after = m.advance(0.001)
    for a, b in zip(before["cells"], after["cells"]):
        assert a["mass_kg"] == b["mass_kg"]
        assert a["total_energy_j"] == b["total_energy_j"]
        assert b["momentum_kg_m_s"] == 0
    assert max(abs(x) for x in after["local_ledgers"]["momentum_residual_kg_m_s"]) < 1e-10
    assert any(abs(x) > 0 for x in after["local_ledgers"]["node_support_impulse_kg_m_s"])
    assert "no global vector" in m.contract()["momentum_scope"]


def test_graph_face_flux_is_single_shared_physical_transport():
    spec = dict(matrix())["branch_pressure"]
    m = GraphHEMNetwork(spec)
    rate, aux = m.evaluate(m._y, m._controls)
    m0, P, C, v, states, _, _ = m.primitive(m._y)
    j = 0
    a, b = m.faces[j]
    area = m.face_areas[j]
    U = [np.array([m0[i], P[i], C[i] + m.eref * m0[i]]) / m.volumes[i] for i in [a, b]]
    F = [
        np.array([u[1], u[1] * v[i] + states[i].pressure_pa, v[i] * (u[2] + states[i].pressure_pa)])
        for u, i in zip(U, [a, b])
    ]
    expected = area * ((F[0] + F[1]) / 2 - m.wave * (U[1] - U[0]) / 2)
    actual = rate[m.face_slice].reshape(3, m.face_count)[:, j]
    # Absolute-energy oracle subtracts ~1e8 J/m³ states; the shifted
    # production coordinates avoid that cancellation.
    assert actual == pytest.approx(expected, abs=1e-7)
    assert sum(rate[: m.n]) == pytest.approx(0, abs=1e-14)
    assert sum(rate[2 * m.n : 3 * m.n]) + m.eref * sum(rate[: m.n]) + sum(
        m.capacity * rate[m.wall_slice]
    ) == pytest.approx(0, abs=1e-8)
    assert aux["shared_flux"].shape == (3, m.face_count)


def test_tiny_high_degree_node_uses_sum_of_incident_areas_for_cfl():
    spec = baseline()
    spec["nodes"][1]["volume_m3"] = 1e-5
    spec["max_step_s"] = 1e-3
    m = GraphHEMNetwork(spec)
    incident = sum(area for (a, b), area in zip(m.faces, m.face_areas) if 1 in (a, b))
    expected = m.cfl * 1e-5 / incident / m.wave
    assert m.advance(1e-6)["maximum_step_s"] == pytest.approx(expected)


@pytest.mark.parametrize("velocity", [0.001, -0.001])
def test_darcy_removes_momentum_but_retains_total_energy(velocity):
    spec = dict(matrix())["laminar_drag"]
    for e in spec["edges"]:
        for p in e["initial"]:
            p["velocity_m_s"] = velocity
    drag = GraphHEMNetwork(spec)
    plain = deepcopy(spec)
    for e in plain["edges"]:
        e["friction"] = "none"
        del e["friction_scope"]
    inviscid = GraphHEMNetwork(plain)
    rd, _ = drag.evaluate(drag._y, drag._controls)
    ri, _ = inviscid.evaluate(inviscid._y, inviscid._controls)
    i = drag.edge_indices[0][0]
    state = drag.states0[i]
    mu = iapws.viscosity_pa_s(state.density_kg_m3, state.temperature_k)
    expected = -32 * mu * drag.volumes[i] * velocity / spec["edges"][0]["diameter_m"] ** 2
    force = rd[drag.n + i] - ri[drag.n + i]
    assert force == pytest.approx(expected, rel=1e-9)
    assert (rd - ri)[2 * drag.n : 3 * drag.n] == pytest.approx(np.zeros(drag.n))
    assert -velocity * force > 0
    assert rd[drag.drag_entropy_index] > 0
    assert drag.advance(0.001)["global_ledger"]["energy_residual_j"] == pytest.approx(0, abs=1e-8)


def test_reverse_orientation_preserves_scalar_physics():
    spec = dict(matrix())["branch_pressure"]
    rev = deepcopy(spec)
    for e in rev["edges"]:
        e["from_node"], e["to_node"] = e["to_node"], e["from_node"]
        e["initial"].reverse()
        e["heat_power_w"].reverse()
        for p in e["initial"]:
            p["velocity_m_s"] *= -1
    for w in rev["walls"]:
        edge, cell = w["cell"].split("/")
        w["cell"] = edge + "/" + str(1 - int(cell))
    a = GraphHEMNetwork(spec).advance(0.001)
    b = GraphHEMNetwork(rev).advance(0.001)
    lookup = {c["id"]: c for c in b["cells"]}
    for cell in a["cells"]:
        name = cell["id"]
        target = (
            name if "/" not in name else name.split("/")[0] + "/" + str(1 - int(name.split("/")[1]))
        )
        other = lookup[target]
        assert cell["pressure_pa"] == pytest.approx(other["pressure_pa"], abs=1e-5)
        assert cell["mass_kg"] == pytest.approx(other["mass_kg"], abs=1e-12)
        assert cell["momentum_kg_m_s"] == pytest.approx(-other["momentum_kg_m_s"], abs=1e-10)


def test_joint_node_and_multiple_wall_heat_closes_locally():
    spec = baseline()
    w = spec["walls"][0]
    w["cell"] = "b"
    w["initial_temperature_k"] = 400
    w2 = deepcopy(w)
    w2["id"] = "w1"
    w2["initial_temperature_k"] = 320
    spec["walls"].append(w2)
    m = GraphHEMNetwork(spec)
    o = m.advance(0.001)
    assert o["walls"][0]["heat_to_fluid_j"] > 0
    assert o["walls"][1]["heat_to_fluid_j"] < 0
    assert o["global_ledger"]["energy_residual_j"] == pytest.approx(0, abs=1e-8)
    assert max(abs(x) for x in o["local_ledgers"]["energy_residual_j"]) < 1e-8


def test_online_power_checkpoint_and_pure_residual():
    spec = baseline("loop")
    m = GraphHEMNetwork(spec)
    controls = m.control()
    controls["fluid"]["a"] = 1000
    controls["walls"]["w0"] = 2000
    m.step(controls, 0.001, expected_revision=0, controller_state={"tick": 1})
    cp = json.loads(json.dumps(m.checkpoint()))
    r = GraphHEMNetwork(spec)
    r.restore(cp)
    assert r.advance(0.001) == m.advance(0.001)
    o = m.snapshot()
    assert o["global_ledger"]["external_fluid_heat_j"] == pytest.approx(2)
    assert o["global_ledger"]["external_wall_heat_j"] == pytest.approx(4)
    before = m.checkpoint()
    rate = m.evaluate(m._y, m._controls)[0]
    assert m.residual(m._y, rate) == pytest.approx(np.zeros(m.size))
    assert m.checkpoint() == before
    with pytest.raises(ValueError, match="Stale"):
        m.step(controls, 0.001, expected_revision=0, controller_state=None)
    assert m.checkpoint() == before


@pytest.mark.parametrize("damage", ["coordinates", "calls", "observation", "source"])
def test_rehashed_checkpoint_corruption_refused(damage):
    spec = baseline()
    m = GraphHEMNetwork(spec)
    m.advance(0.0002)
    cp = m.checkpoint()
    if damage == "coordinates":
        cp["payload"]["coordinates"][0] += 0.01
    elif damage == "calls":
        cp["payload"]["calls"][0]["duration_s"] *= 2
    elif damage == "observation":
        cp["payload"]["observation"]["time_s"] += 0.01
    else:
        cp["payload"]["source"] = {}
    cp["sha256"] = digest(cp["payload"])
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.restore(cp)
    assert m.checkpoint() == before


@pytest.mark.parametrize("ledger", ["face", "drag", "support", "wall"])
def test_local_ledger_damage_cannot_hide_in_global_balance(ledger):
    m = GraphHEMNetwork(baseline())
    y = m._y.copy()
    if ledger == "face":
        y[m.face_slice.start] += 0.01
        y[m.face_slice.start + 1] -= 0.01
    elif ledger == "drag":
        y[m.drag_slice.start + m.node_count] += 0.1
    elif ledger == "support":
        y[m.support_slice.start] += 0.1
    else:
        y[m.wall_exchange_slice.start] += 1
    with pytest.raises(ValueError, match="conservation"):
        m._observe(y, 0, m._controls, None, 0)


def test_intermediate_failure_rolls_back_graph_and_controller(monkeypatch):
    m = GraphHEMNetwork(baseline())
    before = m.checkpoint()
    original = m._evaluate_point
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("Injected stage failure")
        return original(*args)

    monkeypatch.setattr(m, "_evaluate_point", fail)
    with pytest.raises(ValueError, match="Injected"):
        m.step(m.control(), 0.001, expected_revision=0, controller_state={"tick": 1})
    assert m.checkpoint() == before


@pytest.mark.parametrize(
    "bad",
    [
        "duplicate",
        "isolated",
        "self",
        "node_velocity",
        "diameter",
        "missing_scope",
        "false_scope",
        "two_phase_drag",
        "transition_drag",
    ],
)
def test_invalid_topology_or_closure_scope_rejected(bad):
    spec = baseline()
    if bad == "duplicate":
        spec["nodes"][1]["id"] = "a"
    elif bad == "isolated":
        spec["nodes"].append(
            dict(
                id="z",
                volume_m3=0.01,
                initial=deepcopy(spec["nodes"][0]["initial"]),
                heat_power_w=0,
            )
        )
    elif bad == "self":
        spec["edges"][0]["to_node"] = "a"
    elif bad == "node_velocity":
        spec["nodes"][0]["initial"]["velocity_m_s"] = 0.001
    elif bad == "diameter":
        spec["edges"][0]["diameter_m"] = 0
    else:
        spec = dict(matrix())["laminar_drag"]
        if bad == "missing_scope":
            del spec["edges"][0]["friction_scope"]
        elif bad == "false_scope":
            spec["edges"][0]["friction_scope"]["fully_developed"] = False
        elif bad == "two_phase_drag":
            spec = baseline("parallel", "mixture")
            spec["edges"][0]["friction"] = "darcy"
            spec["edges"][0]["friction_scope"] = dict(matrix())["laminar_drag"]["edges"][0][
                "friction_scope"
            ]
        else:
            for p in spec["edges"][0]["initial"]:
                p["velocity_m_s"] = 0.025
    with pytest.raises((ValueError, RuntimeError)):
        GraphHEMNetwork(spec)


@pytest.mark.parametrize("name", ["laminar_heat_transfer", "turbulent_heat_transfer"])
def test_empirical_wall_exchange_uses_actual_h_and_conservative_bound(name):
    spec = dict(matrix())[name]
    m = GraphHEMNetwork(spec)
    p = m.primitive(m._y)
    actual = m._thermal_conductance(p)[0]
    assert 0 < actual < m.conductance[0]
    rate, aux = m.evaluate(m._y, m._controls)
    expected = actual * (p[5][0] - p[4][m.ports[0]].temperature_k)
    assert aux["wall_to_fluid_w"][0] == pytest.approx(expected)
    assert m.capacity[0] * rate[m.wall_slice][0] == pytest.approx(-expected)
    assert m.advance(0.0001)["global_ledger"]["energy_residual_j"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("bad", ["bound", "saturation", "node", "scope", "zero_flow", "transition"])
def test_empirical_heat_transfer_rejects_unsupported_regimes(bad):
    spec = dict(matrix())["laminar_heat_transfer"]
    if bad == "bound":
        spec["walls"][0]["conductance_w_k"] = 0.001
    elif bad == "saturation":
        spec["walls"][0]["initial_temperature_k"] = 550.0
    elif bad == "node":
        spec["walls"][0]["cell"] = "a"
    elif bad == "scope":
        spec["walls"][0]["correlation"]["thermally_developed"] = False
    elif bad == "zero_flow":
        spec["edges"][0]["initial"][0]["velocity_m_s"] = 0
    else:
        state = iapws.properties_pt(3e6, 350.0)
        spec["edges"][0]["initial"][0]["velocity_m_s"] = (
            3000 * state.viscosity_pa_s / (state.rho_kg_m3 * 0.05)
        )
    with pytest.raises((ValueError, RuntimeError)):
        GraphHEMNetwork(spec)


@pytest.mark.parametrize("velocity", [2.0, -2.0])
def test_initialization_does_not_impose_temporary_reflecting_end_conditions(velocity):
    spec = baseline("parallel")
    for edge in spec["edges"]:
        for point in edge["initial"]:
            point["velocity_m_s"] = velocity
    model = GraphHEMNetwork(spec)
    observed = model.advance(1e-6)
    assert observed["global_ledger"]["energy_residual_j"] == pytest.approx(0, abs=1e-8)
    assert max(abs(x) for x in observed["local_ledgers"]["momentum_residual_kg_m_s"]) < 1e-8
