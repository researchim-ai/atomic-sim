"""Independent graph fixtures and absolute-stock numerical reference checks."""

import importlib
import importlib.util
import json
from copy import deepcopy
from math import pi

import numpy as np
import pytest

from atomic_model_hem import eos
from atomic_model_hem_network.reference import Assembly, solve


def fixture_spec(topology="branch", pulse=False, walls=False):
    """Fixtures intentionally do not import production cases or parsing."""
    point = dict(kind="pt", pressure_pa=1e6, temperature_k=350.0, velocity_m_s=0.0)
    links = (
        [("a", "b"), ("b", "c"), ("b", "d")]
        if topology == "branch"
        else [("a", "b"), ("b", "c"), ("c", "a")]
    )
    names = ["a", "b", "c", "d"] if topology == "branch" else ["a", "b", "c"]
    nodes = [
        dict(id=name, volume_m3=0.025, initial=deepcopy(point), heat_power_w=0.0) for name in names
    ]
    if pulse:
        nodes[0]["initial"]["pressure_pa"] += 100.0
    edges = [
        dict(
            id="e" + str(i),
            from_node=a,
            to_node=b,
            cells=2,
            length_m=1.0,
            diameter_m=0.1,
            roughness_m=1e-5,
            friction="none",
            initial=[deepcopy(point) for _ in range(2)],
            heat_power_w=[0.0, 0.0],
        )
        for i, (a, b) in enumerate(links)
    ]
    thermal = (
        [
            dict(
                id="w" + str(i),
                cell=cell,
                capacity_j_k=10000.0,
                initial_temperature_k=351.0,
                conductance_w_k=50.0,
                heat_power_w=10.0,
                ambient_ua_w_k=2.0,
                ambient_temperature_k=349.0,
            )
            for i, cell in enumerate(["b", "e0/0"])
        ]
        if walls
        else [
            dict(
                id="w0",
                cell="b",
                capacity_j_k=10000.0,
                initial_temperature_k=350.0,
                conductance_w_k=0.0,
                heat_power_w=0.0,
                ambient_ua_w_k=0.0,
                ambient_temperature_k=350.0,
            )
        ]
    )
    return dict(
        schema_version=1,
        nodes=nodes,
        edges=edges,
        walls=thermal,
        wave_speed_bound_m_s=2000.0,
        max_step_s=2e-5,
        cfl=0.05,
    )


def program(duration=0.0004):
    return [dict(duration_s=duration, command=None)]


def model_class():
    # Reference-only checks remain runnable while the parent creates model.py.
    if importlib.util.find_spec("atomic_model_hem_network.model") is None:
        pytest.skip("GraphHEMNetwork implementation not yet available")
    return importlib.import_module("atomic_model_hem_network.model").GraphHEMNetwork


def stocks(observation):
    cells = {cell["id"]: cell for cell in observation["cells"]}
    return cells


@pytest.mark.parametrize("topology", ["branch", "loop"])
def test_reference_rest_and_signed_support(topology):
    assembly = Assembly(fixture_spec(topology))
    rhs = assembly.rhs(0.0, assembly.y0)
    np.testing.assert_allclose(rhs[: 3 * assembly.n], 0.0, atol=1e-8)
    assert rhs[assembly.tail] == pytest.approx(0.0, abs=1e-8)
    p = eos.pt(1e6, 350.0).pressure_pa
    expected = np.tile([0.0, pi * 0.1**2 / 4 * p, 0.0], len(assembly.faces))
    np.testing.assert_allclose(rhs[assembly.face_start : assembly.tail], expected, atol=1e-8)
    result = solve(assembly.spec, program())[-1]
    for a, b in zip(assembly.observation(0, assembly.y0)["cells"], result["cells"]):
        assert b["mass_kg"] == pytest.approx(a["mass_kg"], abs=1e-12)
        assert b["total_energy_j"] == pytest.approx(a["total_energy_j"], abs=1e-7)


def test_reference_single_interface_hand_calculation_and_local_incidence():
    assembly = Assembly(fixture_spec(pulse=True))
    y = assembly.y0.copy()
    # Nonzero channel velocity makes pressure work and kinetic energy observable.
    i = assembly.nodes
    y[assembly.n + i] = y[i] * 0.01
    y[2 * assembly.n + i] += 0.5 * y[i] * 0.01**2
    rhs = assembly.rhs(0.0, y)
    node = eos.pt(1.0001e6, 350.0)
    channel = eos.pt(1e6, 350.0)
    ul = np.array([node.density_kg_m3, 0, node.density_kg_m3 * node.internal_energy_j_kg])
    ur = np.array(
        [
            channel.density_kg_m3,
            channel.density_kg_m3 * 0.01,
            channel.density_kg_m3 * (channel.internal_energy_j_kg + 0.01**2 / 2),
        ]
    )
    fl = np.array([0, node.pressure_pa, 0])
    fr = np.array([ur[1], ur[1] * 0.01 + channel.pressure_pa, 0.01 * (ur[2] + channel.pressure_pa)])
    expected = pi * 0.1**2 / 4 * ((fl + fr) / 2 - 2000 * (ur - ul) / 2)
    actual = rhs[assembly.face_start : assembly.face_start + 3]
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-6)
    # Recover every cell's stock rates solely from independently scattered ledgers.
    reconstructed = np.zeros((3, assembly.n))
    suppressed = 0.0
    for k, (left, right, _) in enumerate(assembly.faces):
        flux = rhs[assembly.face_start + 3 * k : assembly.face_start + 3 * k + 3]
        for cell, sign in [(left, -1), (right, 1)]:
            reconstructed[:, cell] += sign * flux
            if cell < assembly.nodes:
                suppressed += sign * flux[1]
                reconstructed[1, cell] = 0
    np.testing.assert_allclose(rhs[: 3 * assembly.n].reshape(3, -1), reconstructed, atol=1e-8)
    assert rhs[assembly.tail] == pytest.approx(-suppressed)
    assert sum(rhs[assembly.n : 2 * assembly.n]) - rhs[assembly.tail] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("topology", ["branch", "loop"])
def test_reference_methods_agree_and_close_energy(topology):
    spec = fixture_spec(topology, pulse=True, walls=True)
    a = solve(spec, program(), "DOP853")[-1]
    b = solve(spec, program(), "Radau")[-1]
    initial = solve(spec, [])[-1]
    for ca, cb in zip(a["cells"], b["cells"]):
        for key, atol in [("mass_kg", 1e-9), ("momentum_kg_m_s", 1e-8), ("total_energy_j", 1e-4)]:
            assert ca[key] == pytest.approx(cb[key], rel=2e-10, abs=atol)
    assert sum(c["mass_kg"] for c in a["cells"]) == pytest.approx(
        sum(c["mass_kg"] for c in initial["cells"]), abs=1e-10
    )
    delta = sum(
        c["total_energy_j"] - z["total_energy_j"] for c, z in zip(a["cells"], initial["cells"])
    )
    delta += sum(w["energy_change_j"] for w in a["walls"])
    ledger = a["global_ledger"]
    assert delta == pytest.approx(
        ledger["external_fluid_heat_j"] + ledger["external_wall_heat_j"], abs=2e-6
    )
    assert ledger["thermal_entropy_production_j_k"] > 0
    assert all(c["momentum_kg_m_s"] == 0 for c in a["cells"][: len(spec["nodes"])])
    assert abs(stocks(a)["a"]["mass_kg"] - stocks(initial)["a"]["mass_kg"]) > 1e-9
    if topology == "branch":
        assert stocks(a)["c"]["mass_kg"] == pytest.approx(stocks(a)["d"]["mass_kg"], abs=1e-11)


def test_reference_control_change_and_friction_scope():
    spec = fixture_spec(walls=True)
    assembly = Assembly(spec)
    controls = deepcopy(assembly.controls)
    controls["fluid"]["b"] = 100.0
    controls["walls"]["w0"] = 20.0
    calls = program(0.0001) + [dict(duration_s=0.0001, command=dict(controls=controls))]
    observed = solve(spec, calls)
    assert observed[-1]["global_ledger"]["external_fluid_heat_j"] == pytest.approx(0.01)
    assert assembly.controls["fluid"]["b"] == 0  # detached spec/control state
    spec["edges"][0]["friction"] = "darcy"
    with pytest.raises(ValueError, match="scope"):
        solve(spec, program(0.0001))


@pytest.mark.parametrize("topology", ["branch", "loop"])
def test_model_branched_and_loop_rest(topology):
    model = model_class()(fixture_spec(topology))
    before = stocks(model.snapshot())
    after = model.advance(0.0004)
    for name, cell in stocks(after).items():
        for key in ["mass_kg", "total_energy_j"]:
            assert cell[key] == pytest.approx(before[name][key], rel=0, abs=1e-7)
    assert abs(after["global_ledger"]["mass_residual_kg"]) < 1e-9
    assert abs(after["global_ledger"]["energy_residual_j"]) < 1e-5


@pytest.mark.parametrize("topology", ["branch", "loop"])
def test_model_pulse_temporal_reference(topology):
    cls = model_class()
    spec = fixture_spec(topology, pulse=True, walls=True)
    references = solve(spec, program(0.001) * 3)
    reference = stocks(references[-1])
    errors = []
    for cap in [1e-5, 5e-6]:
        request = deepcopy(spec)
        request["max_step_s"] = cap
        # Keep CFL from masking max_step refinement.
        request["cfl"] = 0.1
        model = cls(request)
        start = stocks(model.snapshot())
        for ref in references[1:]:
            result = model.advance(0.001)
            actual = stocks(result)
            for key in [
                "integrated_face_mass_kg",
                "integrated_face_energy_j",
                "integrated_face_momentum_kg_m_s",
                "node_support_impulse_kg_m_s",
            ]:
                np.testing.assert_allclose(
                    result["local_ledgers"][key], ref["local_ledgers"][key], rtol=1e-5, atol=1e-5
                )
            for name, expected in stocks(ref).items():
                assert actual[name]["density_kg_m3"] == pytest.approx(
                    expected["density_kg_m3"], rel=1e-8, abs=0
                )
                assert actual[name]["pressure_pa"] == pytest.approx(
                    expected["pressure_pa"], rel=0, abs=0.1
                )
        actual = stocks(result)
        error = []
        for name in reference:
            for key, floor in [
                ("mass_kg", 1e-7),
                ("momentum_kg_m_s", 1e-5),
                ("total_energy_j", 0.1),
            ]:
                scale = max(abs(reference[name][key] - start[name][key]), floor)
                error.append((actual[name][key] - reference[name][key]) / scale)
        errors.append(float(np.linalg.norm(error)))
        assert abs(result["global_ledger"]["mass_residual_kg"]) < 1e-9
        assert abs(result["global_ledger"]["energy_residual_j"]) < 1e-5
    assert errors[-1] < 2e-3
    # At these small caps absolute-stock subtraction and IF97 inversion
    # dominate; require improvement only above that measured precision floor.
    assert errors[-1] < max(errors[0] / 4, 3e-7), errors


@pytest.mark.parametrize("topology", ["branch", "loop"])
def test_model_restart_during_pulse_and_wall_exchange(topology):
    cls = model_class()
    spec = fixture_spec(topology, pulse=True, walls=True)
    model = cls(spec)
    model.advance(0.0002)
    checkpoint = json.loads(json.dumps(model.checkpoint(), allow_nan=False))
    restored = cls(spec)
    assert restored.restore(checkpoint) == model.snapshot()
    assert restored.advance(0.0002) == model.advance(0.0002)


def reverse_edges(spec):
    result = deepcopy(spec)
    for edge in result["edges"]:
        edge["from_node"], edge["to_node"] = edge["to_node"], edge["from_node"]
        edge["initial"].reverse()
        edge["heat_power_w"].reverse()
        for point in edge["initial"]:
            point["velocity_m_s"] *= -1
    for wall in result["walls"]:
        if "/" in wall["cell"]:
            name, index = wall["cell"].split("/")
            edge = next(e for e in result["edges"] if e["id"] == name)
            wall["cell"] = name + "/" + str(edge["cells"] - 1 - int(index))
    return result


@pytest.mark.parametrize("implementation", ["reference", "model"])
def test_scalar_orientation_invariance_with_velocity_sign_flip(implementation):
    spec = fixture_spec(pulse=True, walls=True)
    spec["edges"][0]["initial"][0]["velocity_m_s"] = 0.002
    spec["edges"][0]["initial"][1]["velocity_m_s"] = -0.001
    reversed_spec = reverse_edges(spec)
    if implementation == "reference":
        a, b = [solve(s, program(0.001))[-1] for s in [spec, reversed_spec]]
    else:
        cls = model_class()
        a, b = [cls(s).advance(0.001) for s in [spec, reversed_spec]]
    left, right = stocks(a), stocks(b)
    for name, cell in left.items():
        mapped = name
        if "/" in name:
            edge, index = name.split("/")
            count = next(e["cells"] for e in spec["edges"] if e["id"] == edge)
            mapped = edge + "/" + str(count - 1 - int(index))
        other = right[mapped]
        for key, atol in [
            ("mass_kg", 1e-9),
            ("total_energy_j", 1e-4),
            ("pressure_pa", 0.1),
            ("density_kg_m3", 1e-7),
        ]:
            assert cell[key] == pytest.approx(other[key], rel=1e-10, abs=atol)
        assert cell["momentum_kg_m_s"] == pytest.approx(-other["momentum_kg_m_s"], abs=1e-8)
    for wa, wb in zip(a["walls"], b["walls"]):
        assert wa["temperature_k"] == pytest.approx(wb["temperature_k"], abs=1e-9)
    for key in [
        "integrated_face_mass_kg",
        "integrated_face_momentum_kg_m_s",
        "integrated_face_energy_j",
    ]:
        offset = 0
        sign = 1 if key == "integrated_face_momentum_kg_m_s" else -1
        for edge in spec["edges"]:
            stop = offset + edge["cells"] + 1
            np.testing.assert_allclose(
                a["local_ledgers"][key][offset:stop],
                sign * np.array(b["local_ledgers"][key][offset:stop][::-1]),
                rtol=1e-8,
                atol=1e-7,
            )
            offset = stop


@pytest.mark.parametrize(
    "re,rr", [(100.0, 0.0), (2299.0, 0.01), (4000.0, 0.0), (1e5, 0.001), (1e8, 0.05)]
)
def test_independent_darcy_factor_equation(re, rr):
    from math import log10, sqrt

    factor = Assembly.darcy_factor(re, rr)
    if re < 2300:
        assert factor == 64 / re
    else:
        assert 1 / sqrt(factor) + 2 * log10(rr / 3.7 + 2.51 / (re * sqrt(factor))) == pytest.approx(
            0, abs=1e-11
        )


@pytest.mark.parametrize("re", [0.0, 2300.0, 3999.0, 1e8 + 1])
def test_independent_darcy_refuses_unsupported_re(re):
    with pytest.raises(ValueError, match="Darcy"):
        Assembly.darcy_factor(re, 0.0)


def test_darcy_scope_at_rest_and_energy_preserving_drag():
    from atomic_model_hem_network.cases import matrix

    spec = dict(matrix())["laminar_drag"]
    assembly = Assembly(spec)
    y = assembly.y0.copy()
    rhs = assembly.rhs(0, y)
    undamped = deepcopy(spec)
    for edge in undamped["edges"]:
        edge["friction"] = "none"
        edge.pop("friction_scope")
    other = Assembly(undamped)
    rhs0 = other.rhs(0, other.y0)
    np.testing.assert_array_equal(rhs[: assembly.n], rhs0[: assembly.n])
    np.testing.assert_array_equal(
        rhs[2 * assembly.n : 3 * assembly.n], rhs0[2 * assembly.n : 3 * assembly.n]
    )
    assert rhs[assembly.tail + 5] > 0
    for i, edge in assembly.drag_cells:
        force = rhs[assembly.tail + 6 + i]
        assert force < 0
        assert rhs[assembly.n + i] - rhs0[assembly.n + i] == pytest.approx(force, abs=1e-10)
        # Laminar force is linear in velocity, including its exact zero limit.
        state = assembly.thermos(y)[i]
        from physics.materials import iapws

        mu = iapws.viscosity_pa_s(state.density_kg_m3, state.temperature_k)
        expected = (
            -32 * mu * assembly.volumes[i] * (y[assembly.n + i] / y[i]) / edge["diameter_m"] ** 2
        )
        assert force == pytest.approx(expected, rel=1e-12)
        y[assembly.n + i] = 0
    assert all(
        assembly.drag_force(i, e, y, assembly.thermos(y)[i]) == 0 for i, e in assembly.drag_cells
    )
    for field in spec["edges"][0]["friction_scope"]:
        invalid = deepcopy(spec)
        invalid["edges"][0]["friction_scope"][field] = 1
        bad = Assembly(invalid)
        bad.y0[bad.n : 2 * bad.n] = 0
        with pytest.raises(ValueError, match="scope"):
            bad.rhs(0, bad.y0)


@pytest.mark.parametrize(
    "name",
    [
        "branch_rest",
        "branch_pressure",
        "unequal_branches",
        "loop_pressure",
        "parallel_channels",
        "laminar_drag",
        "mixture_branch",
        "wall_exchange",
        "ambient_source",
    ],
)
def test_case_matrix_reference_api_and_model(name):
    from atomic_model_hem_network.cases import matrix

    spec = dict(matrix())[name]
    calls = program(0.001)
    a = solve(spec, calls, "DOP853")
    b = solve(spec, calls, "Radau")
    assert len(a) == len(calls) + 1
    assert len(a[-1]["wall_temperatures_k"]) == len(spec["walls"])
    assert isinstance(a[-1]["global_integrals"], dict)
    result = model_class()(spec).advance(0.001)
    for expected, independent in zip(a[-1]["cells"], b[-1]["cells"]):
        actual = stocks(result)[expected["id"]]
        for key in [
            "pressure_pa",
            "temperature_k",
            "density_kg_m3",
            "total_energy_j",
            "mass_kg",
            "momentum_kg_m_s",
        ]:
            assert key in expected
        assert expected["pressure_pa"] == pytest.approx(independent["pressure_pa"], abs=0.1, rel=0)
        assert actual["pressure_pa"] == pytest.approx(expected["pressure_pa"], abs=0.1, rel=0)
        assert actual["density_kg_m3"] == pytest.approx(expected["density_kg_m3"], rel=1e-8)
    np.testing.assert_allclose(
        [w["temperature_k"] for w in result["walls"]],
        a[-1]["wall_temperatures_k"],
        atol=1e-7,
        rtol=0,
    )
    for key in [
        "external_fluid_heat_j",
        "external_wall_heat_j",
        "external_entropy_j_k",
        "thermal_entropy_production_j_k",
        "drag_entropy_production_j_k",
    ]:
        assert result["global_ledger"][key] == pytest.approx(
            a[-1]["global_integrals"][key], rel=1e-5, abs=1e-7
        )


def correlated_spec(turbulent=False):
    from atomic_model_hem_network.cases import baseline

    spec = baseline("parallel")
    for edge in spec["edges"]:
        for point in edge["initial"]:
            point["velocity_m_s"] = 1.0 if turbulent else 0.001
    wall = spec["walls"][0]
    wall.update(
        initial_temperature_k=400.0,
        conductance_w_k=1e5 if turbulent else 100.0,
        correlation=dict(
            scope={
                key: True
                for key in [
                    "circular_straight",
                    "fully_developed",
                    "constant_properties",
                    "forced_convection",
                    "negligible_compressibility",
                ]
            },
            constant_wall_temperature=True,
            thermally_developed=True,
        ),
    )
    spec["max_step_s"] = 1e-6 if turbulent else 5e-6
    return spec


@pytest.mark.parametrize("turbulent", [False, True])
def test_independent_correlated_wall_formula_and_bound(turbulent):
    from math import log10, pi, sqrt

    from physics.materials import iapws

    spec = correlated_spec(turbulent)
    assembly = Assembly(spec)
    wall = spec["walls"][0]
    cell = assembly.names.index(wall["cell"])
    state = assembly.thermos(assembly.y0)[cell]
    bulk = iapws.properties_pt(state.pressure_pa, state.temperature_k)
    edge = spec["edges"][0]
    re = bulk.rho_kg_m3 * (1.0 if turbulent else 0.001) * edge["diameter_m"] / bulk.viscosity_pa_s
    pr = bulk.cp_j_kg_k * bulk.viscosity_pa_s / bulk.conductivity_w_m_k
    if turbulent:
        f = (1.82 * log10(re) - 1.64) ** -2
        nu = (f / 8) * (re - 1000) * pr / (1 + 12.7 * sqrt(f / 8) * (pr ** (2 / 3) - 1))
    else:
        nu = 3.66
    expected = nu * bulk.conductivity_w_m_k * pi * edge["length_m"] / edge["cells"]
    actual = assembly.wall_conductance(wall, assembly.y0, state)
    assert actual == pytest.approx(expected, rel=1e-12)
    rhs = assembly.rhs(0, assembly.y0)
    exchange = actual * (400 - state.temperature_k)
    assert rhs[assembly.wall_start] == pytest.approx(-exchange)
    wall["conductance_w_k"] = actual / 2
    with pytest.raises(ValueError, match="bound"):
        assembly.wall_conductance(wall, assembly.y0, state)
    wall["cell"] = "a"
    with pytest.raises(ValueError, match="channel"):
        assembly.wall_conductance(wall, assembly.y0, state)


@pytest.mark.parametrize("turbulent", [False, True])
def test_correlated_wall_reference_temporal_match(turbulent):
    spec = correlated_spec(turbulent)
    refs = solve(spec, program(0.001), "DOP853")
    other = solve(spec, program(0.001), "Radau")[-1]
    model = model_class()(spec)
    result = model.advance(0.001)
    for cell in refs[-1]["cells"]:
        actual = stocks(result)[cell["id"]]
        independent = stocks(other)[cell["id"]]
        assert cell["pressure_pa"] == pytest.approx(independent["pressure_pa"], abs=0.1, rel=0)
        assert actual["pressure_pa"] == pytest.approx(cell["pressure_pa"], abs=0.1, rel=0)
        assert actual["density_kg_m3"] == pytest.approx(cell["density_kg_m3"], rel=1e-8)
    np.testing.assert_allclose(
        [w["temperature_k"] for w in result["walls"]],
        refs[-1]["wall_temperatures_k"],
        atol=1e-7,
        rtol=0,
    )
    assert result["global_ledger"]["thermal_entropy_production_j_k"] == pytest.approx(
        refs[-1]["global_integrals"]["thermal_entropy_production_j_k"], rel=1e-5, abs=1e-8
    )
    assert abs(result["global_ledger"]["energy_residual_j"]) < 1e-5
