"""Mixed finite-inventory assembly, typed heat ports and atomic controls."""

import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

from core.network import NetworkError
from scripts.controlled_assembly_references import IndependentAssembly, cases, request
from scripts.controlled_eos_assembly import CIRCUITS, ControlledEOSAssembly
from scripts.inertial_graph import InertialGraphError, digest


def analytic(e, t, temperatures, forcing=(0.0, 0.0, 0.0)):
    C = [c.channel.system.m0[0] * c.channel.system.eos.cv for c in e.circuits]
    a, b = e.ua[0]
    cw = e.capacity[0]
    M = np.array(
        [[-a / C[0], 0, a / C[0]], [0, -b / C[1], b / C[1]], [a / cw, b / cw, -(a + b) / cw]]
    )
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = M
    augmented[:3, 3] = np.array(forcing) / [C[0], C[1], cw]
    return (expm(augmented * t) @ np.r_[temperatures, 1.0])[:3]


def check_uniform(s, expected):
    for i, name in enumerate(CIRCUITS):
        np.testing.assert_allclose(
            list(s["circuits"][name]["channel"]["temperatures_k"].values()),
            expected[i],
            rtol=0,
            atol=1e-7,
        )
        for side in ["left", "right"]:
            assert s["circuits"][name]["chambers"][side]["T"] == pytest.approx(
                expected[i], abs=1e-7
            )
    np.testing.assert_allclose(
        list(s["wall_temperatures_k"].values()), expected[2], rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("cap", [0.001, 0.0005, 0.00025])
def test_analytic_uniform_three_inventory_exchange(cap):
    e = ControlledEOSAssembly(request(uniform=True), max_step_s=cap)
    expected = analytic(e, 0.1, [350.0, 280.0, 300.0])
    s = e.advance(0.1)
    check_uniform(s, expected)
    assert abs(s["external_heat_j"]) < 1e-8
    assert abs(s["energy_residual_j"]) < 1e-7
    assert s["generated_entropy_j_k"] > 0


def test_piecewise_analytic_online_heat_and_inventory_continuity():
    e = ControlledEOSAssembly(request(uniform=True))
    e.advance(0.05)
    before = analytic(e, 0.05, [350.0, 280.0, 300.0])
    mask = e.snapshot()["active_heat_power_w"]
    for key in mask:
        mask[key] = (
            200.0 if key.startswith("wall.") else 100.0 if key.startswith("primary.") else 0.0
        )
    result = e.step(mask, 0.05, expected_revision=e.revision, command_id="heat.step")
    check_uniform(result, analytic(e, 0.05, before, (100.0, 0.0, 200.0)))
    transition = result["heater_transition"]
    assert transition["left"]["port_temperatures_k"] == transition["right"]["port_temperatures_k"]
    assert (
        transition["left"]["total_energy_change_j"] == transition["right"]["total_energy_change_j"]
    )
    assert result["external_heat_j"] == pytest.approx(6 * 300 * 0.05, abs=1e-7)
    assert len(result["history"]) == 1


@pytest.mark.parametrize("name,spec,schedule", list(cases()))
def test_independent_absolute_energy_reference(name, spec, schedule):
    e = ControlledEOSAssembly(spec)
    s = e.advance(0.01)
    for method in ["DOP853", "Radau"]:
        ref = IndependentAssembly(spec).integrate([0.01], [], method)[0]
        np.testing.assert_allclose(
            list(s["wall_temperatures_k"].values()), ref["wall_temperatures_k"], rtol=0, atol=1e-5
        )
        for i, circuit in enumerate(CIRCUITS):
            for field, tol in [
                ("masses_kg", 1e-7),
                ("velocities_m_s", 1e-5),
                ("temperatures_k", 1e-5),
                ("pressures_pa", 0.02),
            ]:
                np.testing.assert_allclose(
                    list(s["circuits"][circuit]["channel"][field].values()),
                    ref["circuits"][i]["channel"][field],
                    rtol=0,
                    atol=tol,
                )
            for field, tol in [("m", 1e-7), ("u", 1e-5), ("T", 1e-5), ("p", 0.02)]:
                np.testing.assert_allclose(
                    [s["circuits"][circuit]["chambers"][side][field] for side in ["left", "right"]],
                    ref["circuits"][i]["chambers"][field],
                    rtol=0,
                    atol=tol,
                )


def test_json_restart_and_view_purity_with_command_history():
    spec = request()
    e = ControlledEOSAssembly(spec)
    e.advance(0.01)
    initial = [
        (c.channel._heaters.copy(), c.heaters.copy(), c.channel.system.baths.copy())
        for c in e.circuits
    ]
    mask = e.snapshot()["active_heat_power_w"]
    mask["primary.chamber.left"] = 1000.0
    mask["wall.w0"] = 500.0
    e.step(mask, 0.01, expected_revision=e.revision, command_id="first")
    cp = json.loads(json.dumps(e.checkpoint()))
    other = ControlledEOSAssembly(spec)
    other.restore(cp)
    y = np.array(cp["payload"]["coordinates"])
    np.testing.assert_array_equal(e.residual(y, e.evaluate(y)[0]), 0)
    assert e.checkpoint() == cp
    for c, (powers, heaters, baths) in zip(e.circuits, initial):
        assert c.channel._heaters == powers
        np.testing.assert_array_equal(c.heaters, heaters)
        assert c.channel.system.baths == baths
    e.advance(0.02)
    other.advance(0.02)
    assert e.checkpoint() == other.checkpoint()


@pytest.mark.parametrize(
    "defect",
    [
        "stale",
        "duplicate",
        "missing_mask",
        "extra_mask",
        "nonfinite",
        "id",
        "duration",
        "failed_stage",
    ],
)
def test_failed_control_is_atomic(defect):
    e = ControlledEOSAssembly(request())
    e.step(e._heaters, 0.01, expected_revision=0, command_id="one")
    before = e.checkpoint()
    mask = e.snapshot()["active_heat_power_w"]
    with pytest.raises((NetworkError, InertialGraphError)):
        if defect == "stale":
            e.step(mask, 0.01, expected_revision=0, command_id="two")
        elif defect == "duplicate":
            e.step(mask, 0.01, expected_revision=e.revision, command_id="one")
        elif defect == "missing_mask":
            mask.pop("wall.w0")
            e.step(mask, 0.01, expected_revision=e.revision, command_id="two")
        elif defect == "extra_mask":
            mask["unknown"] = 0.0
            e.step(mask, 0.01, expected_revision=e.revision, command_id="two")
        elif defect == "nonfinite":
            mask["wall.w0"] = float("nan")
            e.step(mask, 0.01, expected_revision=e.revision, command_id="two")
        elif defect == "id":
            e.step(mask, 0.01, expected_revision=e.revision, command_id="bad id")
        elif defect == "duration":
            e.step(mask, 1.0, expected_revision=e.revision, command_id="two")
        else:
            mask["wall.w0"] = 1e12
            e.step(mask, 0.01, expected_revision=e.revision, command_id="two")
    assert e.checkpoint() == before
    # Failure did not consume the next ID or revision.
    e.step(e._heaters, 0.001, expected_revision=e.revision, command_id="two")


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_circuit",
        "unknown_kind",
        "missing_port",
        "bool_cell",
        "bad_cell",
        "bad_side",
        "same_circuit",
        "duplicate_wall",
        "negative_ua",
        "capacity",
        "temperature",
        "unknown_field",
    ],
)
def test_typed_port_validation(defect):
    spec = request()
    w = spec["walls"][0]
    p = w["ports"][0]
    if defect == "unknown_circuit":
        p["circuit"] = "other"
    elif defect == "unknown_kind":
        p["component"] = "pump"
    elif defect == "missing_port":
        w["ports"].pop()
    elif defect == "bool_cell":
        p["cell"] = True
    elif defect == "bad_cell":
        p["cell"] = 99
    elif defect == "bad_side":
        spec["walls"][1]["ports"][0]["side"] = "top"
    elif defect == "same_circuit":
        w["ports"][1]["circuit"] = "primary"
    elif defect == "duplicate_wall":
        spec["walls"][1]["id"] = w["id"]
    elif defect == "negative_ua":
        w["conductance_w_k"][0] = -1
    elif defect == "capacity":
        w["capacity_j_k"] = 0
    elif defect == "temperature":
        w["initial_temperature_k"] = 1
    elif defect == "unknown_field":
        p["unknown"] = 1
    with pytest.raises((NetworkError, InertialGraphError)):
        ControlledEOSAssembly(spec)


@pytest.mark.parametrize(
    "defect",
    ["coordinates", "snapshot", "history", "active", "identity", "layout", "clock", "digest"],
)
def test_checkpoint_corruption_is_atomic(defect):
    e = ControlledEOSAssembly(request())
    e.step(e._heaters, 0.01, expected_revision=0, command_id="one")
    cp = e.checkpoint()
    bad = deepcopy(cp)
    if defect == "coordinates":
        bad["payload"]["coordinates"][e.wall_start] = -1e12
    elif defect == "snapshot":
        bad["payload"]["snapshot"]["wall_temperatures_k"]["w0"] += 1
    elif defect == "history":
        bad["payload"]["snapshot"]["history"][0]["command_id"] = "bad id"
    elif defect == "active":
        bad["payload"]["snapshot"]["active_heat_power_w"]["wall.w0"] += 1
    elif defect == "identity":
        bad["payload"]["input_sha256"] = "bad"
    elif defect == "layout":
        bad["payload"]["coordinates"].pop()
    elif defect == "clock":
        bad["payload"]["snapshot"]["revision"] = True
    bad["sha256"] = digest(bad["payload"]) if defect != "digest" else "bad"
    with pytest.raises((NetworkError, InertialGraphError)):
        e.restore(bad)
    assert e.checkpoint() == cp


@pytest.mark.parametrize("defect", ["energy", "entropy", "transfer"])
def test_broken_joint_ledgers_rejected(monkeypatch, defect):
    e = ControlledEOSAssembly(request())
    before = e.checkpoint()
    original = ControlledEOSAssembly.evaluate

    def broken(self, y):
        rate, ratio = original(self, y)
        index = {"energy": self.wall_start, "entropy": -1, "transfer": -5}[defect]
        rate[index] += 100.0
        return rate, ratio

    monkeypatch.setattr(ControlledEOSAssembly, "evaluate", broken)
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(ControlledEOSAssembly, "evaluate", original)
    assert e.checkpoint() == before


def test_wall_port_order_invariance():
    spec = request()
    swapped = deepcopy(spec)
    for w in swapped["walls"]:
        w["ports"].reverse()
        w["conductance_w_k"].reverse()
    a = ControlledEOSAssembly(spec).advance(0.02)
    b = ControlledEOSAssembly(swapped).advance(0.02)
    np.testing.assert_allclose(
        list(a["port_temperatures_k"].values()),
        list(b["port_temperatures_k"].values()),
        rtol=0,
        atol=1e-10,
    )


def test_source_drift_is_atomic(monkeypatch):
    import scripts.controlled_eos_assembly as module

    e = ControlledEOSAssembly(request())
    before = e.checkpoint()
    original = module.signature
    monkeypatch.setattr(module, "signature", lambda: {"drift": True})
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(module, "signature", original)
    assert e.checkpoint() == before


def test_two_competing_commands_cannot_commit_same_revision():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    e = ControlledEOSAssembly(request())
    mask = e.snapshot()["active_heat_power_w"]
    barrier = Barrier(2)

    def call(cid):
        barrier.wait(timeout=10)
        try:
            e.step(mask, 0.005, expected_revision=0, command_id=cid)
            return "committed"
        except InertialGraphError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(call, ["one", "two"]))
    assert results.count("committed") == 1
    assert e.revision == 1 and len(e.snapshot()["history"]) == 1
    assert e.snapshot()["time_s"] == 0.005


def test_concurrent_advances_are_serialized_without_lost_time():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    e = ControlledEOSAssembly(request())
    barrier = Barrier(2)

    def advance(_):
        barrier.wait(timeout=10)
        return e.advance(0.005)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(advance, range(2)))
    assert e.revision == 2 and e.snapshot()["time_s"] == 0.01


def test_repeated_ports_sum_conductances_in_stability_bound():
    spec = request()
    for circuit, T in zip(spec["circuits"].values(), [300.001, 299.999]):
        circuit["channel"]["initial"]["temperatures_k"] = [T] * 4
        for chamber in circuit["chambers"]:
            chamber["initial_temperature_k"] = T
    first = spec["walls"][0]
    first.update(capacity_j_k=1e6, conductance_w_k=[1e7, 1e7])
    duplicate = deepcopy(first)
    duplicate["id"] = "repeated"
    spec["walls"] = [first, duplicate]
    e = ControlledEOSAssembly(spec)
    s = e.advance(0.00001)
    assert s["work"]["maximum_internal_step_s"] < 0.00001
    assert s["work"]["maximum_stability_fraction"] <= 1
    assert s["work"]["substeps"] > 1


def test_distinct_caloric_fluids_remain_mass_isolated():
    spec = request()
    spec["circuits"]["secondary"]["channel"]["fluid"].update(
        cv_j_kg_k=800.0, gamma=1.2, pressure_shift_pa=500000.0
    )
    e = ControlledEOSAssembly(spec)
    s = e.advance(0.02)
    for circuit in CIRCUITS:
        assert abs(s["circuits"][circuit]["mass_residual_kg"]) < 1e-10
    assert abs(s["energy_residual_j"]) < 1e-7
