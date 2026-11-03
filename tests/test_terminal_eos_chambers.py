"""Finite inventory, wall reaction, conservation and independent references."""

import json
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.inertial_graph import InertialGraphError, digest
from scripts.terminal_chamber_references import IndependentTerminal, cases, request
from scripts.terminal_eos_chambers import TerminalEOSChambers


@pytest.mark.parametrize("name,spec", list(cases()))
def test_independent_absolute_energy(name, spec):
    e = TerminalEOSChambers(spec)
    e.advance(0.02)
    s = e.snapshot()
    for method in ["DOP853", "Radau"]:
        ref = IndependentTerminal(spec).integrate([0.02], method)[0]
        for field, tol in [
            ("masses_kg", 1e-7),
            ("velocities_m_s", 1e-5),
            ("temperatures_k", 1e-5),
            ("pressures_pa", 0.02),
        ]:
            np.testing.assert_allclose(
                list(s["channel"][field].values()), ref["channel"][field], rtol=0, atol=tol
            )
        for field, tol in [("m", 1e-7), ("u", 1e-5), ("T", 1e-5), ("p", 0.02)]:
            np.testing.assert_allclose(
                [s["chambers"][side][field] for side in ["left", "right"]],
                ref["chambers"][field],
                rtol=0,
                atol=tol,
            )


@pytest.mark.parametrize("height,shift", [(0.0, 1e5), (10.0, 1e5), (10.0, 1e8)])
def test_rest_and_hydrostatic_equilibrium(height, shift):
    spec = request(height=height, shift=shift)
    e = TerminalEOSChambers(spec)
    before = e.snapshot()
    e.advance(0.1)
    s = e.snapshot()
    assert max(map(abs, s["channel"]["velocities_m_s"].values())) < 1e-9
    for side in ["left", "right"]:
        assert s["chambers"][side]["p"] == pytest.approx(before["chambers"][side]["p"], abs=1e-7)
        assert abs(s["chambers"][side]["u"]) < 1e-9


def test_uniform_heating_analytic_all_inventories():
    spec = request()
    base = TerminalEOSChambers(spec)
    rate = 0.5
    s = base.channel.system
    spec["channel"]["heat"]["power_w"] = (s.m0 * s.eos.cv * rate).tolist()
    for i, c in enumerate(spec["chambers"]):
        c["heat_power_w"] = base.m0[i] * s.eos.cv * rate
    e = TerminalEOSChambers(spec)
    state = e.advance(0.1)
    np.testing.assert_allclose(
        list(state["channel"]["temperatures_k"].values()), 300.05, rtol=0, atol=1e-10
    )
    for side in ["left", "right"]:
        assert state["chambers"][side]["T"] == pytest.approx(300.05, abs=1e-10)
    assert abs(state["energy_residual_j"]) < 1e-7


def test_finite_inventory_changes_and_pairwise_mass_transfer():
    e = TerminalEOSChambers(request(bias=1000))
    initial = e.snapshot()
    s = e.advance(0.1)
    assert s["chambers"]["left"]["p"] < initial["chambers"]["left"]["p"]
    assert s["chambers"]["right"]["p"] > initial["chambers"]["right"]["p"]
    for i, side in enumerate(["left", "right"]):
        assert s["chambers"][side]["m"] - e.m0[i] == pytest.approx(
            -s["channel"]["boundary_transport"][side]["boundary_mass_kg"], abs=1e-12
        )
    assert abs(s["mass_residual_kg"]) < 1e-10
    assert abs(s["energy_residual_j"]) < 1e-7
    assert s["generated_entropy_j_k"] > 0


def test_endwall_reaction_is_not_missing_momentum():
    e = TerminalEOSChambers(request(velocity=2))
    s = e.advance(0.1)
    assert abs(s["chamber_ledgers"]["endwall_impulse_kg_m_s"]) > 1e-6
    assert abs(s["momentum_residual_kg_m_s"]) < 1e-8


def test_orientation_symmetry():
    states = []
    for bias in [1000, -1000]:
        e = TerminalEOSChambers(request(bias=bias))
        states.append(e.advance(0.05))
    for side, opposite in [("left", "right"), ("right", "left")]:
        for field in ["m", "T", "p", "u"]:
            assert states[0]["chambers"][side][field] == pytest.approx(
                (-1 if field == "u" else 1) * states[1]["chambers"][opposite][field], abs=1e-8
            )


def test_json_restart_residual_and_dynamic_views_are_pure():
    spec = request(bias=1000, heat=500)
    e = TerminalEOSChambers(spec)
    initial_baths = deepcopy(e.channel.system.baths)
    e.advance(0.02)
    cp = json.loads(json.dumps(e.checkpoint()))
    y = np.array(cp["payload"]["coordinates"])
    rate = e.evaluate(y)[0]
    np.testing.assert_array_equal(e.residual(y, rate), 0)
    assert e.channel.system.baths == initial_baths
    assert e.checkpoint() == cp
    other = TerminalEOSChambers(spec)
    other.restore(cp)
    e.advance(0.03)
    other.advance(0.03)
    assert e.checkpoint() == other.checkpoint()


@pytest.mark.parametrize("defect", ["mass", "energy", "entropy", "wall_impulse"])
def test_broken_conservation_is_atomic(monkeypatch, defect):
    e = TerminalEOSChambers(request(bias=1000))
    before = e.checkpoint()
    original = TerminalEOSChambers.evaluate

    def broken(self, y):
        rate, ratio, point = original(self, y)
        index = {"mass": self.start, "energy": self.start + 4, "entropy": -2, "wall_impulse": -1}[
            defect
        ]
        rate[index] += 100.0
        return rate, ratio, point

    monkeypatch.setattr(TerminalEOSChambers, "evaluate", broken)
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(TerminalEOSChambers, "evaluate", original)
    assert e.checkpoint() == before


@pytest.mark.parametrize(
    "defect", ["coordinate", "snapshot", "clock", "layout", "digest", "identity"]
)
def test_bad_checkpoint_is_atomic(defect):
    e = TerminalEOSChambers(request())
    e.advance(0.01)
    cp = e.checkpoint()
    bad = deepcopy(cp)
    if defect == "coordinate":
        bad["payload"]["coordinates"][e.start] = -1e9
    elif defect == "snapshot":
        bad["payload"]["snapshot"]["chambers"]["left"]["p"] += 1
    elif defect == "clock":
        bad["payload"]["snapshot"]["revision"] = True
    elif defect == "layout":
        bad["payload"]["coordinates"].pop()
    elif defect == "identity":
        bad["payload"]["input_sha256"] = "bad"
    bad["sha256"] = digest(bad["payload"]) if defect != "digest" else "bad"
    with pytest.raises((NetworkError, InertialGraphError)):
        e.restore(bad)
    assert e.checkpoint() == cp


@pytest.mark.parametrize(
    "defect",
    ["volume", "ua", "temperature", "velocity", "unknown", "chamber_count", "channel_format"],
)
def test_invalid_input_rejected(defect):
    spec = request()
    c = spec["chambers"][0]
    if defect == "volume":
        c["volume_m3"] = 0
    elif defect == "ua":
        c["ambient_ua_w_k"] = -1
    elif defect == "temperature":
        c["initial_temperature_k"] = 1
    elif defect == "velocity":
        c["initial_velocity_m_s"] = 10000
    elif defect == "unknown":
        c["unknown"] = 1
    elif defect == "chamber_count":
        spec["chambers"].pop()
    elif defect == "channel_format":
        spec["channel"]["format"] = "wrong"
    with pytest.raises((NetworkError, InertialGraphError)):
        TerminalEOSChambers(spec)


def test_source_drift_rejected_without_mutation(monkeypatch):
    import scripts.terminal_eos_chambers as module

    e = TerminalEOSChambers(request())
    before = e.checkpoint()
    original = module.signature
    monkeypatch.setattr(module, "signature", lambda: {"drift": True})
    with pytest.raises(InertialGraphError):
        e.advance(0.01)
    monkeypatch.setattr(module, "signature", original)
    assert e.checkpoint() == before


def test_large_finite_chambers_approach_prescribed_exterior_states():
    from scripts.open_channel_references import request as bath_request
    from scripts.open_eos_channel import OpenEOSChannel

    bath = OpenEOSChannel(bath_request(bias=1000))
    bath.advance(0.02)
    target = np.array(list(bath.snapshot()["state"]["velocities_m_s"].values()))
    errors = []
    for volume in [0.5, 5.0, 50.0]:
        spec = request(bias=1000)
        for c in spec["chambers"]:
            c["volume_m3"] = volume
        e = TerminalEOSChambers(spec)
        s = e.advance(0.02)
        errors.append(np.max(abs(np.array(list(s["channel"]["velocities_m_s"].values())) - target)))
    assert errors[0] / errors[1] > 5
    assert errors[1] / errors[2] > 5


def test_temperature_contact_changes_finite_inventory_without_energy_leak():
    spec = request()
    spec["chambers"][0]["initial_temperature_k"] = 330.0
    spec["chambers"][1]["initial_temperature_k"] = 270.0
    e = TerminalEOSChambers(spec)
    s = e.advance(0.05)
    assert s["chambers"]["left"]["T"] < 330.0
    assert s["chambers"]["right"]["T"] > 270.0
    assert abs(s["energy_residual_j"]) < 1e-7
    assert abs(s["mass_residual_kg"]) < 1e-10
    assert s["generated_entropy_j_k"] > 0


@pytest.mark.parametrize("duration", [0, -1, 2, True, float("nan")])
def test_invalid_duration_is_atomic(duration):
    e = TerminalEOSChambers(request())
    before = e.checkpoint()
    with pytest.raises(NetworkError):
        e.advance(duration)
    assert e.checkpoint() == before
