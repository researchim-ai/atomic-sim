"""Regression of the distinct reduced plant; no plant validation claim."""

import json

import numpy as np
import pytest

from atomic_model_plant_extension import water
from atomic_model_plant_extension.initialization import solve
from atomic_model_plant_extension.pwr import SourceBoundPWR
from atomic_model_plant_extension.session import Session
from atomic_model_workspace.common import digest
from physics.materials import iapws


@pytest.mark.parametrize("pressure", [16e6, 16.25e6, 16.5e6])
@pytest.mark.parametrize("quality", [0.0, 1e-7, 0.2, 0.9999999, 1.0])
def test_upper_chart_inverse_and_derivatives(pressure, quality):
    point = water.px(pressure, quality)
    inv = water.rho_u(point.density_kg_m3, point.internal_energy_j_kg, 15e6, 600.0)
    assert abs(inv.pressure_pa - pressure) < 0.02
    assert abs(inv.internal_energy_j_kg - point.internal_energy_j_kg) < 2e-5
    jac, _ = water.mixture_jacobian(pressure, quality)

    def independent(p):
        f, g = iapws.saturation_properties(p)
        return np.array(
            [
                (1 - quality) * f.v_m3_kg + quality * g.v_m3_kg,
                (1 - quality) * f.u_j_kg + quality * g.u_j_kg,
            ]
        )

    numeric = (independent(pressure + 10.0) - independent(pressure - 10.0)) / 20.0
    np.testing.assert_allclose(jac[:, 0], numeric, rtol=2e-6, atol=1e-12)


def test_initialization_no_parameter_changes_and_liquid_port():
    p = SourceBoundPWR()
    r = solve(p)
    assert r["warmup_steps"] == 0 and not r["parameter_changes"]
    assert r["max_inventory_energy_rate_w"] < 1.0
    v = p.network.volumes["pressurizer"]
    port = p.network._node("pressurizer")
    assert port.density_kg_m3 > v.thermo.rho_kg_m3
    assert port.boron_fraction * 1e6 == pytest.approx(
        v.boron_kg / (v.mass_kg * (1 - v.thermo.quality)) * 1e6
    )
    assert p.absorber_worth == pytest.approx(0.06042)


def test_restart_with_commands_and_json_roundtrip():
    s = Session()
    s.advance(0.2, commands={"scram": True, "throttle": 1.0})
    assert s.history[-1]["effective_throttle"] == 0.0
    saved = json.loads(json.dumps(s.checkpoint()))
    restored = Session.restore(saved)
    s.advance(0.2)
    restored.advance(0.2)
    assert s.checkpoint() == restored.checkpoint()


def test_failed_transaction_restores_commands_diagnostics_ledger(monkeypatch):
    s = Session()
    s.advance(0.1)
    before = s.checkpoint()
    original = s.plant.advance

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected after accepted physical step")

    monkeypatch.setattr(s.plant, "advance", fail)
    with pytest.raises(RuntimeError):
        s.advance(0.2, commands={"scram": True, "motor_off": True})
    assert s.checkpoint() == before
    monkeypatch.setattr(s.plant, "advance", original)
    s.advance(0.2)
    ref = Session.restore(before)
    ref.advance(0.2)
    np.testing.assert_allclose(s.plant.pack(), ref.plant.pack(), rtol=2e-10, atol=1e-4)
    assert Session.restore(s.checkpoint()).checkpoint() == s.checkpoint()


@pytest.mark.parametrize(
    "commands", [{"motor_on": True, "motor_off": True}, {"throttle": float("nan")}, {"other": 1}]
)
def test_invalid_commands_atomic(commands):
    s = Session()
    before = s.checkpoint()
    with pytest.raises(ValueError):
        s.advance(0.1, commands=commands)
    assert before == s.checkpoint()


def test_checkpoint_tampering_rejected():
    s = Session()
    saved = s.checkpoint()
    saved["payload"]["source"]["foundation_sha256"] = "0" * 64
    saved["sha256"] = digest(saved["payload"])
    with pytest.raises(ValueError, match="source"):
        Session.restore(saved)


def test_gravity_continuity_at_surge_reversal():
    p = SourceBoundPWR()
    solve(p)
    state = p.pack()
    i = 13 + 3 * len(p.network.volumes) + p.network.junction_names().index("surge")
    state[i] = -1e-8
    left = p.derivatives(0.0, state)[i]
    state[i] = 1e-8
    right = p.derivatives(0.0, state)[i]
    assert abs(left - right) < 1e-8


def test_instantaneous_complete_energy_and_material_balances():
    p = SourceBoundPWR()
    solve(p)
    dot = p.derivatives(0.0, p.pack())
    assert abs(sum(p.account_rates_w(dot).values()) - sum(p.boundary_terms_w().values())) < 0.01
    names = p.network.volume_names()
    assert abs(sum(dot[13 + 3 * i] for i in range(len(names))) + dot[-3]) < 1e-8
    assert abs(sum(dot[15 + 3 * i] for i in range(len(names)))) < 1e-10


@pytest.mark.parametrize("pressure", [16e6, 16.25e6, 16.5e6])
@pytest.mark.parametrize("temperature", [550.0, 650.0, 800.0])
def test_upper_chart_single_phase(pressure, temperature):
    point = water.pt(pressure, temperature)
    recovered = water.rho_u(point.density_kg_m3, point.internal_energy_j_kg, 15e6, 600.0)
    assert abs(recovered.pressure_pa - pressure) < 0.1
    assert abs(recovered.temperature_k - temperature) < 1e-5


def test_campaign_publishes_native_json_boolean_types(tmp_path, monkeypatch):
    from atomic_model_plant_extension.qualification import CASES, run_case

    monkeypatch.setitem(CASES, "nominal", dict(duration_s=0.2, changes=[]))
    result = run_case(
        "nominal", output=tmp_path / "unit_campaign", max_step_s=0.1, output_step_s=0.1
    )
    assert result["full_horizon_accepted"] is True
    assert all(type(v) is bool for v in result["checks"].values())
    json.dumps(result, allow_nan=False)


def test_rhs_is_independent_of_rejected_property_trials():
    p = SourceBoundPWR()
    solve(p)
    y = p.pack()
    before = p.derivatives(0.0, y).copy()
    changed = y.copy()
    changed[-2] *= 1.00001
    p.derivatives(0.0, changed)
    np.testing.assert_array_equal(p.derivatives(0.0, y), before)


def test_directional_guard_roots_simultaneous_and_at_output_boundary():
    from atomic_model_plant_extension.continuation import HybridContinuation

    def event(t, y):
        return t - 0.5

    event.direction = 1
    context = HybridContinuation(
        lambda t, y: np.ones(1),
        np.zeros(1),
        0.0,
        0.5,
        rtol=1e-9,
        atol=1e-11,
        first=0.1,
        max_step=0.2,
        jacstep=1e-6,
        method="Radau",
    )
    result, hit = context.advance_until(0.5, [event, event])
    assert result.t[-1] == pytest.approx(0.5, abs=1e-12)
    assert hit == [0, 1]
    assert result.final[0] == pytest.approx(0.5, abs=1e-12)
    with pytest.raises(Exception, match="Invalid hybrid"):
        context.advance_until(1.0, [event])


def test_open_valve_implicit_trial_extension_has_no_enthalpy_kink():
    p = SourceBoundPWR()
    solve(p)
    y = p.pack()
    index = p.feed_index
    y[index] = 0.0
    p._feed_open = True
    center = p.derivatives(0.0, y).copy()
    left = y.copy()
    right = y.copy()
    left[index] = -1e-5
    right[index] = 1e-5
    dl = (center - p.derivatives(0.0, left)) / 1e-5
    dr = (p.derivatives(0.0, right) - center) / 1e-5
    i = 14 + 3 * p.network.volume_names().index("sg_secondary")
    assert dl[i] == pytest.approx(dr[i], rel=1e-7, abs=0.1)
    p.refresh_algebraic(left)
    with pytest.raises(Exception, match="nonnegative"):
        p._reject_if_out_of_domain()


def test_automatic_power_trip_and_interlock_priority_inside_step():
    p = SourceBoundPWR()
    solve(p)
    p.power_protection.setpoint = 0.99 * p.kinetics.power_w(p.kinetics_state[0])
    p.power_protection.delay_s = 0.35
    initial = p.rod
    p.advance(
        1.0, method="Radau", rtol=1e-7, atol=1e-6, first_step_s=0.1, jacobian_relative_step=2e-6
    )
    assert p.last_protection_event_s == pytest.approx(0.35, abs=1e-9)
    assert p.scram_latched and p.power_protection.latched
    assert p.throttle == 1.0 and p.effective_steam_throttle == 0.0
    assert p.rod > initial


def test_inactive_source_opens_when_initial_guard_is_exactly_zero():
    from types import SimpleNamespace

    from atomic_model_plant_extension.one_way import OneWayFeedPlant

    class ActivationProbe(OneWayFeedPlant):
        # Independent scalar system: p=t, dq_dump/dt=t-q_dump, q_dump(0)=0.
        # It exercises the hybrid active-set contract without water properties.
        def __init__(self):
            self.network = SimpleNamespace(
                volumes={"sg_secondary": SimpleNamespace(pressure_pa=0.0)},
                junctions={
                    "feed": SimpleNamespace(
                        flow_law="momentum", flow_target_kg_s=1.0, flow_time_constant_s=1.0
                    ),
                    "dump": SimpleNamespace(
                        flow_law="target", flow_target_kg_s=0.0, flow_time_constant_s=1.0
                    ),
                },
                junction_names=lambda: ["feed", "dump"],
                volume_names=lambda: ["sg_secondary"],
            )
            self.throttle = 1.0
            self.motor_on = True
            self.scram_latched = False
            self.rod_command = 0.0
            self._rod_rate_override = None
            self.feed_bias_kg_s = 0.0
            self.t_sink = 300.0
            self.maximum_internal_step_s = 0.2
            self.p_dump_open_pa = 0.0
            self.feed_density_ref = 1.0
            self.feed_inertance = 1.0
            self._hybrid = None
            self._hybrid_signature = None
            self._capture = []

        def refresh_algebraic(self, state):
            self.network.volumes["sg_secondary"].pressure_pa = state[0]
            self.network.junctions["feed"].mass_flow_kg_s = state[16]
            self.network.junctions["dump"].flow_target_kg_s = max(0.0, state[0])

        def feed_terms(self):
            return 1.0, 0.0, 0.0

        def derivatives(self, t, state):
            self.refresh_algebraic(state)
            out = np.zeros_like(state)
            out[0] = 1.0
            out[17] = max(0.0, state[0]) - state[17]
            return out

    probe = ActivationProbe()
    state = np.zeros(18)
    state[16] = 1.0
    result = probe._solve_interval(state, 0.0, 1.0, 1e-9, 1e-11, 0.1, 1e-6, "Radau")
    assert result.final[17] == pytest.approx(np.exp(-1.0), rel=1e-8)
    assert any(e["kind"] == "dump_source_opened" and e["t"] == 0.0 for e in probe._capture)
