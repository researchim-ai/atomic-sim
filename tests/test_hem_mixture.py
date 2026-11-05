"""Homogeneous friction equations, phase limits, conservation and transactions."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from atomic_model_hem import eos
from atomic_model_hem_closures import darcy_factor
from atomic_model_hem_mixture import workflow
from atomic_model_hem_mixture.cases import baseline_request
from atomic_model_hem_mixture.friction import FLAGS, evaluate
from atomic_model_hem_mixture.model import MixtureHEMNetwork
from atomic_model_workspace.common import digest
from core.status import DomainError
from physics.materials import iapws

SCOPE = {k: True for k in FLAGS}


def drag(state, velocity=0.001, model="mcadams"):
    return evaluate(state, velocity, 0.05, 0.0, viscosity_model=model, scope=SCOPE)


@pytest.mark.parametrize("model", ["mcadams", "cicchitti"])
@pytest.mark.parametrize("quality", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_saturation_viscosity_choices_and_exact_laminar_formula(model, quality):
    state = eos.px(1e6, quality)
    liquid, vapor = iapws.saturation_properties(1e6)
    result = drag(state, model=model)
    expected = (
        1 / ((1 - quality) / liquid.viscosity_pa_s + quality / vapor.viscosity_pa_s)
        if model == "mcadams"
        else (1 - quality) * liquid.viscosity_pa_s + quality * vapor.viscosity_pa_s
    )
    assert result.viscosity_pa_s == pytest.approx(expected, rel=1e-14)
    assert result.pressure_gradient_pa_m == pytest.approx(32 * expected * 0.001 / 0.05**2)
    assert result.entropy_production_w_m3_k >= 0
    assert result.experimentally_validated is False


@pytest.mark.parametrize("velocity", [0.0, 1e-14, -1e-14, 0.001, -0.001, 2.0, -2.0])
def test_signed_drag_rest_and_darcy_convention(velocity):
    state = eos.px(1e6, 0.5)
    r = drag(state, velocity)
    assert r.force_density_n_m3 * velocity <= 0
    assert r.entropy_production_w_m3_k == -r.force_density_n_m3 * velocity / state.temperature_k
    if velocity == 0:
        assert r.darcy_factor is None and r.force_density_n_m3 == 0
    else:
        assert r.darcy_factor == pytest.approx(darcy_factor(r.reynolds))
        assert r.pressure_gradient_pa_m == pytest.approx(
            r.darcy_factor * r.density_kg_m3 * velocity * abs(velocity) / (2 * 0.05)
        )


@pytest.mark.parametrize("phase", ["liquid", "vapor"])
def test_viscosity_is_continuous_from_saturated_to_single_phase(phase):
    sat = eos.px(1e6, 0.0 if phase == "liquid" else 1.0)
    sign = -1 if phase == "liquid" else 1
    single = eos.pt(1e6, sat.temperature_k + sign * 1e-5)
    for model in ["mcadams", "cicchitti"]:
        assert drag(sat, model=model).viscosity_pa_s == pytest.approx(
            drag(single, model=model).viscosity_pa_s, rel=1e-5
        )


@pytest.mark.parametrize(
    "bad",
    [
        "scope",
        "model",
        "density",
        "internal_energy",
        "temperature",
        "transition",
        "bool",
        "roughness",
    ],
)
def test_out_of_domain_inputs_refused(bad):
    s = eos.px(1e6, 0.5)
    kw = dict(viscosity_model="mcadams", scope=deepcopy(SCOPE))
    v = 0.001
    d = 0.05
    rough = 0.0
    if bad == "scope":
        kw["scope"]["no_slip"] = False
    elif bad == "model":
        kw["viscosity_model"] = "unknown"
    elif bad == "density":
        s = replace(s, density_kg_m3=s.density_kg_m3 * 2)
    elif bad == "internal_energy":
        s = replace(s, internal_energy_j_kg=s.internal_energy_j_kg + 10)
    elif bad == "temperature":
        s = eos.pt(1e6, 700.0)
    elif bad == "transition":
        v = 3000 * drag(s).viscosity_pa_s / (s.density_kg_m3 * d)
    elif bad == "bool":
        v = True
    else:
        rough = 0.1
    with pytest.raises((ValueError, DomainError)):
        evaluate(s, v, d, rough, **kw)


def test_alternative_viscosity_is_sensitivity_not_uncertainty():
    s = eos.px(1e6, 0.5)
    assert drag(s, model="cicchitti").pressure_gradient_pa_m > drag(s).pressure_gradient_pa_m


def test_mixture_drag_only_changes_momentum_not_total_energy_rhs():
    from atomic_model_hem_network.model import GraphHEMNetwork

    request = baseline_request()
    m = MixtureHEMNetwork(request)
    plain = GraphHEMNetwork(request["model"])
    a = m.evaluate(m._y, m._controls)[0]
    b = plain.evaluate(plain._y, plain._controls)[0]
    assert a[2 * m.n : 3 * m.n] == pytest.approx(b[2 * m.n : 3 * m.n])
    i = m.edge_indices[0][0]
    force = a[m.n + i] - b[m.n + i]
    assert force < 0
    assert -force * 0.001 > 0
    o = m.advance(0.001)
    assert abs(o["global_ledger"]["energy_residual_j"]) < 1e-8
    assert o["global_ledger"]["drag_entropy_production_j_k"] > 0


def test_checkpoint_replay_online_power_and_pure_residual():
    request = baseline_request()
    m = MixtureHEMNetwork(request)
    controls = m.control()
    controls["walls"]["w0"] = 1000.0
    m.step(controls, 0.0002, expected_revision=0, controller_state={"tick": 1})
    cp = json.loads(json.dumps(m.checkpoint()))
    r = MixtureHEMNetwork(request)
    r.restore(cp)
    assert r.advance(0.0002) == m.advance(0.0002)
    before = m.checkpoint()
    rate = m.evaluate(m._y, m._controls)[0]
    assert np.max(abs(m.residual(m._y, rate))) == 0
    assert m.checkpoint() == before
    bad = deepcopy(before)
    bad["payload"]["coordinates"][0] += 0.01
    bad["sha256"] = digest(bad["payload"])
    with pytest.raises(ValueError):
        m.restore(bad)
    assert m.checkpoint() == before


def test_late_stage_failure_is_atomic(monkeypatch):
    m = MixtureHEMNetwork(baseline_request())
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
    with pytest.raises(ValueError):
        m.step(m.control(), 0.001, expected_revision=0, controller_state={})
    assert m.checkpoint() == before


def test_double_count_unknown_edge_and_false_scope_rejected():
    for bad in ["double", "unknown", "scope"]:
        request = baseline_request()
        if bad == "double":
            request["model"]["edges"][0]["friction"] = "darcy"
        elif bad == "unknown":
            request["mixture_friction"]["missing"] = deepcopy(request["mixture_friction"]["e0"])
        else:
            request["mixture_friction"]["e0"]["scope"]["no_slip"] = False
        with pytest.raises(ValueError):
            MixtureHEMNetwork(request)


def test_artifact_full_replay_tamper_and_no_overwrite(tmp_path):
    request = baseline_request()
    spec = dict(schema_version=1, model=request, program=[dict(duration_s=0.0001, command=None)])
    folder = tmp_path / "run"
    result = workflow.run(spec, folder)
    assert workflow.replay(folder) == result
    with pytest.raises(FileExistsError):
        workflow.run(spec, folder)
    result["observations"][-1]["cells"][0]["mass_kg"] += 0.01
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        workflow.replay(folder)


@pytest.mark.parametrize(
    "density,energy", [(0.0, 1e6), (-1.0, 1e6), (float("nan"), 1e6), (10.0, float("nan"))]
)
def test_nonfinite_or_nonpositive_state_coordinates_rejected(density, energy):
    state = replace(eos.px(1e6, 0.5), density_kg_m3=density, internal_energy_j_kg=energy)
    with pytest.raises(ValueError):
        drag(state)


def test_cli_domain_refusal_has_no_traceback_or_partial_publication(tmp_path, capsys):
    from atomic_model_hem_mixture.__main__ import main

    request = baseline_request()
    s = eos.px(1e6, 0.5)
    velocity = 3000 * drag(s).viscosity_pa_s / (s.density_kg_m3 * 0.05)
    for edge in request["model"]["edges"]:
        for p in edge["initial"]:
            p["velocity_m_s"] = velocity
    spec = dict(schema_version=1, model=request, program=[dict(duration_s=0.0001, command=None)])
    source = tmp_path / "request.json"
    source.write_text(json.dumps(spec))
    folder = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(["run", str(source), "--output", str(folder)])
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
    assert not folder.exists()


def test_restore_rejects_rehashed_history_change():
    m = MixtureHEMNetwork(baseline_request())
    m.advance(0.0002)
    before = m.checkpoint()
    bad = deepcopy(before)
    bad["payload"]["calls"][0]["duration_s"] *= 2
    bad["sha256"] = digest(bad["payload"])
    with pytest.raises(ValueError):
        m.restore(bad)
    assert m.checkpoint() == before


@pytest.mark.parametrize("model", ["mcadams", "cicchitti"])
def test_absolute_reference_drag_and_stage_cache_lifetime(model):
    from atomic_model_hem_mixture.reference import Assembly
    from atomic_model_hem_network.reference import Assembly as BaseReference

    request = baseline_request(model)
    a = Assembly(request)
    b = BaseReference(request["model"])
    y = a.y0.copy()
    rate = a.rhs(0.0, y)
    plain = b.rhs(0.0, b.y0)
    i = a.names.index("e0/0")
    liquid, vapor = iapws.saturation_properties(1e6)
    mu = (
        1 / (0.5 / liquid.viscosity_pa_s + 0.5 / vapor.viscosity_pa_s)
        if model == "mcadams"
        else (liquid.viscosity_pa_s + vapor.viscosity_pa_s) / 2
    )
    assert rate[a.n + i] - plain[a.n + i] == pytest.approx(
        -32 * mu * a.volumes[i] * 0.001 / 0.05**2, rel=1e-9
    )
    assert rate[2 * a.n : 3 * a.n] == pytest.approx(plain[2 * a.n : 3 * a.n])
    assert a._stage_y is None and a._stage_states is None
    y[a.n + i] *= 1.5
    changed = a.rhs(0.0, y)
    assert not np.array_equal(changed, rate)
    assert a._stage_y is None and a._stage_states is None


@pytest.mark.parametrize("re", [1e-100, 1e-12, 100.0, 1000.0, 2300.0, 3000.0, 4000.0, 1e5, 1e8])
def test_churchill_log_evaluation_and_darcy_factor_convention(re):
    from math import log

    from atomic_model_hem_mixture.friction import churchill

    value = churchill(re, 1e-4)
    if re < 1000:
        assert value == pytest.approx(64 / re, rel=1e-12)
    else:
        A = (2.457 * log(1 / ((7 / re) ** 0.9 + 0.27e-4))) ** 16
        B = (37530 / re) ** 16
        expected = 8 * ((8 / re) ** 12 + 1 / (A + B) ** 1.5) ** (1 / 12)
        assert value == pytest.approx(expected, rel=1e-12)


def test_churchill_transition_requires_explicit_law_choice():
    state = eos.px(1e6, 0.5)
    velocity = 3000 * drag(state).viscosity_pa_s / (state.density_kg_m3 * 0.05)
    with pytest.raises(DomainError):
        drag(state, velocity)
    result = evaluate(
        state, velocity, 0.05, 0.0, viscosity_model="mcadams", scope=SCOPE, darcy_law="churchill"
    )
    assert result.darcy_law == "churchill" and result.pressure_gradient_pa_m > 0
    assert result.entropy_production_w_m3_k > 0


def test_strict_inverse_retry_preserves_both_conserved_coordinates():
    from atomic_model_hem_mixture.inverse import conserved

    density = 5.145385986802207
    u = 2582770.658406389
    state = conserved(density, u, 1e6, eos.px(1e6, 1).temperature_k)
    assert state.density_kg_m3 == pytest.approx(density, rel=2e-9)
    assert state.internal_energy_j_kg == pytest.approx(u, abs=0.002)
    with pytest.raises(ValueError):
        conserved(-1, u, 1e6, 453.0)


def test_inverse_failure_mid_call_rolls_back_everything(monkeypatch):
    import atomic_model_hem_mixture.model as module

    m = MixtureHEMNetwork(baseline_request())
    before = m.checkpoint()
    original = module.conserved
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == m.n + 1:
            raise ValueError("All strict EOS seeds failed")
        return original(*args)

    monkeypatch.setattr(module, "conserved", fail)
    with pytest.raises(ValueError):
        m.advance(0.001)
    monkeypatch.setattr(module, "conserved", original)
    assert m.checkpoint() == before
