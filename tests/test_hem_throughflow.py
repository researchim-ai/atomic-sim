"""Independent thermodynamics, source-curated corrections and open-channel invariants."""

from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem_qualified import closures as prior
from atomic_model_hem_qualified import eos
from atomic_model_hem_throughflow import closures, thermo
from atomic_model_hem_throughflow.benchmark import PROTOCOL, evaluate
from atomic_model_hem_throughflow.cases import baseline, matrix
from atomic_model_hem_throughflow.model import SteadyChannel
from atomic_model_workspace.common import digest


@pytest.mark.parametrize("p", [2e5, 1e6, 8e6, 15.5e6])
@pytest.mark.parametrize("x", [0.0, 1e-7, 0.05, 0.5, 0.999, 1.0])
def test_ph_inverse_all_phase_charts(p, x):
    s = eos.px(p, x)
    r = thermo.ph(p, s.enthalpy_j_kg)
    assert abs(r.enthalpy_j_kg - s.enthalpy_j_kg) <= 2e-6
    assert r.density_kg_m3 == pytest.approx(s.density_kg_m3, rel=2e-10)
    assert r.quality == pytest.approx(x, abs=2e-12)


@pytest.mark.parametrize("p,t", [(2e5, 310.0), (1e6, 400.0), (8e6, 600.0), (15.5e6, 890.0)])
def test_ph_single_phase_and_derivatives(p, t):
    s = eos.pt(p, t)
    r = thermo.ph(p, s.enthalpy_j_kg)
    assert r.temperature_k == pytest.approx(t, abs=1e-8)
    vp, vh = thermo.volume_derivatives(s)
    dp = p * 1e-6
    dh = 0.1
    fd_p = (
        1 / thermo.ph(p + dp, s.enthalpy_j_kg).density_kg_m3
        - 1 / thermo.ph(p - dp, s.enthalpy_j_kg).density_kg_m3
    ) / (2 * dp)
    fd_h = (
        1 / thermo.ph(p, s.enthalpy_j_kg + dh).density_kg_m3
        - 1 / thermo.ph(p, s.enthalpy_j_kg - dh).density_kg_m3
    ) / (2 * dh)
    assert vp == pytest.approx(fd_p, rel=5e-5)
    assert vh == pytest.approx(fd_h, rel=5e-5)


@pytest.mark.parametrize("x", [0.01, 0.05, 0.5, 0.9])
def test_actual_mixture_entropy_derivatives_not_assumed_exact_gibbs(x):
    s = eos.px(15.5e6, x)
    p = s.pressure_pa
    h = s.enthalpy_j_kg
    sp, sh = thermo.entropy_derivatives(s)
    dp = 15.0
    dh = 0.1
    fp = (thermo.ph(p + dp, h).entropy_j_kg_k - thermo.ph(p - dp, h).entropy_j_kg_k) / (2 * dp)
    fh = (thermo.ph(p, h + dh).entropy_j_kg_k - thermo.ph(p, h - dh).entropy_j_kg_k) / (2 * dh)
    assert sp == pytest.approx(fp, rel=2e-5)
    assert sh == pytest.approx(fh, rel=2e-5)
    assert abs(sh - 1 / s.temperature_k) > 1e-12


def scope():
    return {k: True for k in closures.FLAGS}


@pytest.mark.parametrize(
    "p,G,x,q",
    [
        (15e6, 300.0, 0.0, 3274000.0),
        (15e6, 300.0, 0.05, 3111000.0),
        (15e6, 500.0, 0.1, 2739000.0),
        (16e6, 300.0, 0.1, 2597000.0),
        (16e6, 500.0, 0.05, 2545000.0),
        (15e6, 1500.0, 0.1, 1649000.0),
    ],
)
def test_new_chf_primary_source_corners(p, G, x, q):
    r = closures.critical_heat_flux(p, G, x, 0.008, 2.0, scope=scope())
    assert r["nominal_8mm_w_m2"] == q
    assert r["diameter_factor"] == 1.0
    f, g = prior.transport(p)
    alpha = x * f.rho_kg_m3 / (x * f.rho_kg_m3 + (1 - x) * g.rho_kg_m3)
    assert r["heated_length_factor"] == pytest.approx(
        np.exp(0.008 / 2 * np.exp(2 * alpha)), rel=1e-14
    )


def test_diameter_factor_and_midpoint_independent_printed_values():
    r = closures.critical_heat_flux(15.5e6, 400.0, 0.1, 0.016, 2.0, scope=scope())
    assert r["nominal_8mm_w_m2"] == (2884 + 2739 + 2597 + 2268) * 250.0
    assert r["diameter_factor"] == pytest.approx(2**-0.5)


@pytest.mark.parametrize(
    "args",
    [
        (14.99e6, 500.0, 0.05, 0.016, 2.0),
        (15.5e6, 299.0, 0.05, 0.016, 2.0),
        (15.5e6, 500.0, 0.101, 0.016, 2.0),
        (15.5e6, 500.0, 0.05, 0.026, 2.0),
        (15.5e6, 500.0, 0.05, 0.016, 0.1),
    ],
)
def test_chf_no_extrapolation(args):
    with pytest.raises(ValueError):
        closures.critical_heat_flux(*args, scope=scope())


def test_prior_chen_numerical_parity_at_old_supported_point():
    p = 15.5e6
    x = 0.05
    G = 1500.0
    D = 0.008
    L = 2.0
    s = eos.px(p, x)
    tw = s.temperature_k + 2.0
    old = prior.boiling(
        s, G, D, tw, L, scope={k: True for k in prior.HEAT_FLAGS}, chf_model="table1995"
    )
    assert closures.heat_flux(p, x, G, D, tw, L, scope=scope()) == pytest.approx(
        old.heat_flux_w_m2, rel=2e-13
    )
    prediction = closures.wall_temperature(p, x, G, D, old.heat_flux_w_m2, L, scope=scope())
    assert prediction["wall_temperature_k"] == pytest.approx(tw, abs=1e-8)


def test_wall_inverse_and_post_chf_refusal():
    p = 15.5e6
    x = 0.05
    G = 520.0
    D = 0.01581
    L = 3.0
    r = closures.wall_temperature(p, x, G, D, 2e5, L, scope=scope())
    assert abs(r["heat_flux_residual_w_m2"]) < 0.001
    assert not r["validated"]
    with pytest.raises(ValueError, match="Post-CHF"):
        closures.wall_temperature(p, x, G, D, r["chf"]["heat_flux_w_m2"], L, scope=scope())
    with pytest.raises(ValueError):
        closures.heat_flux(p, x, G, D, eos.px(p, x).temperature_k + 11, L, scope=scope())
    bad = scope()
    bad["vertical_upflow"] = False
    with pytest.raises(ValueError):
        closures.wall_temperature(p, x, G, D, 2e5, L, scope=bad)


@pytest.mark.parametrize("name,spec", matrix())
def test_open_balances_positive_stocks_restart_and_phase_events(name, spec):
    m = SteadyChannel(spec)
    m.advance_to(spec["length_m"] * 0.2)
    cp = m.checkpoint()
    n = SteadyChannel(spec)
    n.restore(cp)
    assert n.advance_to(spec["length_m"]) == m.advance_to(spec["length_m"])
    o = m.snapshot()
    ledger = o["ledgers"]
    assert o["state"]["density_kg_m3"] > 0
    assert abs(ledger["energy_residual_j_kg"]) < 1e-5
    assert abs(ledger["momentum_residual_pa"]) < 0.001
    assert abs(ledger["corrected_numerical_entropy_residual_j_kg_k"]) < 1e-5
    assert ledger["drag_entropy_j_kg_k"] >= 0
    assert all(abs(e["enthalpy_residual_j_kg"]) <= 2e-6 for e in m.events)
    assert o["state"]["phase"] != "numerical_trial"
    if name in (
        "boiling_entry",
        "vapor_exit",
        "condensation_entry",
        "liquid_exit",
        "piecewise_heat",
    ):
        assert m.events


def test_explicit_property_defect_remains_visible():
    m = SteadyChannel(baseline())
    o = m.advance_to(m.L)["ledgers"]
    assert abs(o["entropy_residual_j_kg_k"]) > 0.001
    assert abs(o["property_gibbs_defect_j_kg_k"]) > 0.001
    assert abs(o["corrected_numerical_entropy_residual_j_kg_k"]) < 1e-5


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass_flux_kg_m2_s", True),
        ("gravity_m_s2", float("nan")),
        ("max_step_m", 0.0),
        ("viscosity_model", "unknown"),
        ("heat_flux_w_m2", []),
        ("unknown", 1),
    ],
)
def test_strict_request_domain(field, value):
    r = baseline()
    r[field] = value
    with pytest.raises(ValueError):
        SteadyChannel(r)


def test_friction_requires_explicit_scope():
    r = baseline()
    r["friction_scope"]["no_slip"] = False
    with pytest.raises(ValueError):
        SteadyChannel(r)


def test_rehashed_checkpoint_cannot_forge_phase_history():
    m = SteadyChannel(baseline())
    m.advance_to(2.0)
    before = m.checkpoint()
    bad = deepcopy(before)
    bad["payload"]["phase_events"][0]["position_m"] += 0.01
    bad["sha256"] = digest(bad["payload"])
    with pytest.raises(ValueError):
        m.restore(bad)
    assert m.checkpoint() == before


def test_late_failure_leaves_original_state_unchanged(monkeypatch):
    m = SteadyChannel(baseline())
    before = m.checkpoint()
    rate = m.rate
    count = [0]

    def fail(*args, **kwargs):
        count[0] += 1
        if count[0] > 20:
            raise ValueError("injected later stage failure")
        return rate(*args, **kwargs)

    monkeypatch.setattr(m, "rate", fail)
    with pytest.raises(ValueError):
        m.advance_to(2.0)
    assert count[0] > 20 and m.checkpoint() == before


def test_real_heldout_failure_is_retained_not_fitted_away():
    r = evaluate()
    cohorts = r["cohort_metrics"]
    assert r["assessed_rows"] == 12320
    assert cohorts["development"]["candidate_rows"] == 71
    assert cohorts["holdout"]["candidate_rows"] == 79
    assert cohorts["holdout"]["prediction_failures"] == 0
    assert (
        cohorts["holdout"]["mean_absolute_relative_heat_flux_error"]
        > PROTOCOL["maximum_mean_absolute_relative_heat_flux_error"]
    )
    assert cohorts["holdout"]["unresolved_superheat_rows"] == 79
    assert r["screening_status"] == "failed" and not r["physical_acceptance"]
    dev = {x["run"] for x in r["rows"] if x["cohort"] == "development"}
    test = {x["run"] for x in r["rows"] if x["cohort"] == "holdout"}
    assert dev.isdisjoint(test)


def test_cli_atomic_publication_replay_resume_and_failed_screen(tmp_path):
    import json

    from atomic_model_hem_throughflow.__main__ import main

    demo = tmp_path / "run.json"
    assert main(["demo", "--output", str(demo)]) == 0
    assert main(["replay", str(demo)]) == 0
    before = demo.read_bytes()
    with pytest.raises(SystemExit):
        main(["demo", "--output", str(demo)])
    assert demo.read_bytes() == before
    spec = baseline()
    m = SteadyChannel(spec)
    m.advance_to(2.0)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(dict(request=spec, targets_m=[4.0, 7.1], checkpoint=m.checkpoint()))
    )
    out = tmp_path / "resumed.json"
    assert main(["run", str(request), "--output", str(out)]) == 0
    assert main(["replay", str(out)]) == 0
    screen = tmp_path / "screen.json"
    assert main(["benchmark", "--output", str(screen), "--require-pass"]) == 3
    assert main(["audit-benchmark", str(screen)]) == 0


@pytest.mark.parametrize("p,t", [(2e5, 300.0), (1e6, 300.0), (8e6, 900.0), (15.5e6, 900.0)])
def test_ph_temperature_domain_faces_without_out_of_domain_stencil(p, t):
    state = eos.pt(p, t)
    inverse = thermo.ph(p, state.enthalpy_j_kg)
    assert inverse.temperature_k == pytest.approx(t, abs=1e-9)
    assert abs(inverse.enthalpy_j_kg - state.enthalpy_j_kg) <= 2e-6


@pytest.mark.parametrize("side", ["liquid", "vapor", "unknown"])
def test_wrong_mixture_derivative_side_rejected(side):
    state = eos.px(15.5e6, 0.05)
    with pytest.raises(ValueError):
        thermo.volume_derivatives(state, side)
    with pytest.raises(ValueError):
        thermo.entropy_derivatives(state, side)


def test_replay_rejects_rehashed_unrecognized_status_metadata(tmp_path):
    import json

    from atomic_model_hem_throughflow.__main__ import main

    path = tmp_path / "run.json"
    assert main(["demo", "--output", str(path)]) == 0
    original = json.loads(path.read_text())
    for level in ("payload", "input"):
        bad = deepcopy(original)
        target = bad["payload"] if level == "payload" else bad["payload"]["input"]
        target["validated"] = True
        bad["sha256"] = digest(bad["payload"])
        path.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            main(["replay", str(path)])
