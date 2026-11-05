"""Conserved-coordinate, constitutive-domain and transactional regressions."""

from copy import deepcopy

import numpy as np
import pytest

from atomic_model_hem_qualified import closures as c
from atomic_model_hem_qualified import eos
from atomic_model_hem_qualified.cases import baseline, boiling_case, matrix, program
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_qualified.reference import Assembly


@pytest.mark.parametrize("pressure", [1e5, 2e5, 1e6, 8e6, 15.5e6, 16e6])
@pytest.mark.parametrize("quality", [0.0, 1e-10, 0.01, 0.5, 0.99, 1 - 1e-10, 1.0])
def test_inverse_far_seed(pressure, quality):
    s = eos.px(pressure, quality)
    r = eos.rho_u(s.density_kg_m3, s.internal_energy_j_kg)
    assert abs(r.density_kg_m3 / s.density_kg_m3 - 1) <= eos.RHO_RTOL
    assert abs(r.internal_energy_j_kg - s.internal_energy_j_kg) <= eos.U_ATOL
    assert r.pressure_pa == pytest.approx(pressure, rel=2e-10)


@pytest.mark.parametrize(
    "p,t,v,u,s",
    [
        (3e6, 300.0, 0.00100215168, 112324.818, 392.294792),
        (3e6, 500.0, 0.00120241800, 971934.985, 2580.41912),
        (3500.0, 700.0, 92.3015898, 3012628.19, 10174.9996),
    ],
)
def test_published_if97(p, t, v, u, s):
    if p < eos.P_MIN:
        with pytest.raises(ValueError):
            eos.pt(p, t)
        return
    state = eos.pt(p, t)
    assert 1 / state.density_kg_m3 == pytest.approx(v, rel=5e-9)
    assert state.internal_energy_j_kg == pytest.approx(u, rel=5e-9)
    assert state.entropy_j_kg_k == pytest.approx(s, rel=5e-9)


@pytest.mark.parametrize(
    "state",
    [
        eos.pt(3e6, 350),
        eos.pt(15.5e6, 600),
        eos.pt(1e6, 600),
        eos.px(1e6, 0.5),
        eos.px(15.5e6, 0.05),
    ],
)
def test_inverse_derivatives_against_conserved_finite_difference(state):
    d = eos.derivatives(state)
    rho, u = state.density_kg_m3, state.internal_energy_j_kg
    for h, dr, du, pk, tk in [
        (rho * 1e-5, 1, 0, "pressure_wrt_density", "temperature_wrt_density"),
        (1.0, 0, 1, "pressure_wrt_energy", "temperature_wrt_energy"),
    ]:

        def f(sign):
            return eos.rho_u(
                rho + sign * h * dr, u + sign * h * du, state.pressure_pa, state.temperature_k
            )

        a, b = f(1), f(-1)
        assert (a.pressure_pa - b.pressure_pa) / (2 * h) == pytest.approx(d[pk], rel=3e-5, abs=1e-6)
        assert (a.temperature_k - b.temperature_k) / (2 * h) == pytest.approx(
            d[tk], rel=3e-5, abs=1e-8
        )


@pytest.mark.parametrize("x,side", [(0.0, "liquid"), (1.0, "vapor")])
def test_endpoint_derivative_requires_side(x, side):
    state = eos.px(1e6, x)
    with pytest.raises(ValueError):
        eos.derivatives(state)
    a = eos.derivatives(state, "mixture")
    b = eos.derivatives(state, side)
    assert a["equilibrium_sound_speed_squared_m2_s2"] != b["equilibrium_sound_speed_squared_m2_s2"]
    with pytest.raises(ValueError):
        eos.derivatives(state, "vapor" if x == 0 else "liquid")


@pytest.mark.parametrize("p,i", [(15e6, 0), (16e6, 1)])
@pytest.mark.parametrize("G,j", [(1000, 0), (1500, 1), (2000, 2)])
@pytest.mark.parametrize("x,k", [(0.0, 0), (0.05, 1), (0.1, 2)])
def test_chf_corners(p, i, G, j, x, k):
    assert c.table_chf(p, G, x, 0.008) == c.HIGH_CHF[i, j, k] * 1000


@pytest.mark.parametrize(
    "p,G,x,D",
    [
        (1e6, 1500, 0.05, 0.008),
        (15.5e6, 999, 0.05, 0.008),
        (15.5e6, 1500, 0.101, 0.008),
        (15.5e6, 1500, 0.05, 0.009),
    ],
)
def test_chf_rejects_gaps_and_extrapolation(p, G, x, D):
    with pytest.raises(ValueError):
        c.table_chf(p, G, x, D)


@pytest.mark.parametrize("p,G,model", [(1e6, 500, "biasi_inl"), (15.5e6, 1500, "table1995")])
def test_boiling_independent_formula_and_differential(p, G, model):
    spec = boiling_case(p, G, model)
    m = QualifiedHEMNetwork(spec)
    ref = Assembly(spec)
    state = eos.px(p, 0.05)
    kw = dict(scope={k: True for k in c.HEAT_FLAGS}, chf_model=model)
    tw = state.temperature_k + 2
    result = c.boiling(state, G, 0.008, tw, 2, **kw)
    independent = ref.wall_conductance(ref.walls[0], ref.y0, state)
    assert independent == pytest.approx(m._thermal_conductance(m.primitive(m._y))[0], rel=1e-12)
    eps = 1e-3
    fd = (
        c.boiling(state, G, 0.008, tw + eps, 2, **kw).heat_flux_w_m2
        - c.boiling(state, G, 0.008, tw - eps, 2, **kw).heat_flux_w_m2
    ) / (2 * eps)
    assert fd == pytest.approx(result.differential_conductance_w_m2_k, rel=1e-6)
    with pytest.raises(ValueError, match="post-CHF"):
        c.boiling(state, G, 0.008, tw, 2, maximum_chf_fraction=result.chf_fraction * 0.99, **kw)
    assert result.validated is False
    assert m.contract()["wall_laws"][0]["law"] == result.variant


@pytest.mark.parametrize("name,spec", matrix())
def test_dynamic_replay_and_positive_stocks(name, spec):
    m = QualifiedHEMNetwork(spec)
    first = program(name)[0]["duration_s"]
    m.advance(first)
    cp = m.checkpoint()
    restored = QualifiedHEMNetwork(spec)
    restored.restore(cp)
    assert restored.advance(first) == m.advance(first)
    assert min(cell["mass_kg"] for cell in m.snapshot()["cells"]) > 0
    bad = deepcopy(cp)
    bad["sha256"] = "0" * 64
    before = m.checkpoint()
    with pytest.raises(ValueError):
        m.restore(bad)
    assert m.checkpoint() == before


def test_refined_mesh_and_high_pressure():
    spec = baseline(quality=0.5, cells=32, pressure=15.5e6)
    m = QualifiedHEMNetwork(spec)
    assert m.n == 66
    assert all(s.pressure_pa > 15e6 for s in m.primitive(m._y)[4])


def test_initial_rho_u_chart_and_no_input_mutation():
    spec = baseline()
    state = eos.px(1e6, 0.5)
    spec["model"]["nodes"][0]["initial"] = dict(
        kind="rho_u",
        density_kg_m3=state.density_kg_m3,
        internal_energy_j_kg=state.internal_energy_j_kg,
        pressure_seed_pa=1e6,
        temperature_seed_k=state.temperature_k,
        velocity_m_s=0.0,
    )
    original = deepcopy(spec)
    m = QualifiedHEMNetwork(spec)
    assert spec == original
    assert np.isfinite(m._y).all()


def test_late_constitutive_failure_rolls_back(monkeypatch):
    spec = boiling_case(1e6, 500.0, "biasi_inl")
    m = QualifiedHEMNetwork(spec)
    before = m.checkpoint()
    original = c.boiling
    counter = [0]

    def fail_after_stages(*args, **kwargs):
        counter[0] += 1
        if counter[0] > 7:
            raise ValueError("post-CHF test rejection")
        return original(*args, **kwargs)

    monkeypatch.setattr(c, "boiling", fail_after_stages)
    with pytest.raises(ValueError, match="post-CHF"):
        m.advance(2e-5)
    assert counter[0] > 7
    assert m.checkpoint() == before


def test_rehashed_checkpoint_cannot_change_stocks():
    from atomic_model_workspace.common import digest

    spec = baseline()
    m = QualifiedHEMNetwork(spec)
    m.advance(1e-4)
    before = m.checkpoint()
    bad = deepcopy(before)
    bad["payload"]["coordinates"][0] += 1e-6
    bad["sha256"] = digest(bad["payload"])
    with pytest.raises(ValueError):
        m.restore(bad)
    assert m.checkpoint() == before


def test_closure_sources_and_unknown_uncertainty_explicit():
    contract = QualifiedHEMNetwork(baseline()).contract()
    assert contract["closure_version"] == c.VERSION
    assert contract["closure_sources"]["chf"] == c.CHF_SOURCE
    assert "Not calibrated" in contract["closure_uncertainty"]


def test_cli_roundtrip_and_source_tamper(tmp_path):
    import json

    from atomic_model_hem_qualified.__main__ import main
    from atomic_model_workspace.common import digest

    path = tmp_path / "demo.json"
    assert main(["demo", "--output", str(path)]) == 0
    assert main(["replay", str(path)]) == 0
    with pytest.raises(SystemExit):
        main(["demo", "--output", str(path)])
    saved = json.loads(path.read_text())
    saved["payload"]["source"]["qualified"]["eos.py"] = "0" * 64
    saved["sha256"] = digest(saved["payload"])
    path.write_text(json.dumps(saved))
    with pytest.raises(SystemExit):
        main(["replay", str(path)])


@pytest.mark.parametrize("p", [2e5, 1e6, 15.5e6])
@pytest.mark.parametrize("x", [0.0, 1.0])
def test_saturation_jacobian_one_sided_quality(p, x):
    jac, _ = eos.mixture_jacobian(p, x)
    dp = p * 1e-5
    a, b = eos.px(p + dp, x), eos.px(p - dp, x)
    tangent = np.array(
        [
            (1 / a.density_kg_m3 - 1 / b.density_kg_m3) / (2 * dp),
            (a.internal_energy_j_kg - b.internal_energy_j_kg) / (2 * dp),
        ]
    )
    assert tangent == pytest.approx(jac[:, 0], rel=3e-6, abs=1e-12)
    dx = 1e-5 if x == 0 else -1e-5
    a, b = eos.px(p, x + dx), eos.px(p, x)
    one_sided = np.array(
        [
            (1 / a.density_kg_m3 - 1 / b.density_kg_m3) / dx,
            (a.internal_energy_j_kg - b.internal_energy_j_kg) / dx,
        ]
    )
    assert one_sided == pytest.approx(jac[:, 1], rel=2e-9)


def test_primary_chf_transcription_regression_and_interpolation():
    assert c.table_chf(15e6, 1500.0, 0.1, 0.008) == 1649000.0
    # Four printed source corners, x=.1, G=1500/2000, p=15/16MPa.
    assert c.table_chf(15.5e6, 1750.0, 0.1, 0.008) == (1649 + 1843 + 1419 + 1688) * 250.0


def test_inverse_never_projects_domain_face_mismatch():
    state = eos.px(eos.P_MAX, 0.99)
    # An energy shift cannot be ignored by simply snapping the pressure/quality.
    rho, u = state.density_kg_m3, state.internal_energy_j_kg + 1.0
    try:
        result = eos.rho_u(rho, u)
    except ValueError:
        return
    assert abs(result.internal_energy_j_kg - u) <= eos.U_ATOL
    assert abs(result.density_kg_m3 / rho - 1) <= eos.RHO_RTOL


def test_spatial_evidence_has_stable_json_identity():
    import json

    from atomic_model_hem_qualified.qualification import spatial
    from atomic_model_workspace.common import digest

    report = spatial(True)
    assert digest(report) == digest(json.loads(json.dumps(report)))


@pytest.mark.parametrize("p", [1e5, 2e5, 1e6, 8e6, 15.5e6, 16e6])
@pytest.mark.parametrize("t", [300.0, 450.0, 600.0, 900.0])
def test_single_phase_domain_faces_roundtrip(p, t):
    state = eos.pt(p, t)
    result = eos.rho_u(state.density_kg_m3, state.internal_energy_j_kg)
    assert abs(result.density_kg_m3 / state.density_kg_m3 - 1) <= eos.RHO_RTOL
    assert abs(result.internal_energy_j_kg - state.internal_energy_j_kg) <= eos.U_ATOL
