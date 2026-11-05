"""Equation/domain verification only; no experimental validation claim."""

from dataclasses import replace
from math import log10, sqrt

import pytest

from atomic_model_hem_closures import (
    CHF_1995_CUBE_SHADED,
    CLOSURE_VERSION,
    FORCED_FLOW_CHF_AVAILABILITY,
    CHF1995CubeScope,
    PoolScope,
    TubeScope,
    chf_1995_cube,
    darcy_factor,
    forced_flow_chf,
    friction_factor,
    friction_pressure_drop_pa,
    heat_transfer,
    single_phase_transfer,
    zuber_pool_chf_screening,
)
from core.status import Applicability, DomainError
from physics.materials.iapws import properties_pt, saturation_properties

SCOPE = TubeScope(True, True, True, True, True)
POOL = PoolScope(True, True, True)


@pytest.fixture
def water():
    return properties_pt(101325, 300)


def test_laminar_darcy_and_poiseuille_pressure(water):
    assert darcy_factor(1000) == pytest.approx(0.064)
    diameter, length, velocity = 0.01, 2, 0.01
    flux = water.rho_kg_m3 * velocity
    expected = 32 * water.viscosity_pa_s * length * velocity / diameter**2
    actual = friction_pressure_drop_pa(water, flux, diameter, length, scope=SCOPE)
    assert actual == pytest.approx(expected, rel=1e-12)
    assert friction_pressure_drop_pa(water, -flux, diameter, length, scope=SCOPE) == -actual
    result = friction_factor(water, flux, diameter, scope=SCOPE)
    assert result.provenance.version == CLOSURE_VERSION
    assert not result.provenance.experimentally_validated
    assert "omit critical" in result.transport_provenance


@pytest.mark.parametrize(
    "re,rr,expected",
    [
        (1e5, 0, 0.01798977308427384),
        (1e5, 0.001, 0.022174535944515076),
        (4000, 0, 0.0399070140556349),
    ],
)
def test_colebrook_values_and_residual(re, rr, expected):
    f = darcy_factor(re, rr)
    assert f == pytest.approx(expected, rel=1e-11)
    x = 1 / sqrt(f)
    assert abs(x + 2 * log10(rr / 3.7 + 2.51 * x / re)) <= 2e-12


@pytest.mark.parametrize(
    "re", [0, -1, 2300, 3000, 3999.9, 1e8 + 1, float("nan"), float("inf"), 5e-324]
)
def test_reynolds_domain(re):
    with pytest.raises(DomainError):
        darcy_factor(re)


@pytest.mark.parametrize("rr", [-1, 0.051, float("nan"), float("inf")])
def test_roughness_domain(rr):
    with pytest.raises(DomainError):
        darcy_factor(1000, rr)


def test_colebrook_envelope_and_roughness_monotonicity():
    for re in (4000, 1e5, 1e8):
        assert darcy_factor(re, 0.05) > darcy_factor(re, 0)


def test_laminar_heat_value_and_sign(water):
    flux = 1000 * water.viscosity_pa_s / 0.01
    result = heat_transfer(
        water,
        flux,
        0.01,
        310,
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
    )
    assert result.nusselt == 3.66
    assert result.coefficient_w_m2_k == pytest.approx(3.66 * water.conductivity_w_m_k / 0.01)
    assert result.heat_flux_w_m2 == pytest.approx(result.coefficient_w_m2_k * 10)
    assert result.prandtl == pytest.approx(
        water.cp_j_kg_k * water.viscosity_pa_s / water.conductivity_w_m_k
    )
    cooled = heat_transfer(
        water,
        -flux,
        0.01,
        290,
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
    )
    assert cooled.heat_flux_w_m2 == pytest.approx(-result.heat_flux_w_m2)


@pytest.mark.parametrize(
    "change",
    [
        {"region": 4},
        {"region": 3},
        {"viscosity_pa_s": float("nan")},
        {"conductivity_w_m_k": 0},
        {"applicability": Applicability.OUT_OF_DOMAIN},
        {"T_k": 650},
        {"p_pa": float("nan")},
    ],
)
def test_state_rejections(water, change):
    with pytest.raises(DomainError):
        friction_factor(replace(water, **change), 10, 0.01, scope=SCOPE)


@pytest.mark.parametrize(
    "change",
    [
        {"circular_straight": False},
        {"fully_developed": False},
        {"constant_properties": False},
        {"forced_convection": False},
        {"negligible_compressibility": False},
        {"heat_transfer_regime": "two_phase"},
        {"heat_transfer_regime": "boiling"},
        {"heat_transfer_regime": "CHF"},
        {"heat_transfer_regime": "postCHF"},
    ],
)
def test_scope_rejections(water, change):
    with pytest.raises(DomainError):
        friction_factor(water, 10, 0.01, scope=replace(SCOPE, **change))


@pytest.mark.parametrize(
    "kw",
    [
        {"constant_wall_temperature": False},
        {"thermally_developed": False},
        {"roughness_m": 1e-6},
        {"wall_temperature_k": 400},
        {"wall_temperature_k": float("nan")},
        {"mass_flux_kg_m2_s": 1e7},
    ],
)
def test_heat_domain(water, kw):
    args = dict(
        state=water,
        mass_flux_kg_m2_s=10,
        diameter_m=0.01,
        wall_temperature_k=310,
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
    )
    args.update(kw)
    with pytest.raises(DomainError):
        heat_transfer(**args)


def test_saturation_and_vapor():
    liquid, vapor = saturation_properties(101325)
    for state in (liquid, vapor):
        with pytest.raises(DomainError):
            friction_factor(state, 1, 0.01, scope=SCOPE)
    steam = properties_pt(101325, 400)
    result = heat_transfer(
        steam, 0.1, 0.01, 410, scope=SCOPE, constant_wall_temperature=True, thermally_developed=True
    )
    assert result.coefficient_w_m2_k > 0
    with pytest.raises(DomainError):
        heat_transfer(
            steam,
            0.1,
            0.01,
            350,
            scope=SCOPE,
            constant_wall_temperature=True,
            thermally_developed=True,
        )


@pytest.mark.parametrize(
    "flux,diameter,length,roughness",
    [
        (0, 0.01, 1, 0),
        (1, 0, 1, 0),
        (1, -0.01, 1, 0),
        (float("nan"), 0.01, 1, 0),
        (1, 0.01, -1, 0),
        (1, 0.01, 0, 0),
        (1, 0.01, 1, -1),
    ],
)
def test_geometry_and_flow_rejections(water, flux, diameter, length, roughness):
    with pytest.raises(DomainError):
        friction_pressure_drop_pa(water, flux, diameter, length, scope=SCOPE, roughness_m=roughness)


def test_zuber_atmospheric_screening():
    # Rounded atmospheric saturation sigma supplied as a test fixture, not fitted evidence.
    result = zuber_pool_chf_screening(
        101325, 0.0589, surface_tension_source="rounded test fixture at 1 atm", scope=POOL
    )
    assert result.heat_flux_w_m2 == pytest.approx(1.108e6, rel=0.003)
    assert not result.qualifies_forced_flow
    assert not result.provenance.experimentally_validated
    assert "screening" in result.provenance.correlation


@pytest.mark.parametrize(
    "kw",
    [
        {"pressure_pa": 22e6},
        {"pressure_pa": float("nan")},
        {"surface_tension_n_m": 0},
        {"surface_tension_n_m": float("inf")},
        {"surface_tension_source": ""},
        {"scope": replace(POOL, application="forced_flow")},
        {"scope": replace(POOL, application="postCHF")},
        {"scope": replace(POOL, saturated_quiescent=False)},
        {"scope": replace(POOL, large_horizontal_upward_wetted=False)},
        {"scope": replace(POOL, pure_water=False)},
    ],
)
def test_zuber_scope_rejections(kw):
    args = dict(
        pressure_pa=101325, surface_tension_n_m=0.0589, surface_tension_source="fixture", scope=POOL
    )
    args.update(kw)
    with pytest.raises(DomainError):
        zuber_pool_chf_screening(**args)


def _heat_at(water, re, *, pr=None, **kwargs):
    # Synthetic cp only for equation/domain boundary tests, not an EOS validation.
    if pr is not None:
        water = replace(water, cp_j_kg_k=pr * water.conductivity_w_m_k / water.viscosity_pa_s)
    args = dict(scope=SCOPE, constant_wall_temperature=True, thermally_developed=True)
    args.update(kwargs)
    return heat_transfer(water, re * water.viscosity_pa_s / 0.01, 0.01, 310, **args)


@pytest.mark.parametrize(
    "re,pr,expected",
    [
        (1e5, 7, 598.5338811602347),
        (4000, 0.5, 11.72098307034553),
        (5e6, 2000, 164763.03678619582),
        (1e4, 1, 35.36668175645087),
    ],
)
def test_gnielinski_reference_and_boundaries(water, re, pr, expected):
    result = _heat_at(water, re, pr=pr, correlation="gnielinski")
    assert result.nusselt == pytest.approx(expected, rel=2e-12)
    assert result.provenance.correlation == "gnielinski-smooth-inl-1.82"
    assert result.provenance.source.startswith("https://mooseframework.inl.gov/")
    assert not result.provenance.experimentally_validated
    assert result.heat_flux_w_m2 == pytest.approx(result.coefficient_w_m2_k * 10)


@pytest.mark.parametrize(
    "re,pr",
    [
        (2300, 7),
        (3000, 7),
        (3999.99, 7),
        (5e6 + 1, 7),
        (1e5, 0.49999),
        (1e5, 2000.01),
    ],
)
def test_gnielinski_domain(water, re, pr):
    with pytest.raises(DomainError):
        _heat_at(water, re, pr=pr, correlation="gnielinski")


@pytest.mark.parametrize(
    "re,correlation", [(1000, "gnielinski"), (10000, "laminar"), (1000, "typo")]
)
def test_explicit_correlation_selection(water, re, correlation):
    with pytest.raises(DomainError):
        _heat_at(water, re, correlation=correlation)


def test_auto_selects_both_branches(water):
    assert _heat_at(water, 1000).nusselt == 3.66
    result = _heat_at(water, 10000)
    assert result.nusselt == _heat_at(water, 10000, correlation="gnielinski").nusselt
    assert result.provenance != _heat_at(water, 1000).provenance


@pytest.mark.parametrize("region", [1, 2])
@pytest.mark.parametrize("offset", [-1e-6, 0, 1e-6])
def test_turbulent_wall_saturation_guard(region, offset):
    tsat = saturation_properties(101325)[0].T_k
    state = properties_pt(101325, 300 if region == 1 else 400)
    flux = 1e5 * state.viscosity_pa_s / 0.01
    args = dict(
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
        correlation="gnielinski",
    )
    safe = offset < 0 if region == 1 else offset > 0
    if safe:
        assert heat_transfer(state, flux, 0.01, tsat + offset, **args).coefficient_w_m2_k > 0
    else:
        with pytest.raises(DomainError, match="saturation"):
            heat_transfer(state, flux, 0.01, tsat + offset, **args)


def test_network_adapter_stage_updates_and_flow_reversal(water):
    args = dict(
        diameter_m=0.01,
        length_m=2,
        roughness_m=0,
        wall_temperature_k=310,
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
    )
    velocity = 1e5 * water.viscosity_pa_s / (0.01 * water.rho_kg_m3)
    initial = single_phase_transfer(water, velocity, **args)
    reverse = single_phase_transfer(water, -velocity, **args)
    assert reverse == initial
    slower = single_phase_transfer(water, velocity / 2, **args)
    assert slower.coefficient_w_m2_k < initial.coefficient_w_m2_k
    with pytest.raises(DomainError, match="Transition"):
        single_phase_transfer(water, velocity * 0.03, **args)
    with pytest.raises(DomainError, match="saturation"):
        single_phase_transfer(water, velocity, **{**args, "wall_temperature_k": 400})
    mixed = replace(water, region=4)
    with pytest.raises(DomainError):
        single_phase_transfer(mixed, velocity, **args)


@pytest.mark.parametrize(
    "change",
    [
        {"length_m": 0},
        {"length_m": float("nan")},
        {"diameter_m": 0},
        {"roughness_m": 1e-6},
        {"velocity_m_s": 0},
        {"velocity_m_s": float("inf")},
    ],
)
def test_adapter_geometry_and_flow(water, change):
    args = dict(
        velocity_m_s=1,
        diameter_m=0.01,
        length_m=2,
        roughness_m=0,
        wall_temperature_k=310,
        scope=SCOPE,
        constant_wall_temperature=True,
        thermally_developed=True,
    )
    args.update(change)
    with pytest.raises(DomainError):
        single_phase_transfer(water, **args)


def test_forced_flow_chf_is_explicitly_blocked():
    assert FORCED_FLOW_CHF_AVAILABILITY.status == "partial_1995_tiny_cube_only"
    assert FORCED_FLOW_CHF_AVAILABILITY.sources
    with pytest.raises(DomainError, match="Primary IAEA 1995 LUT PDF downloaded") as exc:
        forced_flow_chf(1e6, 1000, 0.1, diameter_m=0.008, length_m=1)
    assert exc.value.applicability == Applicability.UNSUPPORTED


def test_primary_iaea_audit_identity():
    audit = FORCED_FLOW_CHF_AVAILABILITY
    assert audit.primary_pdf_size_bytes == 13609780
    assert (
        audit.primary_pdf_sha256
        == "8d039d4f8a950a1c013ddae312aee744ae5eb0e3efbaa337ce2a7fea96021c1c"
    )
    assert audit.primary_table_pdf_pages == tuple(range(184, 191))
    assert "shading masks" in audit.reason


CUBE_SCOPE = CHF1995CubeScope(True, True, True, True, True)


@pytest.mark.parametrize(
    "p,g,x,q_kw",
    [
        (300000, 1000, 0, 4189),
        (300000, 1000, 0.05, 3285),
        (300000, 1500, 0, 4209),
        (300000, 1500, 0.05, 3339),
        (500000, 1000, 0, 4873),
        (500000, 1000, 0.05, 4075),
        (500000, 1500, 0, 4909),
        (500000, 1500, 0.05, 4093),
    ],
)
def test_1995_primary_cube_corners(p, g, x, q_kw):
    result = chf_1995_cube(p, g, x, diameter_m=0.008, scope=CUBE_SCOPE)
    assert result.heat_flux_w_m2 == q_kw * 1000
    assert "1995" in result.provenance.correlation
    assert "2006" not in result.provenance.correlation
    assert result.primary_pdf_page == 184
    assert not result.qualifies_channel_chf
    assert not result.provenance.experimentally_validated
    assert result.source_corners_unshaded


def test_1995_cube_interpolation():
    result = chf_1995_cube(400000, 1250, 0.025, diameter_m=0.008, scope=CUBE_SCOPE)
    assert result.heat_flux_w_m2 == 4121.5 * 1000  # arithmetic mean of eight primary corners
    edge = chf_1995_cube(300000, 1000, 0.025, diameter_m=0.008, scope=CUBE_SCOPE)
    assert edge.heat_flux_w_m2 == 3737 * 1000
    assert not any(v for plane in CHF_1995_CUBE_SHADED for row in plane for v in row)


@pytest.mark.parametrize(
    "kw",
    [
        {"pressure_pa": 299999},
        {"pressure_pa": 500001},
        {"mass_flux_kg_m2_s": 999},
        {"mass_flux_kg_m2_s": 1501},
        {"mass_flux_kg_m2_s": -1250},
        {"quality": -1e-12},
        {"quality": 0.050000001},
        {"pressure_pa": float("nan")},
        {"mass_flux_kg_m2_s": float("inf")},
        {"quality": float("nan")},
        {"diameter_m": 0.008000001},
        {"diameter_m": 0},
        {"scope": replace(CUBE_SCOPE, steady_vertical_upflow=False)},
        {"scope": replace(CUBE_SCOPE, pure_water=False)},
        {"scope": replace(CUBE_SCOPE, circular_tube_internal_flow=False)},
        {"scope": replace(CUBE_SCOPE, uniform_heating=False)},
        {"scope": replace(CUBE_SCOPE, equilibrium_quality=False)},
        {"scope": replace(CUBE_SCOPE, application="postCHF")},
    ],
)
def test_1995_cube_strict_rejections(kw):
    args = dict(
        pressure_pa=400000,
        mass_flux_kg_m2_s=1250,
        quality=0.025,
        diameter_m=0.008,
        scope=CUBE_SCOPE,
    )
    args.update(kw)
    with pytest.raises(DomainError):
        chf_1995_cube(**args)
