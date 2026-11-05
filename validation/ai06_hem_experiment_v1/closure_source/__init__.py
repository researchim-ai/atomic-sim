"""Versioned, restricted water closures; SI throughout, no network solver.

Scope flags are caller attestations, not inferred flow-regime qualification.
See docs/AI_06_HEM_CLOSURES_RU.md for sources and deliberate exclusions.
"""

from dataclasses import dataclass
from math import isfinite, log10, pi, sqrt

from core.status import Applicability, DomainError
from physics.materials.iapws import ThermoState, saturation_properties

__version__ = "1.2.0"
CLOSURE_VERSION = "hem-closures/1.2.0"


@dataclass(frozen=True)
class Provenance:
    correlation: str
    source: str
    domain: str
    uncertainty: str
    version: str = CLOSURE_VERSION
    experimentally_validated: bool = False


LAMINAR = Provenance(
    "darcy-poiseuille",
    "https://archive.nptel.ac.in/content/storage2/courses/112104118/recap/11-recapa.htm",
    "Circular straight fully developed Newtonian flow; 0 < Re < 2300; constant properties",
    "Analytical idealization; no assigned experimental uncertainty or safety factor",
)
COLEBROOK = Provenance(
    "colebrook-1939",
    "https://doi.org/10.1680/ijoti.1939.13150",
    "Circular straight fully developed flow; 4000 <= Re <= 1e8; 0 <= epsilon/D <= 0.05",
    "Empirical roughness law; no calibrated uncertainty here; solve tolerance is not model accuracy",
)
LAMINAR_HEAT = Provenance(
    "laminar-nu-3.66-constant-Tw",
    "https://archive.nptel.ac.in/content/storage2/courses/103105052/AdvHeatMass_L_29.pdf",
    "Smooth circular tube; hydrodynamically/thermally developed; constant Tw and properties; Re < 2300",
    "Rounded analytical limit; no assigned experimental uncertainty; no boiling/condensation",
)
GNIELINSKI = Provenance(
    "gnielinski-smooth-inl-1.82",
    "https://mooseframework.inl.gov/docs/doxygen/modules/ADWallHeatTransferCoefficientGnielinskiMaterial_8C_source.html",
    "Smooth circular fully developed constant-property flow; 4000 <= Re <= 5e6; 0.5 <= Pr <= 2000",
    "Empirical correlation; no local calibration or assigned uncertainty; no boiling/condensation",
)
ZUBER = Provenance(
    "zuber-1959-pool-screening",
    "https://digital.library.unt.edu/ark:/67531/metadc872491/",
    "Saturated quiescent pure-water pool; large horizontal upward-facing wetted heater; normal gravity",
    "Hydrodynamic screening only; no calibrated uncertainty, design margin, DNBR or forced-flow qualification",
)
TRANSPORT_NOTE = (
    "physics.materials.iapws: IAPWS 2008 viscosity and 2011 conductivity; "
    "both omit critical enhancement; no property uncertainty propagated"
)


@dataclass(frozen=True)
class TubeScope:
    """All flags must describe the actual local flow, including negligible buoyancy/compressibility."""

    circular_straight: bool
    fully_developed: bool
    constant_properties: bool
    forced_convection: bool
    negligible_compressibility: bool
    heat_transfer_regime: str = "single_phase"


@dataclass(frozen=True)
class FrictionResult:
    darcy_factor: float
    reynolds: float
    relative_roughness: float
    colebrook_residual: float | None
    provenance: Provenance
    transport_provenance: str = TRANSPORT_NOTE


@dataclass(frozen=True)
class HeatTransferResult:
    nusselt: float
    coefficient_w_m2_k: float
    heat_flux_w_m2: float  # positive wall -> fluid
    reynolds: float
    prandtl: float
    provenance: Provenance = LAMINAR_HEAT
    transport_provenance: str = TRANSPORT_NOTE


@dataclass(frozen=True)
class PoolScope:
    saturated_quiescent: bool
    large_horizontal_upward_wetted: bool
    pure_water: bool
    application: str = "pool_screening"


@dataclass(frozen=True)
class CHFScreeningResult:
    heat_flux_w_m2: float
    pressure_pa: float
    surface_tension_n_m: float
    surface_tension_source: str
    provenance: Provenance = ZUBER
    qualifies_forced_flow: bool = False


@dataclass(frozen=True)
class ForcedFlowCHFAvailability:
    bounded_1995_cube_available: bool = True
    status: str = "partial_1995_tiny_cube_only"
    reason: str = (
        "Primary IAEA 1995 LUT PDF downloaded: numerical tables are raster images; "
        "full-table values and shading masks have not been transcribed/audited. "
        "No audited 2006 LUT payload bundled; general CHF is blocked. "
        "Only separately named chf_1995_cube admits eight manually audited 1995 corners"
    )
    primary_pdf_sha256: str = "8d039d4f8a950a1c013ddae312aee744ae5eb0e3efbaa337ce2a7fea96021c1c"
    primary_pdf_size_bytes: int = 13609780
    primary_table_pdf_pages: tuple[int, ...] = (184, 185, 186, 187, 188, 189, 190)
    sources: tuple[str, ...] = (
        "https://doi.org/10.1016/j.nucengdes.2007.02.014",
        "https://doi.org/10.2172/1159036",
        "https://oecd-nea.org/upload/docs/application/pdf/2024-01/nea_wkp_2023_1_2024-01-18_16-23-40_237.pdf",
        "https://www-pub.iaea.org/MTCD/Publications/PDF/te_1203_prn.pdf",
    )


FORCED_FLOW_CHF_AVAILABILITY = ForcedFlowCHFAvailability()


def forced_flow_chf(
    pressure_pa: float,
    mass_flux_kg_m2_s: float,
    quality: float,
    *,
    diameter_m: float,
    length_m: float,
) -> float:
    """Explicit blocked interface; never substitutes pool CHF or an invented table.

    No numeric result is available for any input in this release. This signature
    does not imply that geometry alone would qualify a future LUT implementation.
    """
    _unsupported(FORCED_FLOW_CHF_AVAILABILITY.reason)


def _positive(name: str, value: float, *, zero: bool = False) -> None:
    if not isfinite(value) or (value < 0 if zero else value <= 0):
        raise DomainError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _unsupported(message: str) -> None:
    raise DomainError(message, Applicability.UNSUPPORTED)


def _scope(scope: TubeScope) -> None:
    if scope.heat_transfer_regime != "single_phase":
        _unsupported("Two-phase friction, boiling heat transfer, CHF and postCHF are unsupported")
    if not all(
        flag is True
        for flag in (
            scope.circular_straight,
            scope.fully_developed,
            scope.constant_properties,
            scope.forced_convection,
            scope.negligible_compressibility,
        )
    ):
        _unsupported("Requires straight circular developed constant-property forced flow")


def _transport(state: ThermoState) -> tuple[float, float]:
    if state.applicability != Applicability.IN_DOMAIN or state.region not in (1, 2):
        _unsupported("Only in-domain IF97 region 1/2; never use mixed region-4 transport")
    # Deliberately restricted to the backend's available subcritical saturation envelope.
    if not (273.15 <= state.T_k <= 623.15 and 611.212677 <= state.p_pa <= 16.5291642526e6):
        raise DomainError(
            "Restricted water envelope: 273.15..623.15 K, 611.212677..16529164.2526 Pa"
        )
    for name in ("rho_kg_m3", "viscosity_pa_s", "conductivity_w_m_k", "cp_j_kg_k"):
        _positive(name, getattr(state, name))
    liquid, vapor = saturation_properties(state.p_pa)
    if (state.region == 1 and state.T_k >= liquid.T_k) or (
        state.region == 2 and state.T_k <= vapor.T_k
    ):
        _unsupported("Single-phase closures require strictly off-saturation bulk water")
    return state.viscosity_pa_s, state.conductivity_w_m_k


def friction_factor(
    state: ThermoState,
    mass_flux_kg_m2_s: float,
    diameter_m: float,
    *,
    scope: TubeScope,
    roughness_m: float = 0.0,
) -> FrictionResult:
    """Darcy (not Fanning) factor. Signed G allowed; zero flow is outside this API."""
    _scope(scope)
    mu, _ = _transport(state)
    _positive("abs(mass_flux)", abs(mass_flux_kg_m2_s))
    _positive("diameter", diameter_m)
    _positive("roughness", roughness_m, zero=True)
    re = abs(mass_flux_kg_m2_s) * diameter_m / mu
    rr = roughness_m / diameter_m
    factor, residual, provenance = _darcy(re, rr)
    return FrictionResult(factor, re, rr, residual, provenance)


def darcy_factor(reynolds: float, relative_roughness: float = 0.0) -> float:
    """Dimensionless Darcy factor for the declared circular single-phase flow scope.

    Caller must enforce physical scope at every stage; Re alone cannot detect phases,
    geometry, development or compressibility. Transition and zero flow reject.
    """
    return _darcy(reynolds, relative_roughness)[0]


def _darcy(re: float, rr: float) -> tuple[float, float | None, Provenance]:
    _positive("reynolds", re)
    if not isfinite(re) or not 0 <= rr <= 0.05:
        raise DomainError("Finite Reynolds number and epsilon/D <= 0.05 required")
    if re < 2300:
        factor = 64 / re
        _positive("Darcy factor", factor)
        return factor, None, LAMINAR
    if not 4000 <= re <= 1e8:
        raise DomainError("Transition Re in [2300,4000) and Re > 1e8 unsupported")
    # Solve x + 2 log10(rr/3.7 + 2.51*x/Re) = 0, x = 1/sqrt(f).
    # Positive monotonic residual; bracket valid over the declared Re/roughness range.
    low, high = 1.0, 100.0
    for _ in range(100):
        x = (low + high) / 2
        residual = x + 2 * log10(rr / 3.7 + 2.51 * x / re)
        if abs(residual) <= 1e-12:
            return 1 / x**2, residual, COLEBROOK
        if residual > 0:
            high = x
        else:
            low = x
    raise DomainError("Colebrook root failed to converge")


def friction_pressure_drop_pa(
    state: ThermoState,
    mass_flux_kg_m2_s: float,
    diameter_m: float,
    length_m: float,
    *,
    scope: TubeScope,
    roughness_m: float = 0.0,
) -> float:
    """Signed local resistance dp = f L/D G|G|/(2 rho), excluding gravity/acceleration."""
    _positive("length", length_m)
    result = friction_factor(
        state, mass_flux_kg_m2_s, diameter_m, scope=scope, roughness_m=roughness_m
    )
    value = (
        result.darcy_factor
        * length_m
        / diameter_m
        * mass_flux_kg_m2_s
        * abs(mass_flux_kg_m2_s)
        / (2 * state.rho_kg_m3)
    )
    if not isfinite(value):
        raise DomainError("Nonfinite pressure drop")
    return value


def heat_transfer(
    state: ThermoState,
    mass_flux_kg_m2_s: float,
    diameter_m: float,
    wall_temperature_k: float,
    *,
    scope: TubeScope,
    constant_wall_temperature: bool,
    thermally_developed: bool,
    roughness_m: float = 0.0,
    correlation: str = "auto",
    length_m: float | None = None,
) -> HeatTransferResult:
    """Developed smooth-tube laminar/Gnielinski transfer, with strict phase guards.

    correlation: auto, laminar, gnielinski. Optional length is checked if supplied;
    development is an explicit caller attestation, never inferred from cell length.
    """
    if correlation not in ("auto", "laminar", "gnielinski"):
        _unsupported("Unknown heat-transfer correlation")
    if length_m is not None:
        _positive("length", length_m)
    friction = friction_factor(
        state, mass_flux_kg_m2_s, diameter_m, scope=scope, roughness_m=roughness_m
    )
    if constant_wall_temperature is not True or thermally_developed is not True or roughness_m != 0:
        _unsupported("Heat transfer requires smooth tube, constant Tw and thermal development")
    _positive("wall_temperature", wall_temperature_k)
    if not 273.15 <= wall_temperature_k <= 623.15:
        raise DomainError("Wall temperature outside restricted water envelope")
    tsat = saturation_properties(state.p_pa)[0].T_k
    if (state.region == 1 and wall_temperature_k >= tsat) or (
        state.region == 2 and wall_temperature_k <= tsat
    ):
        _unsupported("Wall at/across saturation: boiling/condensation heat transfer unsupported")
    pr = state.cp_j_kg_k * state.viscosity_pa_s / state.conductivity_w_m_k
    _positive("Prandtl", pr)
    re = friction.reynolds
    if re < 2300:
        if correlation == "gnielinski":
            raise DomainError("Gnielinski requires 4000 <= Re <= 5e6")
        nu, provenance = 3.66, LAMINAR_HEAT
    else:
        if correlation == "laminar":
            raise DomainError("Laminar heat transfer requires Re < 2300")
        if not (4000 <= re <= 5e6 and 0.5 <= pr <= 2000):
            raise DomainError("Gnielinski requires 4000 <= Re <= 5e6 and 0.5 <= Pr <= 2000")
        # INL primary implementation's smooth-tube Darcy factor; intentionally
        # distinct from the momentum Colebrook solve. No rough-tube substitution.
        f = (1.82 * log10(re) - 1.64) ** -2
        nu = f / 8 * (re - 1000) * pr / (1 + 12.7 * sqrt(f / 8) * (pr ** (2 / 3) - 1))
        provenance = GNIELINSKI
    h = nu * state.conductivity_w_m_k / diameter_m
    q = h * (wall_temperature_k - state.T_k)
    if not all(isfinite(v) for v in (h, q, pr)):
        raise DomainError("Nonfinite heat-transfer result")
    _positive("heat-transfer coefficient", h)
    return HeatTransferResult(nu, h, q, re, pr, provenance)


def single_phase_transfer(
    state: ThermoState,
    velocity_m_s: float,
    diameter_m: float,
    length_m: float,
    roughness_m: float,
    wall_temperature_k: float,
    *,
    scope: TubeScope,
    constant_wall_temperature: bool,
    thermally_developed: bool,
    correlation: str = "auto",
) -> HeatTransferResult:
    """Stage-local network adapter; returns h in coefficient_w_m2_k, no cached state.

    Positive length is mandatory but does not establish upstream development.
    Signed velocity is supported; zero flow and transitional Re reject.
    """
    _positive("length", length_m)
    _positive("abs(velocity)", abs(velocity_m_s))
    return heat_transfer(
        state,
        state.rho_kg_m3 * velocity_m_s,
        diameter_m,
        wall_temperature_k,
        scope=scope,
        constant_wall_temperature=constant_wall_temperature,
        thermally_developed=thermally_developed,
        roughness_m=roughness_m,
        correlation=correlation,
        length_m=length_m,
    )


def zuber_pool_chf_screening(
    pressure_pa: float,
    surface_tension_n_m: float,
    *,
    surface_tension_source: str,
    scope: PoolScope,
) -> CHFScreeningResult:
    """Saturated pool CHF scale, never a channel CHF limit or postCHF closure.

    Caller supplies surface tension at the same saturation pressure with provenance;
    the current IF97 backend has no surface-tension function. Gravity fixed at 9.80665.
    """
    if scope.application != "pool_screening" or not all(
        flag is True
        for flag in (
            scope.saturated_quiescent,
            scope.large_horizontal_upward_wetted,
            scope.pure_water,
        )
    ):
        _unsupported(
            "Zuber is restricted to saturated pool screening; forced-flow CHF/postCHF rejected"
        )
    _positive("pressure", pressure_pa)
    _positive("surface_tension", surface_tension_n_m)
    if not isinstance(surface_tension_source, str) or not surface_tension_source.strip():
        raise DomainError("Surface tension requires a source at the supplied saturation pressure")
    if not 611.212677 <= pressure_pa <= 16.5291642526e6:
        raise DomainError("Saturation backend limited to 273.15..623.15 K")
    liquid, vapor = saturation_properties(pressure_pa)
    latent = vapor.h_j_kg - liquid.h_j_kg
    density_gap = liquid.rho_kg_m3 - vapor.rho_kg_m3
    _positive("latent heat", latent)
    _positive("density difference", density_gap)
    q = (
        pi
        / 24
        * latent
        * sqrt(vapor.rho_kg_m3)
        * (surface_tension_n_m * 9.80665 * density_gap) ** 0.25
    )
    _positive("CHF screening flux", q)
    return CHFScreeningResult(q, pressure_pa, surface_tension_n_m, surface_tension_source)


CHF_1995_CUBE_PROVENANCE = Provenance(
    "iaea-tecdoc-1203-1995-lut-single-cube",
    "https://www-pub.iaea.org/MTCD/Publications/PDF/te_1203_prn.pdf",
    "8 mm tube only; steady vertical upflow water, uniform heating; "
    "300000 <= p <= 500000 Pa; 1000 <= G <= 1500 kg/(m2 s); 0 <= x <= 0.05",
    "Unshaded published table corners, manually read; no uncertainty assigned; "
    "nominal local LUT only, no length/surface corrections or CHF/postCHF qualification",
)
# Native raster anchors: PDF p.184 / printed p.176, Table II.I. Source PDF
# SHA-256 is in FORCED_FLOW_CHF_AVAILABILITY. Raster is 3758 x 5638 pixels.
# Axes in header y=0..420; p=300 rows G=1000/1500 y=2320..2550;
# p=500 rows y=3820..4050. Columns x=0 and .05 at raster x~1640 and ~1790.
# Two AI agent contexts independently read the same primary image and agreed on all
# eight values and unshaded status (parent confirmation in task conversation).
# This is transcription review, not independent experimental validation.
# Shape [pressure][mass_flux][thermohydraulic_quality], values in kW/m2.
CHF_1995_CUBE_KW_M2 = (
    ((4189.0, 3285.0), (4209.0, 3339.0)),
    ((4873.0, 4075.0), (4909.0, 4093.0)),
)
CHF_1995_CUBE_SHADED = (
    ((False, False), (False, False)),
    ((False, False), (False, False)),
)


@dataclass(frozen=True)
class CHF1995CubeScope:
    pure_water: bool
    circular_tube_internal_flow: bool
    steady_vertical_upflow: bool
    uniform_heating: bool
    equilibrium_quality: bool
    application: str = "nominal_local_chf_table"


@dataclass(frozen=True)
class CHF1995CubeResult:
    heat_flux_w_m2: float
    provenance: Provenance = CHF_1995_CUBE_PROVENANCE
    primary_pdf_sha256: str = FORCED_FLOW_CHF_AVAILABILITY.primary_pdf_sha256
    primary_pdf_page: int = 184
    source_corners_unshaded: bool = True
    qualifies_channel_chf: bool = False


def chf_1995_cube(
    pressure_pa: float,
    mass_flux_kg_m2_s: float,
    quality: float,
    *,
    diameter_m: float,
    scope: CHF1995CubeScope,
) -> CHF1995CubeResult:
    """Trilinear interpolation of exactly eight unshaded 1995 LUT values.

    Not the complete 1995/2006 LUT; no extrapolation, clipping, safety-factor,
    geometry/length corrections, transient CHF qualification or postCHF transfer.
    x is local equilibrium thermohydraulic quality; x=0 is the saturated endpoint.
    """
    if scope.application != "nominal_local_chf_table" or not all(
        flag is True
        for flag in (
            scope.pure_water,
            scope.circular_tube_internal_flow,
            scope.steady_vertical_upflow,
            scope.uniform_heating,
            scope.equilibrium_quality,
        )
    ):
        _unsupported(
            "1995 cube requires steady vertical upflow, uniformly heated water tube and equilibrium quality"
        )
    if not isfinite(diameter_m) or diameter_m != 0.008:
        raise DomainError("1995 cube diameter must equal 0.008 m; no diameter correction")
    fractions = []
    for name, value, low, high in (
        ("pressure", pressure_pa, 300000.0, 500000.0),
        ("mass_flux", mass_flux_kg_m2_s, 1000.0, 1500.0),
        ("quality", quality, 0.0, 0.05),
    ):
        if not isfinite(value) or not low <= value <= high:
            raise DomainError(f"1995 cube {name} outside [{low}, {high}]; no extrapolation")
        fractions.append((value - low) / (high - low))
    q = 0.0
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                # Any shaded corner would invalidate this admitted interpolation cell.
                if CHF_1995_CUBE_SHADED[i][j][k]:
                    _unsupported("1995 interpolation cell has a shaded corner")
                weight = 1.0
                for index, fraction in zip((i, j, k), fractions):
                    weight *= fraction if index else 1 - fraction
                q += weight * CHF_1995_CUBE_KW_M2[i][j][k]
    return CHF1995CubeResult(q * 1000.0)
