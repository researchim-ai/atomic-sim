"""Frozen NIST component comparison: production Colebrook and independent Haaland."""

import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

import atomic_model_hem_closures as production_closures
from atomic_model_validation import digitize

from .evidence import matrix

DATA = Path(__file__).resolve().parents[1] / "atomic_model_validation/data/nist_tn2294"
PIN = "0d9cf1255ba5336b29b0c933fbaecce695125782cd402ec1045858f9b2dd64c4"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(folder):
    folder = Path(folder)
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError("Symlink evidence is forbidden")
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def dataset(folder=DATA):
    folder = Path(folder)
    names = {"source.pdf", "vectors.json", "measurements.csv", "protocol.json", "NOTICE.md"}
    if inventory(folder) != names | {"manifest.json"} or sha(folder / "manifest.json") != PIN:
        raise ValueError("Frozen evidence inventory/manifest changed")
    manifest = read(folder / "manifest.json")
    for name in names:
        if sha(folder / name) != manifest["artifacts"][name]:
            raise ValueError("Frozen evidence changed: " + name)
    rows = list(csv.DictReader(io.StringIO((folder / "measurements.csv").read_text())))
    rows = [dict(series=r["series"], **{k: float(v) for k, v in r.items() if k != "series"})
            for r in rows]
    if rows != digitize.rows(read(folder / "vectors.json")) or len(rows) != 38:
        raise ValueError("Digitization mismatch")
    return read(folder / "protocol.json"), rows


def darcy_factor(reynolds, relative_roughness):
    """Haaland turbulent Darcy factor; restricted to this experimental regime."""
    if not (math.isfinite(reynolds) and reynolds >= 4000
            and math.isfinite(relative_roughness) and 0 <= relative_roughness < 0.1):
        raise ValueError("Outside turbulent reference domain")
    return (-1.8 * math.log10((relative_roughness / 3.7)**1.11 + 6.9 / reynolds))**-2


def pressure_drop(length, diameter, density, velocity, factor):
    """Signed Darcy loss, Pa; numerical reference, not measured dimensional evidence."""
    if not all(math.isfinite(x) for x in (length, diameter, density, velocity, factor)):
        raise ValueError("Nonfinite Darcy input")
    if length < 0 or diameter <= 0 or density <= 0 or factor <= 0:
        raise ValueError("Invalid Darcy input")
    return factor * length / diameter * density * velocity * abs(velocity) / 2


def closure_sources():
    root = Path(production_closures.__file__).resolve().parent
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*.py"))}


def source_identity():
    return dict(dataset_manifest_sha256=PIN,
                production_closure_version=production_closures.CLOSURE_VERSION,
                production_closure_sources=closure_sources(),
                implementation={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                digitization_sha256=sha(Path(digitize.__file__)))


def metrics(rows):
    errors = [r["relative_error"] for r in rows]
    return dict(samples=len(rows), max_abs_relative_error=max(map(abs, errors)),
                mean_relative_error=sum(errors)/len(errors),
                rms_relative_error=math.sqrt(sum(e*e for e in errors)/len(errors)),
                passed=all(r["center_passed"] for r in rows))


def comparison(protocol, observations, factor_function):
    rr = protocol["roughness_m"] / protocol["diameter_m"]
    rows = []
    for row in observations:
        re, observed = row["reynolds"], row["darcy_f"]
        factor = factor_function(re, rr)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid comparison factor")
        # Evaluate the dimensional closure at explicitly arbitrary normalization,
        # then reduce to the observed dimensionless quantity. No measured Pa.
        normalized_loss = pressure_drop(1, protocol["diameter_m"], 2, 1, factor) * protocol["diameter_m"]
        if not math.isclose(normalized_loss, factor, rel_tol=2e-14):
            raise ValueError("Darcy pressure-drop normalization mismatch")
        error = (factor-observed)/observed
        dx, dy = row["reynolds_reading_bound"], row["darcy_f_reading_bound"]
        interval = [factor_function(re+dx, rr), factor_function(re-dx, rr)]
        rows.append(dict(row, modeled_darcy_f=factor, normalized_pressure_loss=normalized_loss, relative_error=error,
                         center_passed=abs(error) <= protocol["max_relative_error"],
                         modeled_reading_interval=interval,
                         reading_envelope_gap=max(0, interval[0]-observed-dy,
                                                  observed-dy-interval[1])))
    return dict(**metrics(rows), rows=rows,
                series={s: metrics([r for r in rows if r["series"] == s])
                        for s in sorted({r["series"] for r in rows})})


def evaluate(folder=DATA):
    before = source_identity()
    protocol, observations = dataset(folder)
    production = comparison(protocol, observations, production_closures.darcy_factor)
    baseline = comparison(protocol, observations, darcy_factor)
    if before != source_identity():
        raise ValueError("Sources changed during comparison")
    return dict(schema_version=2, source=before, protocol=protocol,
                samples=len(observations), passed=production["passed"],
                primary_comparison="production_colebrook",
                comparisons=dict(production_colebrook=production, independent_haaland=baseline),
                quantity="2*delta_p*D/(rho*u^2*L) = Darcy f", units="1",
                evidence_kind="digitized_external_experiment", calibrated=False,
                dimensional_pressure_drop_experiment=False,
                reference_only=False, hem_implementation_tested=False,
                production_coefficient_tested=True,
                scope="Component Darcy coefficient only; turbulent NIST points; no laminar, network, transient or two-phase validation",
                experimental_uncertainty_complete=False,
                uncertainty_review=dict(
                    source="NIST TN 2294, printed pp. 17-18, 22-23",
                    mass_flow="each meter: 0.05% reading above 0.264 kg/s; otherwise 0.95 kg/h",
                    averaged_density_bound_kg_m3=0.707,
                    calculated_viscosity_relative_uncertainty=0.03,
                    pressure_example="p.22: mean standard uncertainty 0.4%, other conditions within 0.5%; not assigned to each figure-18 point",
                    missing=["per-point pressure-gradient fit and covariance", "diameter/roughness uncertainty",
                             "per-point flow, density, temperature and length records"],
                    propagation="f=pi^2*rho*D^5*delta_p/(8*L*mass_flow^2); requires joint input covariance",
                    colebrook_band="+/-3% correlation agreement, not measurement uncertainty"),
                uncertainty_acceptance="blocked_missing_per_point_budget",
                reading_bounds="symbol extents; not measurement confidence intervals",
                independence_from_correlation_training="not_established",
                validated=False, dynamic_hem_validated=False, two_phase_hem_validated=False,
                plant_referenced=False, validation_matrix=matrix())


def run(output):
    result = evaluate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(DATA, output / "data")
        root = Path(production_closures.__file__).resolve().parent
        for name in result["source"]["production_closure_sources"]:
            target = output / "closure_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        write(output / "report.json", result)
        write(output / "manifest.json", dict(source=source_identity(), artifacts={
            name: sha(output / name) for name in sorted(inventory(output))}))
        verify(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def verify(output):
    output = Path(output)
    files = ({"data/" + name for name in inventory(DATA)} | {"report.json"}
             | {"closure_source/" + name for name in closure_sources()})
    if inventory(output) != files | {"manifest.json"}:
        raise ValueError("Audit inventory mismatch")
    manifest = read(output / "manifest.json")
    expected = dict(source=source_identity(), artifacts={
        name: sha(output / name) for name in sorted(files)})
    if manifest != expected:
        raise ValueError("Audit provenance/hash mismatch")
    for name, expected_sha in closure_sources().items():
        if sha(output / "closure_source" / name) != expected_sha:
            raise ValueError("Frozen closure source mismatch")
    result = evaluate(output / "data")
    if read(output / "report.json") != result:
        raise ValueError("Audit recomputation mismatch")
    return result
