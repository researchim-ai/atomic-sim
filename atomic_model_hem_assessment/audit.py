"""Strict admission census, not validation inferred from the existence of data."""

import csv
import gzip
import io
from collections import Counter
from pathlib import Path

from atomic_model_hem_qualified import closures, eos
from atomic_model_hem_qualified.model import signature as model_signature
from atomic_model_workspace.common import digest, keys, number, read

from . import acoustics
from .importer import BOUNDS, COLUMNS, FIELDS, bounded_zip, encode, extract, finite, sha

DATA = Path(__file__).parent / "data" / "hiper"
PIN = "d8fb751b48916f99505d4660c09bc3f678efca80dd9c97884d9a27ad613aa242"


def dataset(folder=DATA):
    folder = Path(folder)
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError("Dataset symlink")
    if {p.name for p in folder.iterdir()} != {"manifest.json", "NOTICE.md", "observations.csv.gz"}:
        raise ValueError("Unexpected dataset inventory")
    manifest_bytes = (folder / "manifest.json").read_bytes()
    if sha(manifest_bytes) != PIN:
        raise ValueError("Dataset manifest changed")
    manifest = read(folder / "manifest.json")
    for name, value in manifest["artifacts"].items():
        if sha((folder / name).read_bytes()) != value:
            raise ValueError("Dataset artifact changed")
    with gzip.open(folder / "observations.csv.gz", "rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("Observation expansion limit")
    reader = csv.DictReader(io.StringIO(raw.decode()))
    if reader.fieldnames != COLUMNS:
        raise ValueError("Observation columns changed")
    rows = []
    text = {"row_id", "source_workbook", "coolant"}
    for row in reader:
        row = {k: v if k in text else finite(v) for k, v in row.items()}
        for k in ("run", "source_row"):
            if row[k] != int(row[k]):
                raise ValueError("Noninteger source identity")
            row[k] = int(row[k])
        for field in BOUNDS:
            if not row[field + "_lower"] <= row[field] <= row[field + "_upper"]:
                raise ValueError("Invalid observation interval")
        rows.append(row)
    if len(rows) != 12320 or len({r["row_id"] for r in rows}) != 12320:
        raise ValueError("Observation cardinality/identity")
    return manifest, rows


def identity():
    return dict(
        dataset_manifest_sha256=PIN,
        dataset_artifacts={
            name: sha((DATA / name).read_bytes())
            for name in ("manifest.json", "observations.csv.gz", "NOTICE.md")
        },
        acoustic_data=acoustics.source_identity(),
        model=model_signature(),
        implementation={
            p.name: sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def admission(row):
    """All independent failures, using actual frozen closure calls when eligible."""
    reasons = []
    p, x, D, L, G, tw = (
        row[k]
        for k in (
            "pressure_pa",
            "quality",
            "diameter_m",
            "length_m",
            "mass_flux_kg_m2_s",
            "wall_temperature_k",
        )
    )
    if not eos.P_MIN + 100 < p < eos.P_MAX - 100:
        reasons.append("outside_dynamic_pressure_domain")
    if not 0 <= x <= 0.1:
        reasons.append("outside_wet_quality_domain")
    if not 0 < L <= 6:
        reasons.append("outside_boiling_heated_length_domain")
    if not 0 < row["wall_superheat_k"] <= 10:
        reasons.append("outside_wall_superheat_domain")
    table = D == 0.008 and (
        (3e5 <= p <= 5e5 and 1000 <= G <= 1500 and 0 <= x <= 0.05)
        or (15e6 <= p <= 16e6 and 1000 <= G <= 2000 and 0 <= x <= 0.1)
    )
    biasi = (
        0.27e6 <= p <= 14e6
        and 100 <= G <= 600
        and 0 <= x <= 0.1
        and 0.005 <= D <= 0.0375
        and 0.2 <= L <= 6
    )
    if not table and not biasi:
        reasons.append("no_supported_chf_correlation_for_geometry_and_state")
    if not reasons:
        state = eos.px(p, x)
        try:
            closures.boiling(
                state,
                G,
                D,
                tw,
                L,
                scope={k: True for k in closures.HEAT_FLAGS},
                chf_model="table1995" if table else "biasi_inl",
            )
        except ValueError as exc:
            reasons.append("production_closure_rejected:" + str(exc))
    return dict(
        local_closure_eligible=not reasons,
        reasons=reasons,
        system_experiment_eligible=False,
        system_reasons=[
            "prescribed_steady_throughflow_not_closed_finite_graph",
            "no_time_histories_for_dynamic_validation",
            "flow_orientation_and_gravity_model_require_independent_review",
        ],
    )


def envelope_comparison(measured, modeled):
    """Absolute asymmetric bounded intervals; no Gaussian/95% interpretation."""
    for interval in (measured, modeled):
        keys(interval, ("value", "lower", "upper"))
        for key in interval:
            number(interval[key], -1e30, 1e30, key)
        if not interval["lower"] <= interval["value"] <= interval["upper"]:
            raise ValueError("Interval does not enclose nominal")
    residual = modeled["value"] - measured["value"]
    return dict(
        residual=residual,
        relative_residual=None if measured["value"] == 0 else residual / abs(measured["value"]),
        envelope_gap=max(
            0.0, modeled["lower"] - measured["upper"], measured["lower"] - modeled["upper"]
        ),
        overlaps=modeled["lower"] <= measured["upper"] and measured["lower"] <= modeled["upper"],
        physical_acceptance=False,
        confidence_interpretation="None; reported bounded endpoints, no covariance assumption",
    )


def assess_target(target):
    keys(
        target,
        (
            "id",
            "pressure_pa",
            "temperature_k",
            "requires_post_chf",
            "requires_metastability",
            "requires_open_throughflow",
        ),
    )
    if not isinstance(target["id"], str) or not target["id"]:
        raise ValueError("Target identifier")
    reasons = []
    for name, lo, hi in [
        ("pressure_pa", eos.P_MIN + 100, eos.P_MAX - 100),
        ("temperature_k", eos.T_MIN, eos.T_MAX),
    ]:
        bounds = target[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Target interval")
        a, b = [number(v, 0, 1e12, name) for v in bounds]
        if a > b:
            raise ValueError("Reversed target interval")
        outside = a <= lo or b >= hi if name == "pressure_pa" else a < lo or b > hi
        if outside:
            reasons.append("uncovered_" + name)
    for field in ("requires_post_chf", "requires_metastability", "requires_open_throughflow"):
        if type(target[field]) is not bool:
            raise ValueError("Target requirements must be booleans")
        if target[field]:
            reasons.append("unsupported_" + field)
    return dict(
        target=target,
        supported_envelope=not reasons,
        reasons=reasons,
        plant_referenced=False,
        physical_acceptance=False,
    )


def evaluate(folder=DATA):
    source = identity()
    manifest, rows = dataset(folder)
    assessed = [dict(row_id=r["row_id"], run=r["run"], **admission(r)) for r in rows]
    counts = Counter(reason for row in assessed for reason in row["reasons"])
    runs = sorted({r["run"] for r in rows})
    summary = dict(
        schema_version=1,
        source=source,
        dataset_id=manifest["dataset_id"],
        runs=len(runs),
        observations=len(rows),
        local_closure_eligible=sum(r["local_closure_eligible"] for r in assessed),
        matched_system_experiments=0,
        rejection_counts=dict(sorted(counts.items())),
        ranges={
            k: [min(r[k] for r in rows), max(r[k] for r in rows)]
            for k in FIELDS
            if k not in ("run", "coolant")
        },
        per_run={
            str(run): dict(
                observations=sum(r["run"] == run for r in rows),
                eligible=sum(r["run"] == run and r["local_closure_eligible"] for r in assessed),
            )
            for run in runs
        },
        assessment=assessed,
        uncertainty=manifest["uncertainty"],
        acoustic_screen=acoustics.screen(),
        comparison_performed=False,
        error_metrics=None,
        physical_acceptance=False,
        dynamic_hem_validated=False,
        plant_referenced=False,
        conclusion="Real primary measurements imported; no matched production closure/system benchmark. No parameter tuning or range extension.",
    )
    dataset(folder)
    if identity() != source:
        raise ValueError("Assessment source changed during computation")
    return summary


def verify_reimport(archive):
    manifest, _ = dataset()
    raw = Path(archive).read_bytes()
    rows, workbooks, _ = extract(raw)
    with bounded_zip(raw) as archive:
        readmes = [n for n in archive.namelist() if n.endswith("Readme_Publication.pdf")]
        if len(readmes) != 1 or sha(archive.read(readmes[0])) != manifest["readme_pdf_sha256"]:
            raise ValueError("Primary README changed")
    normalized = encode(rows)
    bundled = (DATA / "observations.csv.gz").read_bytes()
    if workbooks != manifest["workbooks"] or gzip.decompress(normalized) != gzip.decompress(
        bundled
    ):
        raise ValueError("Primary workbook or deterministic normalization mismatch")
    return dict(
        workbooks=len(workbooks),
        observations=len(rows),
        normalized_csv_exact=True,
        compressed_bytes_exact=normalized == bundled,
        archive_bytes_exact=sha(raw) == manifest["download_archive_sha256"],
    )


def audit(path):
    saved = read(path)
    keys(saved, ("payload", "sha256"))
    if saved["sha256"] != digest(saved["payload"]) or saved["payload"]["source"] != identity():
        raise ValueError("Assessment digest/source mismatch")
    fresh = evaluate()
    if digest(fresh) != saved["sha256"]:
        raise ValueError("Assessment recomputation mismatch")
    return fresh
