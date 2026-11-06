"""Five-component joint covariance of additive residual errors; no implicit independence."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import norm

from atomic_model_research.release import identity as release_identity
from atomic_model_training.spec import exact
from atomic_model_workspace.common import digest

COMPONENTS = ("data", "parameters", "closures", "mesh", "integrator")


def identity():
    return dict(
        base_release=release_identity(),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def covariance(value, size):
    if np.iscomplexobj(value):
        raise ValueError("Complex covariance is unsupported")
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (size, size) or not np.isfinite(a).all() or np.any(np.diag(a) < 0):
        raise ValueError("Finite covariance with nonnegative diagonal required")
    diagonal = np.diag(a)
    zero = diagonal == 0
    if np.any(a[zero, :] != 0) or np.any(a[:, zero] != 0):
        raise ValueError("Zero variance requires exactly zero cross covariance")
    positive = diagonal > 0
    if positive.any():
        scales = np.sqrt(diagonal[positive])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            correlation = a[np.ix_(positive, positive)] / scales[:, None] / scales[None, :]
        tolerance = 64 * np.finfo(float).eps * len(correlation)
        if not np.isfinite(correlation).all() or not np.allclose(
            correlation, correlation.T, rtol=0, atol=tolerance
        ):
            raise ValueError("Covariance symmetry fails after variance normalization")
        if np.min(np.linalg.eigvalsh(correlation)) < -tolerance:
            raise ValueError("Covariance is not positive semidefinite")
    # Retain supplied values; no PSD projection or cross-correlation repair.
    return a


def evaluate(spec):
    exact(
        spec,
        {
            "schema_version",
            "evidence_kind",
            "distribution",
            "confidence",
            "cases",
            "target_coverage",
        },
    )
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["distribution"] != "gaussian"
    ):
        raise ValueError("Only declared Gaussian marginal error model supported")
    if spec["evidence_kind"] not in ("numerical", "experimental"):
        raise ValueError("Unknown evidence kind")
    for key in ("confidence", "target_coverage"):
        value = spec[key]
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Probability domain(0,1)")
    cases = spec["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("Coverage case budget")
    families = {}
    ids = set()
    records = []
    z = float(norm.isf((1 - spec["confidence"]) / 2))
    for case in cases:
        exact(
            case,
            {
                "id",
                "family",
                "split",
                "observed",
                "predicted",
                "units",
                "component_covariances",
                "joint_covariance",
                "provenance",
            },
        )
        if not all(isinstance(case[k], str) and case[k] for k in ("id", "family", "provenance")):
            raise ValueError("Explicit physical family and provenance required")
        if case["id"] in ids:
            raise ValueError("Duplicate coverage case")
        ids.add(case["id"])
        split = case["split"]
        family = case["family"]
        if split not in ("calibration", "validation"):
            raise ValueError("Unknown coverage split")
        if family in families and families[family] != split:
            raise ValueError("Coverage physical-family leakage")
        families[family] = split
        observed = np.asarray(case["observed"], float)
        predicted = np.asarray(case["predicted"], float)
        if (
            observed.ndim != 1
            or not 1 <= len(observed) <= 32
            or predicted.shape != observed.shape
            or not np.isfinite(observed).all()
            or not np.isfinite(predicted).all()
            or not isinstance(case["units"], list)
            or len(case["units"]) != len(observed)
            or not all(isinstance(u, str) and u for u in case["units"])
        ):
            raise ValueError("Observed/predicted dimension or units differ")
        n = len(observed)
        exact(case["component_covariances"], COMPONENTS)
        joint = covariance(case["joint_covariance"], 5 * n)
        for i, name in enumerate(COMPONENTS):
            component = covariance(case["component_covariances"][name], n)
            if not np.allclose(
                joint[i * n : (i + 1) * n, i * n : (i + 1) * n],
                component,
                rtol=64 * np.finfo(float).eps,
                atol=0,
            ):
                raise ValueError("Joint covariance diagonal block differs: " + name)
        summation = np.tile(np.eye(n), (1, 5))
        with np.errstate(over="ignore", invalid="ignore"):
            total = summation @ joint @ summation.T
        if not np.isfinite(total).all():
            raise ValueError("Propagated covariance exceeds finite arithmetic domain")
        variance = np.diag(total)
        if np.any(variance < 0):
            raise ValueError("Negative propagated variance; no silent repair")
        widths = z * np.sqrt(variance)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = observed - predicted
        if not np.isfinite(residual).all() or not np.isfinite(widths).all():
            raise ValueError("Residual/interval exceeds finite arithmetic domain")
        covered = np.abs(residual) <= widths
        records.append(
            dict(
                id=case["id"],
                family=family,
                split=split,
                units=case["units"],
                residual=residual.tolist(),
                total_covariance=total.tolist(),
                interval_half_width=widths.tolist(),
                covered=covered.tolist(),
                component_covariances=case["component_covariances"],
                joint_covariance_sha256=digest(case["joint_covariance"]),
                provenance=case["provenance"],
            )
        )
    heldout = [value for r in records if r["split"] == "validation" for value in r["covered"]]
    if not heldout:
        raise ValueError("No held-out validation cases")
    coverage = sum(heldout) / len(heldout)
    return dict(
        source=identity(),
        input_sha256=digest(spec),
        evidence_kind=spec["evidence_kind"],
        cases=records,
        heldout_components=len(heldout),
        heldout_coverage=coverage,
        target_met=coverage >= spec["target_coverage"],
        confidence=spec["confidence"],
        assumptions=[
            "Gaussian marginal errors explicitly supplied",
            "Covariance PSD and symmetry tested after variance normalization; zero variance requires zero cross covariance",
            "All five residual-error contributions additive; cross-component covariance mandatory",
            "Coverage fraction descriptive; no independence between cases or simultaneous-coverage claim",
        ],
        physical_admissibility="requires independently admitted experimental/model/configuration evidence",
        experimental_validation_promoted=False,
        validated=False,
        plant_referenced=False,
    )
