"""Identifiability, sensitivities and conditional measurement precision; no tuning."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy

from atomic_model_hem_assessment.audit import dataset
from atomic_model_hem_assessment.audit import identity as data_identity
from atomic_model_hem_throughflow.benchmark import evaluate as frozen_benchmark
from atomic_model_hem_throughflow.model import signature as flow_signature
from atomic_model_workspace.common import digest, keys, read

from . import cohorts, precision, prediction, reference, region

PROTOCOL = dict(
    id="conditional-chen-identifiability-v1",
    parameters=["convective_gain", "nucleate_gain"],
    domain="Nonnegative unbounded diagnostic gains, nominal p/x/G/D held fixed; no imposed upper bound or regularization",
    measurement="Published asymmetric wall-temperature and heat-flux endpoints, intersected with declared local model domain",
    maximum_gain_width_at_unit_reference=0.2,
    sensitivity_relative_rank_tolerance=1e-8,
    precision_relative_htc_box_width=0.2,
    physical_status="No automatic calibration/validation or production coefficient update",
    data_reuse="Retrospective data previously examined; no fresh validation claim",
)


def identity():
    return dict(
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__),
        model=flow_signature(),
        data=data_identity(),
        implementation={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def lean(envelope):
    return {k: envelope[k] for k in ("id", "lower_basis", "upper_basis", "observed_q_bounds")}


def agree(lp, oracle):
    if lp["status"] != oracle["status"]:
        raise ValueError("LP/reference feasibility disagreement")
    if lp["bounds"] is None:
        return
    for a, b in zip(lp["bounds"], oracle["bounds"]):
        for key in ("lower", "upper"):
            x, y = a[key], b[key]
            if (x is None) != (y is None) or x is not None and abs(x - y) > 2e-6 * (1 + abs(x)):
                raise ValueError("LP/reference parameter bound disagreement")


def analyse_rows(rows):
    envelopes = [prediction.envelope(r) for r in rows]
    condensed = [lean(e) for e in envelopes]
    lp = region.solve(condensed)
    A, b = region.constraints(condensed)
    oracle = reference.enumerate_region(A, b)
    agree(lp, oracle)
    sensitivities = []
    precision_rows = []
    for row in rows:
        p, x, G, D, q = [
            row[k]
            for k in ("pressure_pa", "quality", "mass_flux_kg_m2_s", "diameter_m", "heat_flux_w_m2")
        ]
        predicted = prediction.predict(p, x, G, D, q, row["length_over_diameter"] * D)
        sensitivities.append(dict(row_id=row["row_id"], **predicted))
        precision_rows.append(
            dict(
                row_id=row["row_id"],
                **precision.requirement(row, PROTOCOL["precision_relative_htc_box_width"]),
            )
        )
    singular = np.linalg.svd(
        np.array([r["sensitivity_k_per_gain"] for r in sensitivities]), compute_uv=False
    )
    rank = (
        int(sum(singular > singular[0] * PROTOCOL["sensitivity_relative_rank_tolerance"]))
        if len(singular) and singular[0] > 0
        else 0
    )
    reasons = []
    if lp["status"] != "bounded":
        reasons.append("unbounded_or_infeasible_parameter_region")
    elif any(
        x["upper"] - x["lower"] > PROTOCOL["maximum_gain_width_at_unit_reference"]
        for x in lp["bounds"]
    ):
        reasons.append("parameter_interval_wider_than_declared_resolution")
    if rank < 2:
        reasons.append("sensitivity_rank_deficient")
    scales = [
        r["maximum_joint_scale"] for r in precision_rows if r["maximum_joint_scale"] is not None
    ]
    return dict(
        observations=len(rows),
        runs=len({r["run"] for r in rows}),
        parameter_region=lp,
        independent_vertex_reference=oracle,
        direct_unboundedness_certificate=reference.monotone_cone_certificate(condensed),
        conditional_identifiable_at_declared_resolution=not reasons,
        identification_rejections=reasons,
        sensitivity=dict(
            singular_values=singular.tolist(),
            rank=rank,
            condition_number=None if rank < 2 else float(singular[0] / singular[-1]),
            interpretation="Local unit-gain Jacobian rank, not proof of bounded interval identification",
        ),
        sensitivity_rows=sensitivities,
        measurement_envelopes=envelopes,
        precision_rows=precision_rows,
        precision_summary=dict(
            rows_with_unbounded_current_htc=sum(
                r["current"]["htc_upper_w_m2_k"] is None for r in precision_rows
            ),
            rows_limited_by_heat_flux_noise=sum(
                r["status"] == "heat_flux_uncertainty_floor_exceeds_target" for r in precision_rows
            ),
            maximum_joint_scale_range=None if not scales else [min(scales), max(scales)],
        ),
        physical_acceptance=False,
        parameter_update=None,
    )


def evaluate():
    before = identity()
    manifest, all_rows = dataset()
    benchmark = frozen_benchmark()
    ids = {r["row_id"] for r in benchmark["rows"] if r["candidate"]}
    selected = [r for r in all_rows if r["row_id"] in ids]
    if len(ids) != len(selected):
        raise ValueError("Frozen benchmark/primary data binding")
    grouped, binding = cohorts.split(selected, manifest)
    development = analyse_rows(grouped["development"])
    evaluation = analyse_rows(grouped["evaluation"])
    # No evaluation rows are supplied to the development feasible-set solve.
    reasons = list(development["identification_rejections"]) + [
        "retrospective_data_already_examined",
        "input_quality_covariance_and_full_uncertainty_unestablished",
    ]
    after = identity()
    if before != after:
        raise ValueError("Identification source/data changed")
    return dict(
        schema_version=1,
        source=before,
        protocol=PROTOCOL,
        cohorts=binding,
        development=development,
        evaluation=evaluation,
        frozen_production_benchmark=dict(
            protocol_id=benchmark["protocol"]["id"],
            screening_status=benchmark["screening_status"],
            original_parity_holdout_metrics=benchmark["cohort_metrics"]["holdout"],
        ),
        calibration_status="rejected",
        calibration_rejections=reasons,
        parameter_update=None,
        physical_acceptance=False,
        validated=False,
        plant_referenced=False,
    )


def audit(path):
    record = read(path)
    keys(record, ("payload", "sha256"))
    if record["sha256"] != digest(record["payload"]) or record["payload"]["source"] != identity():
        raise ValueError("Identification digest/source mismatch")
    expected = evaluate()
    if digest(expected) != record["sha256"]:
        raise ValueError("Identification replay mismatch")
    return expected
