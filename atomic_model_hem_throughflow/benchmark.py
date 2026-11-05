"""Frozen conditional local wall-law benchmark; all exclusions/failures retained.

Measured/reduced p,x,G,q condition a local closure, not an independent whole-loop
prediction. Run-disjoint descriptive cohorts, no parameter fitting is supported.
"""

from collections import Counter

import numpy as np

from atomic_model_hem_assessment.audit import dataset, envelope_comparison
from atomic_model_hem_assessment.audit import identity as data_identity
from atomic_model_hem_qualified import eos

from . import closures
from .model import signature

PROTOCOL = dict(
    id="hiper-conditional-local-chen-v1",
    partition="even run IDs: development descriptive cohort; odd run IDs: held-out evaluation cohort",
    selection="All primary rows admitted by input domain, observed saturated-preCHF regime, and upstream heat-flux interval consistency; never by model residual",
    observable="Inner wall temperature and heat-flux consistency conditioned on published local p,x,G,D,q",
    quality="Published enthalpy-equilibrium quality, not directly measured void fraction",
    heated_distance="Published local l/di multiplied by published di; no reconstruction of a whole-loop boundary condition",
    minimum_holdout_runs=5,
    minimum_holdout_rows=20,
    maximum_mean_absolute_relative_heat_flux_error=0.2,
    minimum_nominal_wall_interval_coverage=0.95,
    allowed_prediction_failures=0,
    criteria_kind="Predeclared engineering screening targets, not a confidence level or calibrated accuracy guarantee",
    fitting_performed=False,
    automatic_physical_promotion=False,
)


def metrics(rows):
    candidates = [r for r in rows if r["candidate"]]
    predictions = [r for r in candidates if r["prediction"] is not None]
    failures = len(candidates) - len(predictions)
    return dict(
        candidate_rows=len(candidates),
        runs=len({r["run"] for r in candidates}),
        predicted_rows=len(predictions),
        prediction_failures=failures,
        mean_absolute_relative_heat_flux_error=None
        if not predictions
        else float(np.mean([abs(r["relative_heat_flux_error"]) for r in predictions])),
        mean_absolute_wall_temperature_error_k=None
        if not predictions
        else float(np.mean([abs(r["wall_temperature_error_k"]) for r in predictions])),
        nominal_wall_interval_coverage=None
        if not candidates
        else sum(r["wall_interval_contains_prediction"] for r in predictions) / len(candidates),
        complete_uncertainty_cases=0,
        unresolved_superheat_rows=sum(
            r.get("observed_htc_identifiability") is None
            or r["observed_htc_identifiability"]["upper_unbounded"]
            for r in predictions
        ),
    )


def evaluate():
    source = dict(model=signature(), data=data_identity())
    _, observations = dataset()
    assessed = []
    prefix = {}
    for row in observations:
        run = row["run"]
        lo, hi = prefix.get(run, (float("-inf"), float("inf")))
        lo = max(lo, row["heat_flux_w_m2_lower"])
        hi = min(hi, row["heat_flux_w_m2_upper"])
        prefix[run] = (lo, hi)
        p, x, G, D, q, tw = [
            row[k]
            for k in (
                "pressure_pa",
                "quality",
                "mass_flux_kg_m2_s",
                "diameter_m",
                "heat_flux_w_m2",
                "wall_temperature_k",
            )
        ]
        L = row["length_over_diameter"] * D
        reasons = []
        if not 15e6 <= p <= 16e6:
            reasons.append("outside_corrected_table_pressure")
        if not 0 <= x <= 0.1:
            reasons.append("outside_wet_quality")
        if not 300 <= G <= 2000:
            reasons.append("outside_mass_flux")
        if not 0.005 <= D <= 0.025 or not 0.2 <= L <= 10 or L / D < 5:
            reasons.append("outside_geometry")
        if lo > hi:
            reasons.append("upstream_uniform_flux_inconsistent_with_reported_bounds")
        if 15e6 <= p <= 16e6:
            ts = eos.px(p, 0.0).temperature_k
            if not 0 < tw - ts <= 10 or not 0 < row["wall_superheat_k"] <= 10:
                reasons.append("observed_outside_saturated_preCHF_superheat")
        if q <= 0:
            reasons.append("nonpositive_heat_flux")
        result = dict(
            row_id=row["row_id"],
            run=run,
            cohort="development" if run % 2 == 0 else "holdout",
            candidate=not reasons,
            exclusions=reasons,
            prediction=None,
            prediction_failure=None,
        )
        if not reasons:
            scope = {k: True for k in closures.FLAGS}
            try:
                prediction = closures.wall_temperature(p, x, G, D, q, L, scope=scope)
                modeled_q = closures.heat_flux(p, x, G, D, tw, L, scope=scope)
                wall_interval = dict(
                    value=tw,
                    lower=row["wall_temperature_k_lower"],
                    upper=row["wall_temperature_k_upper"],
                )
                wall = prediction["wall_temperature_k"]
                pressure_bounds = (row["pressure_pa_lower"], row["pressure_pa_upper"])
                htc_interval = None
                if eos.P_MIN <= pressure_bounds[0] <= pressure_bounds[1] <= eos.P_MAX:
                    ts_lower = eos.px(pressure_bounds[0], 0.0).temperature_k
                    ts_upper = eos.px(pressure_bounds[1], 0.0).temperature_k
                    superheat_lower = row["wall_temperature_k_lower"] - ts_upper
                    superheat_upper = row["wall_temperature_k_upper"] - ts_lower
                    htc_interval = dict(
                        superheat_k=[superheat_lower, superheat_upper],
                        lower_w_m2_k=None
                        if superheat_upper <= 0
                        else row["heat_flux_w_m2_lower"] / superheat_upper,
                        upper_w_m2_k=None
                        if superheat_lower <= 0
                        else row["heat_flux_w_m2_upper"] / superheat_lower,
                        upper_unbounded=superheat_lower <= 0,
                        meaning="Worst-case box from published p/Twall/q endpoints; unknown confidence/covariance; not fitted",
                    )
                result.update(
                    observed_htc_identifiability=htc_interval,
                    prediction=prediction,
                    modeled_heat_flux_at_measured_wall_w_m2=modeled_q,
                    relative_heat_flux_error=modeled_q / q - 1,
                    wall_temperature_error_k=wall - tw,
                    wall_interval_contains_prediction=wall_interval["lower"]
                    <= wall
                    <= wall_interval["upper"],
                    wall_interval_comparison=envelope_comparison(
                        wall_interval, dict(value=wall, lower=wall, upper=wall)
                    ),
                    uncertainty_limit="Nominal inputs only; measured/reduced input covariances and quality uncertainty unestablished; no complete propagated interval",
                    uniform_flux_feasible_interval_w_m2=[lo, hi],
                    heated_distance_m=L,
                )
            except ValueError as exc:
                result["prediction_failure"] = str(exc)
        assessed.append(result)
    holdout = metrics([r for r in assessed if r["cohort"] == "holdout"])
    development = metrics([r for r in assessed if r["cohort"] == "development"])
    enough = (
        holdout["runs"] >= PROTOCOL["minimum_holdout_runs"]
        and holdout["candidate_rows"] >= PROTOCOL["minimum_holdout_rows"]
    )
    passed = (
        enough
        and holdout["prediction_failures"] == 0
        and holdout["mean_absolute_relative_heat_flux_error"] <= 0.2
        and holdout["nominal_wall_interval_coverage"] >= 0.95
    )
    if dict(model=signature(), data=data_identity()) != source:
        raise ValueError("Benchmark source changed")
    return dict(
        schema_version=1,
        source=source,
        protocol=PROTOCOL,
        cohort_metrics=dict(development=development, holdout=holdout),
        exclusions=dict(
            sorted(Counter(reason for r in assessed for reason in r["exclusions"]).items())
        ),
        assessed_rows=len(assessed),
        rows=assessed,
        screening_status="insufficient_matched_data"
        if not enough
        else "passed_unvalidated"
        if passed
        else "failed",
        physical_acceptance=False,
        dynamic_hem_validated=False,
        plant_referenced=False,
        limitations=[
            "Conditional local closure test, not full-loop prediction",
            "Steady HIPER observations do not establish transient boiling/condensation/flashing dynamics",
            "Full experimental uncertainty and input covariance are not established",
            "Uniform upstream heat flux is consistent with published bounds, not proven exact",
        ],
    )
