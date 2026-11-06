"""Primary-data consistency and admission census; unsupported modes stay denied."""

import hashlib
from pathlib import Path

from atomic_model_hem_qualified import eos

from .dataset import load


def identity():
    from .dataset import PIN

    return dict(
        dataset_manifest_sha256=PIN,
        code={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def evaluate():
    manifest, traces = load()
    scalars = manifest["scalars"]
    ts = scalars["Saturation Temperature_value (degC)"] + 273.15
    mean_p = scalars["Mean Pressure_value (kPa)"] * 1000.0
    source_chf = scalars["CHF_value"] * 10000.0
    chf_time = scalars["CHF_Time_value"]
    superheat_error = max(
        abs(r["surface_temperature_k"] - ts - r["wall_superheat_k"]) for r in traces["thermal"]
    )
    htc_errors = [
        abs(r["htc_w_m2_k"] - r["heat_flux_w_m2"] / r["wall_superheat_k"])
        for r in traces["thermal"]
        if r["htc_w_m2_k"] is not None and r["wall_superheat_k"] != 0
    ]
    if superheat_error > 1e-9 or not htc_errors or max(htc_errors) > 1e-6:
        raise ValueError("Published derived-field algebra is not reproducible")
    computed_ts = eos.saturation(float(mean_p))[0].temperature_k
    counters = dict(
        negative_heat_flux=0,
        nonpositive_superheat=0,
        after_published_chf=0,
        above_80percent_published_chf=0,
        outside_forced_chen_superheat=0,
    )
    for r in traces["thermal"]:
        counters["negative_heat_flux"] += r["heat_flux_w_m2"] < 0
        counters["nonpositive_superheat"] += r["wall_superheat_k"] <= 0
        counters["after_published_chf"] += r["time_s"] > chf_time
        counters["above_80percent_published_chf"] += r["heat_flux_w_m2"] >= 0.8 * source_chf
        counters["outside_forced_chen_superheat"] += not 0 <= r["wall_superheat_k"] <= 10
    return dict(
        schema_version=1,
        source=identity(),
        primary_case="B-274",
        native_counts=manifest["native_counts"],
        retained_counts=manifest["retained_counts"],
        separate_clocks=True,
        rowwise_pressure_temperature_join_permitted=False,
        max_superheat_algebra_error_k=superheat_error,
        max_htc_algebra_error_w_m2_k=max(htc_errors),
        source_mean_pressure_pa=mean_p,
        source_saturation_temperature_k=ts,
        if97_saturation_temperature_k=computed_ts,
        if97_vs_published_iapws95_difference_k=computed_ts - ts,
        saturation_comparison_status="property-formulation descriptor, not experimental thermal-history validation",
        pressure_samples_outside_qualified_eos=sum(
            not eos.P_MIN <= r["pressure_pa"] <= eos.P_MAX for r in traces["pressure"]
        ),
        thermal_regime_census=counters,
        production_forced_boiling_admitted_rows=0,
        admission_reasons=[
            "Pool geometry and G=0 are outside bounded forced-flow Chen/K1K4",
            "Large portions include post-CHF/cooling; unsupported regimes not projected",
            "Heat flux/wall temperature/HTC share reconstructed sensor inputs",
            "Instrument calibration/covariance and independent electrical drive not qualified",
            "Single case is not an independent facility-held-out validation set",
        ],
        experimental_validation_status="not_admitted_for_current_production_closures",
        validated=False,
        plant_referenced=False,
    )
