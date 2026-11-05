"""Embedded candidate metadata only; never read checkout files at runtime.

Admitted from ai06_dynamic_data_candidate_v1/manifest.json, 2026-11-05.
The factual Table 1 subset is not a matched dynamic validation dataset.
"""

CANDIDATE_ID = "alamgir-kan-lienhard-1980-figure4"
PRIMARY_URL = (
    "https://engines.egr.uh.edu/sites/engines/files/talks/"
    "ExperimentalStudyOfDepressurization.pdf"
)


def matrix():
    rows = [dict(regime="steady_single_phase_friction", status="compared_limited",
                 evidence="NIST TN 2294 figure 18; four repeat series, one apparatus",
                 missing="per-point measurement uncertainty and dimensional raw records")]
    requirements = {
        "evaporation": "heated two-phase mass/energy inventory, heat flux and void fraction time histories",
        "condensation": "vapor inlet state, heat removal, condensate rate and pressure time histories",
        "flashing": "inlet enthalpy, depressurization, quality/void fraction and discharge histories",
        "blowdown": "vessel inventory, geometry, boundary pressure and mass/pressure/temperature histories",
    }
    for regime, missing in requirements.items():
        row = dict(regime=regime, status="blocked_missing_real_data", evidence=None,
                   missing=missing + "; traceable calibration and measurement uncertainty")
        if regime in {"flashing", "blowdown"}:
            row.update(
                status="blocked_missing_matched_dynamic_data",
                candidate_id=CANDIDATE_ID,
                primary_url=PRIMARY_URL,
                evidence="Primary published hot-water rapid depressurization candidate; metadata only",
                candidate_metadata=dict(
                    authors="Md. Alamgir; C. Y. Kan; J. H. Lienhard",
                    title="An Experimental Study of the Rapid Depressurization of Hot Water",
                    year=1980,
                    journal="Journal of Heat Transfer 102, 433-438",
                    admitted_manifest="ai06_dynamic_data_candidate_v1/manifest.json",
                    factual_table="Table 1, printed p.435; four runs, nine numeric fields each",
                    runs=["HW-18", "HW-17", "HW-20", "HW-27"],
                    available="initial pressure/temperature, minimum pressure and tabulated depressurization quantities",
                    csv_sha256="8d74f91af068fc597e263a3a7b742f769e2750b6221229e40758d5eefe6de263",
                    pressure_history="Figure 4 photographed oscillograms, not numerical time histories",
                    digitized_history_available=False,
                    raw_sensor_records_available=False,
                    matched_dynamic_data_available=False,
                    comparison_performed=False,
                    experimental_acceptance=False,
                    uncertainty_complete=False,
                    archive_redistribution_authorized=False,
                    mismatch_flags=[
                        "moving_plug_opening_dynamics_and_area_time_boundary_missing",
                        "metastability_and_nucleation_delay_outside_equilibrium_HEM",
                        "full_initial_axial_and_wall_temperature_profiles_missing",
                        "downstream_pressure_history_missing",
                        "numerical_pressure_histories_not_digitized",
                        "no_massflow_or_void_time_series_in_extract",
                    ],
                ),
                missing=missing + "; matched opening/boundary and initial conditions; "
                        "nonequilibrium scope resolution; digitized histories with uncertainty",
            )
        rows.append(row)
    return rows
