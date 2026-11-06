"""Issue the bounded numerical delivery receipt after all factual gates pass."""

import json
from pathlib import Path

from atomic_model_controlled_plant.session import identity
from scripts.extended_plant_release import envelope


def main():
    summary = envelope("validation/ai050607_controlled_summary_v1.json")
    checks = json.loads(Path("validation/ai050607_controlled_checks_v1.json").read_text())
    cold = json.loads(Path("validation/ai050607_controlled_installed_v1.json").read_text())
    if (
        not summary["all_three_blocks_numerical_acceptance"]
        or checks["source"] != identity()
        or cold["source"] != identity()
    ):
        raise ValueError("Delivery evidence incomplete or stale")
    tasks = [
        "AI-05.03",
        "AI-05.05",
        "AI-05.08",
        "AI-06.06",
        *[f"AI-07.{i:02d}" for i in range(1, 9)],
    ]
    result = dict(
        schema_version=1,
        delivery_id="ai050607-controlled-numerical-v1",
        date="2026-11-06",
        source=identity(),
        model_id="pwr-reduced-flow-controlled",
        model_version="1.0.0",
        accepted_tasks=tasks,
        all_three_blocks_numerical_acceptance=True,
        regression_tests=checks["regression"]["passed"],
        installed_tests=checks["installed_tests"]["passed"],
        full_current_code_replayed_programs=10,
        phase_cases=4,
        phase_trajectories=12,
        completed_horizons_s=summary["completed_horizons_s"],
        evidence=[
            "validation/ai050607_controlled_summary_v1.json",
            "validation/ai050607_controlled_checks_v1.json",
            "validation/ai050607_controlled_installed_v1.json",
            "validation/ai07_controlled_configuration_v1.json",
            "validation/ai06_pool_b274_independent_import_v2.json",
            "validation/ai06_pool_b274_offline_primary_v1.json",
            "artifacts/ai050607_controlled_final_v1",
            "artifacts/ai07_controlled_events_final_v1.json",
            "artifacts/ai06_four_phenomena_v2.json",
        ],
        archive="artifacts/ai050607_controlled_publication_v1.tar.gz",
        archive_receipt="validation/ai050607_controlled_publication_v1.json",
        scope="Declared numerical configuration and exact accepted programs; not arbitrary regimes, a real plant or experimental validation",
        experimental_validation_status="not_established; pool B274 screened but not admitted for current forced-flow closure",
        legacy_failed_horizons_preserved=True,
        load600_all_state_equilibrium_failed_and_preserved=True,
        calibration_performed=False,
        validated=False,
        plant_referenced=False,
    )
    with Path("validation/ai050607_controlled_delivery_v1.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
