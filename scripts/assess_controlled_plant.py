"""Source-bound system decision: full horizons, time control and phase replay."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant.qualification import publish
from atomic_model_controlled_plant.session import identity
from atomic_model_phase_assessment.dataset import load
from atomic_model_phase_assessment.vessel import identity as phase_identity
from scripts.controlled_plant_release import admission, verify_run
from scripts.extended_plant_release import envelope

LIMITS = dict(
    nominal_relative_state_difference=1e-5,
    coastdown_relative_state_difference=1e-3,
    rod_stop_spread_s=1e-10,
    source_open_spread_s=1e-3,
)
NAMES = [
    "coastdown",
    "coastdown_peer",
    "feed",
    "load_rejection",
    "natural",
    "natural_perturbation",
    "nominal_radau",
    "nominal_bdf",
    "scram",
    "sink",
]


def protocol():
    return dict(
        source=identity(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=LIMITS,
        validated=False,
        plant_referenced=False,
    )


def difference(left, right):
    a = np.array(left)
    b = np.array(right)
    return float(np.max(abs(a - b) / np.maximum(1.0, np.maximum(abs(a), abs(b)))))


def assess(root, events, phase, primary_import):
    root = Path(root)
    reports = {k: verify_run(root / k) for k in NAMES}
    checks = {k: r["full_horizon_accepted"] for k, r in reports.items()}
    # Independently reevaluate the final settling derivative in current code.
    from atomic_model_controlled_plant import Session
    from sdk.api import apply_commands

    load_folder = root / "load_rejection"
    c = envelope(load_folder / "checkpoint-envelope.json")["payload"]
    audit = Session(**c["config"])
    plant = audit.plant
    for entry in c["history"]:
        apply_commands(plant, entry["commands"])
        if entry["protection_event_s"] is not None:
            plant.scram_latched = True
    y = np.array(reports["load_rejection"]["final_state"])
    plant.time_s = reports["load_rejection"]["completed_horizon_s"]
    plant.refresh_algebraic(y)
    plant._feed_open = bool(y[plant.feed_index] > 0.0 or plant.feed_terms()[0] > 0.0)
    measured = float(np.max(abs(plant.derivatives(plant.time_s, y)) / np.maximum(abs(y), 1.0)))
    recorded = reports["load_rejection"]["load_rejection_settling"]["final_normalized_rate_per_s"]
    checks["current_code_settling_reevaluation"] = (
        abs(measured - recorded) <= 1e-12 and measured <= 1e-5
    )
    initial = reports["nominal_radau"]["initialization"]
    checks["direct_ranked_initialization"] = (
        initial["constrained_jacobian_rank"] == initial["constrained_unknowns"] == 40
        and initial["warmup_steps"] == 0
        and not initial["parameter_changes"]
        and initial["max_mass_rate_kg_s"] <= 1e-6
        and initial["max_inventory_energy_rate_w"] <= 1.0
        and max(initial["momentum_residuals_pa"].values()) <= 0.01
    )
    nominal = difference(
        reports["nominal_radau"]["final_state"], reports["nominal_bdf"]["final_state"]
    )
    coast = difference(
        reports["coastdown"]["final_state"], reports["coastdown_peer"]["final_state"]
    )
    checks["nominal_two_methods_and_accuracies"] = (
        nominal <= LIMITS["nominal_relative_state_difference"]
    )
    checks["coastdown_two_temporal_controls"] = (
        coast <= LIMITS["coastdown_relative_state_difference"]
    )
    checks["load_to_full_equilibrium_7200s"] = (
        reports["load_rejection"]["case"] == "load_rejection_equilibrium"
        and reports["load_rejection"]["completed_horizon_s"] == 7200.0
    )
    e = envelope(events)
    from scripts import qualify_controlled_events

    if (
        e["source"]["plant"] != identity()
        or e["source"]["driver_sha256"]
        != hashlib.sha256(Path(qualify_controlled_events.__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("Controlled event source differs")
    checks["three_output_clocks_and12_restarts"] = (
        len(e["runs"]) == 3
        and e["full_replays"] == 12
        and e["first_event_spreads"]["rod_stop"]["spread_s"] <= LIMITS["rod_stop_spread_s"]
        and e["first_event_spreads"]["dump_source_opened"]["spread_s"]
        <= LIMITS["source_open_spread_s"]
    )
    p = envelope(phase)
    if p["source"] != phase_identity():
        raise ValueError("Four-phenomenon source differs")
    checks["four_independent_phase_programs"] = (
        {r["case"] for r in p["cases"]}
        == {"boiling", "condensation", "flashing", "limited_blowdown"}
        and sum(len(r["runs"]) for r in p["cases"]) == 12
        and all(run["restart_equal"] for row in p["cases"] for run in row["runs"])
    )
    manifest, _ = load()
    import json

    primary = json.loads(Path(primary_import).read_text())
    checks["independent_primary_import"] = (
        primary["primary_sha256"] == manifest["primary_sha256"]
        and primary["native_counts"] == manifest["native_counts"]
        and primary["compared_counts"] == manifest["retained_counts"]
    )
    scenarios = admission(
        [root / name for name in NAMES], [root / name / "replay.json" for name in NAMES]
    )
    checks["all10_histories_recomputed_and_admitted"] = (
        len(scenarios["admitted"]) == 10
        and not scenarios["rejected"]
        and not scenarios["candidates_awaiting_full_replay"]
    )
    return dict(
        protocol=protocol(),
        checks={k: bool(v) for k, v in checks.items()},
        all_three_blocks_numerical_acceptance=bool(all(checks.values())),
        nominal_relative_state_difference=nominal,
        coastdown_relative_state_difference=coast,
        completed_horizons_s={k: r["completed_horizon_s"] for k, r in reports.items()},
        ledger_max_relative={
            k: max(r["ledger"][k] for r in reports.values())
            for k in ["energy_relative", "mass_relative", "boron_relative"]
        },
        equilibrium_postconditions=reports["load_rejection"]["load_rejection_settling"],
        event_spreads=e["first_event_spreads"],
        independent_phase_cases=4,
        phase_trajectories=12,
        fully_replayed_programs=10,
        agent_scenario_manifest=scenarios,
        scope="Numerical qualification of the exact reduced source-bound configuration/programs, not a plant reference or experimental attestation",
        experimental_validation_status="not_established; pool B274 not admitted for current forced boiling closure",
        validated=False,
        plant_referenced=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declare", action="store_true")
    for name in ["root", "events", "phase", "primary-import", "protocol"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if not a.declare and (a.protocol is None or envelope(a.protocol) != protocol()):
        raise ValueError("Predeclared system protocol changed")
    result = protocol() if a.declare else assess(a.root, a.events, a.phase, a.primary_import)
    publish(a.output, result)


if __name__ == "__main__":
    main()
