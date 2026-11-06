"""Independent final system acceptance, including load-rejection settling.

Thresholds are fixed before the full transient finishes. A completed horizon is
necessary, but cannot alone qualify an unsettled load rejection.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomic_model_plant_extension.initialization import solve
from atomic_model_plant_extension.pwr import SourceBoundPWR
from atomic_model_plant_extension.session import identity
from scripts.extended_plant_release import envelope, verify_run

THRESHOLDS = dict(
    nominal_relative_state_difference=1e-5,
    coastdown_relative_state_difference=1e-3,
    load_final_normalized_rate_per_s=1e-5,
    load_tail_power_flow_relative_range=1e-3,
    load_tail_feed_relative_range=1e-2,
    load_tail_duration_s=100.0,
)


def protocol():
    return dict(
        source=identity(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds=THRESHOLDS,
        validated=False,
        plant_referenced=False,
    )


def relative_difference(left, right):
    a = np.array(left, dtype=float)
    b = np.array(right, dtype=float)
    return float(np.max(abs(a - b) / np.maximum(1.0, np.maximum(abs(a), abs(b)))))


def final_rate(folder):
    from sdk.api import apply_commands

    folder = Path(folder)
    r = verify_run(folder)
    checkpoint = envelope(folder / "checkpoint-envelope.json")["payload"]
    plant = SourceBoundPWR()
    solve(plant)
    for entry in checkpoint["history"]:
        apply_commands(plant, entry["commands"])
        if entry["protection_event_s"] is not None:
            plant.scram_latched = True
    state = np.array(r["final_state"])
    plant.time_s = r["completed_horizon_s"]
    plant.refresh_algebraic(state)
    plant._feed_open = bool(state[plant.feed_index] > 0.0 or plant.feed_terms()[0] > 0.0)
    derivative = plant.derivatives(plant.time_s, state)
    return float(np.max(abs(derivative) / np.maximum(abs(state), 1.0)))


def settling(folder):
    folder = Path(folder)
    r = verify_run(folder)
    rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    begin = r["requested_horizon_s"] - THRESHOLDS["load_tail_duration_s"]
    tail = [row for row in rows if row["time_s"] >= begin]
    if len(tail) < 2 or tail[0]["time_s"] > begin:
        raise ValueError("Missing full settling tail")
    ranges = {}
    for key in ["relative_fission", "primary_flow_kg_s", "feed_flow_kg_s"]:
        values = [row[key] for row in tail]
        ranges[key] = (max(values) - min(values)) / max(1.0, abs(sum(values) / len(values)))
    rate = final_rate(folder)
    checks = dict(
        final_rate=rate <= THRESHOLDS["load_final_normalized_rate_per_s"],
        tail_power=ranges["relative_fission"] <= THRESHOLDS["load_tail_power_flow_relative_range"],
        tail_primary_flow=ranges["primary_flow_kg_s"]
        <= THRESHOLDS["load_tail_power_flow_relative_range"],
        tail_feed=ranges["feed_flow_kg_s"] <= THRESHOLDS["load_tail_feed_relative_range"],
    )
    return dict(
        final_normalized_rate_per_s=rate,
        tail_relative_ranges=ranges,
        checks=checks,
        settled=all(checks.values()),
    )


def assess(campaign, coast, peer, natural, perturb, events):
    campaign = Path(campaign)
    folders = {
        name: campaign / name
        for name in ["nominal_radau", "nominal_bdf", "scram", "load_rejection", "feed", "sink"]
    }
    folders.update(coastdown=Path(coast), coastdown_peer=Path(peer))
    reports = {k: verify_run(v) for k, v in folders.items()}
    checks = {k: r["full_horizon_accepted"] for k, r in reports.items()}
    nominal = relative_difference(
        reports["nominal_radau"]["final_state"], reports["nominal_bdf"]["final_state"]
    )
    transient = relative_difference(
        reports["coastdown"]["final_state"], reports["coastdown_peer"]["final_state"]
    )
    checks["nominal_independent_time_control"] = (
        nominal <= THRESHOLDS["nominal_relative_state_difference"]
    )
    checks["coastdown_independent_time_control"] = (
        transient <= THRESHOLDS["coastdown_relative_state_difference"]
    )
    settled = settling(folders["load_rejection"])
    checks["load_rejection_settled"] = settled["settled"]
    natural_r = envelope(Path(natural) / "result.json")
    perturb_r = envelope(Path(perturb) / "result.json")
    from scripts.qualify_extended_natural import driver_identity

    if natural_r["source"] != driver_identity() or perturb_r["source"] != driver_identity():
        raise ValueError("Natural source changed")
    checks["zero_head_natural_1800s"] = (
        natural_r["full_horizon_accepted"] and natural_r["completed_horizon_s"] == 1800.0
    )
    checks["natural_heat_removal_perturbation"] = (
        perturb_r["full_horizon_accepted"] and perturb_r["completed_horizon_s"] == 120.0
    )
    event_r = envelope(events)
    if event_r["source"]["plant"] != identity():
        raise ValueError("Event source changed")
    checks["three_output_clocks_and_phase_restarts"] = (
        len(event_r["runs"]) == 3 and event_r["full_replays"] == 12
    )
    return dict(
        protocol=protocol(),
        checks={k: bool(v) for k, v in checks.items()},
        all_system_checks_passed=bool(all(checks.values())),
        nominal_relative_state_difference=nominal,
        coastdown_relative_state_difference=transient,
        load_rejection_settling=settled,
        evidence={k: str(v) for k, v in folders.items()},
        natural_circulation_evidence=str(natural),
        natural_perturbation_evidence=str(perturb),
        events_evidence=str(events),
        validated=False,
        plant_referenced=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declare", action="store_true")
    p.add_argument("--protocol", type=Path)
    p.add_argument("--output", required=True, type=Path)
    for name in ["campaign", "coast", "peer", "natural", "perturb", "events"]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    from atomic_model_plant_extension.qualification import publish

    if not a.declare and (a.protocol is None or envelope(a.protocol) != protocol()):
        raise ValueError("Predeclared acceptance protocol/source mismatch")
    result = (
        protocol()
        if a.declare
        else assess(a.campaign, a.coast, a.peer, a.natural, a.perturb, a.events)
    )
    publish(a.output, result)


if __name__ == "__main__":
    main()
