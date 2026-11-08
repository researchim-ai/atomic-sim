"""Strict full load-rejection reference with journalled solver-history windows.

No physical state or ledger is projected. Existing Session replay already
reproduces reset_solver_before_step records. The unwindowed refusal is retained.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant.qualification import (
    CASES, CRITERIA, protocol, publish, sampled_observation,
)
from atomic_model_controlled_plant.session import Session, identity
from atomic_model_multiphysics.provenance import sha
from atomic_model_workspace.common import digest

CASE = "load_rejection_equilibrium"
RESET_INTERVAL_S = 500.0
SETTINGS = dict(method="Radau", rtol=1e-8, atol=1e-7, max_step_s=0.5, output_step_s=5.0)


def reset_history(session):
    before = session.state_hash()
    checkpoint = session.checkpoint()
    plant = session.plant
    for key in ("_hybrid", "_hybrid_signature", "_continuation", "_continuation_signature"):
        setattr(plant, key, None)
    if session.state_hash() != before or session.checkpoint() != checkpoint:
        raise ValueError("Solver reset changed physical/journal state")
    return dict(time_s=plant.time_s, state_sha256_before=before, state_sha256_after=session.state_hash())


def tail_checks(plant, samples, declared):
    horizon = CASES[CASE]["duration_s"]
    tail = [r for r in samples if r["time_s"] >= horizon - declared["load_rejection_tail_s"]]
    if not tail or tail[-1]["time_s"] != horizon:
        raise ValueError("Missing full settling tail")
    spans = {
        key: (max(r[key] for r in tail) - min(r[key] for r in tail))
        / max(1.0, abs(sum(r[key] for r in tail) / len(tail)))
        for key in ("relative_fission", "primary_flow_kg_s", "feed_flow_kg_s")
    }
    rate = float(np.max(abs(plant.derivatives(plant.time_s, plant.pack())) / np.maximum(abs(plant.pack()), 1.0)))
    checks = dict(
        settled_final_rate=rate <= declared["load_final_normalized_rate_limit_per_s"],
        settled_power=spans["relative_fission"] <= declared["load_tail_power_and_primary_flow_limit"],
        settled_primary_flow=spans["primary_flow_kg_s"] <= declared["load_tail_power_and_primary_flow_limit"],
        settled_feed_flow=spans["feed_flow_kg_s"] <= declared["load_tail_feed_limit"],
    )
    return checks, dict(final_normalized_rate_per_s=rate, tail_relative_ranges=spans)


def run(output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    source, driver = identity(), sha(__file__)
    declared = protocol()
    publish(output / "protocol.json", declared)
    session = Session(maximum_internal_step_s=SETTINGS["max_step_s"])
    p = session.plant
    horizon = CASES[CASE]["duration_s"]
    changes = dict(CASES[CASE]["changes"])
    started = time.monotonic()
    events, resets, rows = [], [], []
    failure = None
    maximum_quality, minimum_feed = 0.0, float("inf")
    trace, samples_path = output / "trace.jsonl", output / "observations.jsonl"
    try:
        with trace.open("x") as log, samples_path.open("x") as samples:
            samples.write(json.dumps(sampled_observation(p), allow_nan=False) + "\n")
            samples.flush()
            while p.time_s < horizon:
                now = p.time_s
                if now > 0 and now % RESET_INTERVAL_S == 0:
                    resets.append(reset_history(session))
                next_change = min([t for t in changes if t > now] + [horizon])
                duration = min(SETTINGS["output_step_s"], next_change - now, horizon - now)
                result = session.advance(duration, commands=changes.get(now, {}), method=SETTINGS["method"], rtol=SETTINGS["rtol"], atol=SETTINGS["atol"])
                transition = result["transition"]
                quality = max(float(p.network.volumes[n].thermo.quality or 0.0) for n in ("core", "hot_leg", "sg_primary", "cold_leg"))
                maximum_quality = max(maximum_quality, quality)
                feed = float(p.network.junctions["feed"].mass_flow_kg_s)
                minimum_feed = min(minimum_feed, feed)
                row = dict(time_s=p.time_s, relative_fission=float(p.kinetics_state[0]),
                           primary_quality=quality, feed_flow_kg_s=feed,
                           primary_flow_kg_s=float(p.network.junctions["pump"].mass_flow_kg_s),
                           omega_rad_s=float(p.omega), effective_throttle=p.effective_steam_throttle,
                           energy=deepcopy(p.last_energy_ledger), state_sha256=transition["state_sha256"])
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                samples.write(json.dumps(sampled_observation(p), allow_nan=False) + "\n"); samples.flush()
                if transition["events"] or transition["protection_event_s"] is not None:
                    events.append(transition)
                if int(p.time_s) % 100 == 0:
                    print(CASE, "windowed", p.time_s, flush=True)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), last_accepted_time_s=p.time_s)
    ledger = session.ledger.report() if session.revision else None
    checks = {}
    settling = None
    if ledger:
        checks = {key: ledger[key] <= CRITERIA[key] for key in ("energy_relative", "mass_relative", "boron_relative")}
        checks.update(feed_nonnegative=minimum_feed >= CRITERIA["minimum_feed_flow_kg_s"],
                      primary_scope=maximum_quality <= CRITERIA["maximum_primary_quality"])
    if failure is None and p.time_s == horizon:
        extra, settling = tail_checks(p, rows, declared)
        checks.update(extra)
    unchanged = source == identity() and driver == sha(__file__)
    accepted = failure is None and p.time_s == horizon and ledger is not None and all(checks.values()) and unchanged
    checkpoint = session.checkpoint()
    publish(output / "checkpoint-envelope.json", checkpoint)
    result = dict(
        schema_version=1, source=source, protocol_sha256=digest(declared), case=CASE,
        requested_horizon_s=horizon, completed_horizon_s=p.time_s, full_horizon_accepted=bool(accepted),
        checks={k: bool(v) for k, v in checks.items()}, failure=failure, ledger=ledger,
        initialization=session.initialization, solver=SETTINGS, final_state=list(map(float, p.pack())),
        events=events, steps=session.revision, trace_sha256=sha(trace), observations_sha256=sha(samples_path),
        checkpoint_sha256=checkpoint["sha256"], wall_seconds=time.monotonic() - started,
        load_rejection_settling=settling, validated=False, plant_referenced=False,
        reference_execution=dict(driver_sha256=driver, reset_interval_s=RESET_INTERVAL_S,
                                 resets=resets, source_unchanged=unchanged,
                                 scope="Explicit solver-history windows, unchanged stocks/ledger/tolerances; replay journal includes every reset"),
    )
    publish(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args(argv).output)
    print(json.dumps(dict(accepted=result["full_horizon_accepted"], failure=result["failure"])))
    return 0 if result["full_horizon_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
