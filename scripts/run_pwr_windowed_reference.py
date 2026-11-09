"""Strict full load-rejection reference with journalled solver-history windows.

No physical state or ledger is projected. Existing Session replay already
reproduces reset_solver_before_step records. The unwindowed refusal is retained.
"""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant.qualification import (
    CASES, CRITERIA, protocol, publish, sampled_observation,
)
from atomic_model_controlled_plant.session import Session, identity
from atomic_model_plant_extension.session import OPTIONS
from atomic_model_multiphysics.provenance import sha
from atomic_model_workspace.common import digest
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_multiphysics.provenance import seal, unseal

CASE = "load_rejection_equilibrium"
RESET_INTERVAL_S = 500.0
SETTINGS = dict(method="Radau", rtol=1e-8, atol=1e-7, max_step_s=0.5, output_step_s=5.0)


def save_progress(output, session, *, source, driver, resets, trace, observations):
    """Atomic, immutable prefix receipt; never an accepted full-horizon result.

    Session checkpoints restore by replay, not by injecting a physical state.
    Both flushed logs are bound to the receipt before publishing it.
    """
    if source != identity() or driver != sha(__file__):
        raise ValueError("Cannot checkpoint changed source/runtime")
    checkpoint = session.checkpoint()
    receipt = dict(
        schema_version=1, case=CASE, source=source, driver_sha256=driver,
        time_s=session.plant.time_s, revision=session.revision,
        state_sha256=session.state_hash(), checkpoint=checkpoint,
        solver=SETTINGS, resets=deepcopy(resets),
        trace_sha256=sha(trace), observations_sha256=sha(observations),
        trace_bytes=Path(trace).stat().st_size,
        observations_bytes=Path(observations).stat().st_size,
        full_horizon_accepted=False, validated=False, plant_referenced=False,
        recovery="Source-bound Session.restore replay; not a direct state snapshot",
    )
    path = Path(output) / "progress-checkpoints" / f"step-{session.revision:06d}.json"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, seal(receipt))
    return path


def reset_history(session):
    before = session.state_hash()
    checkpoint = session.checkpoint()
    plant = session.plant
    for key in ("_hybrid", "_hybrid_signature", "_continuation", "_continuation_signature"):
        setattr(plant, key, None)
    if session.state_hash() != before or session.checkpoint() != checkpoint:
        raise ValueError("Solver reset changed physical/journal state")
    return dict(time_s=plant.time_s, state_sha256_before=before, state_sha256_after=session.state_hash())


def restore_progress(path):
    """Admit immutable log prefixes before expensive exact Session replay."""
    path = Path(path).absolute()
    receipt = unseal(read_json(path, limit=16 * 1024**2))
    if (receipt["schema_version"] != 1 or receipt["case"] != CASE
        or receipt["source"] != identity() or receipt["driver_sha256"] != sha(__file__)
        or receipt["solver"] != SETTINGS
        or any(receipt[k] is not False for k in ("full_horizon_accepted", "validated", "plant_referenced"))):
        raise ValueError("Progress source, solver or scope differs")
    horizon = CASES[CASE]["duration_s"]
    clock = receipt["time_s"]
    if (type(clock) not in (int, float) or not np.isfinite(clock) or not 0 < clock <= horizon
        or (clock != horizon and clock % RESET_INTERVAL_S != 0)):
        raise ValueError("Progress is not a supported window boundary")
    payload = unseal(receipt["checkpoint"])
    history = payload["history"]
    if (type(receipt["revision"]) is not int or not 0 < receipt["revision"] <= 10000
        or len(history) != receipt["revision"] or payload["revision"] != receipt["revision"]
        or payload["config"]["maximum_internal_step_s"] != SETTINGS["max_step_s"]):
        raise ValueError("Progress checkpoint revision/config differs")
    blobs = {}
    for name, filename in (("trace", "trace.jsonl"), ("observations", "observations.jsonl")):
        length = receipt[name + "_bytes"]
        if type(length) is not int or not 0 < length <= 32 * 1024**2:
            raise ValueError("Progress log byte budget")
        log = path.parent.parent / filename
        if log.is_symlink() or not log.is_file():
            raise ValueError("Progress log must be a regular file")
        with log.open("rb") as stream:
            blob = stream.read(length)
        if (len(blob) != length or not blob.endswith(b"\n")
            or hashlib.sha256(blob).hexdigest() != receipt[name + "_sha256"]):
            raise ValueError("Progress log prefix differs")
        blobs[name] = blob
    rows = [json.loads(line) for line in blobs["trace"].splitlines()]
    observations = [json.loads(line) for line in blobs["observations"].splitlines()]
    if len(rows) != len(history) or len(observations) != len(history) + 1 or observations[0]["time_s"] != 0:
        raise ValueError("Progress sample count differs")
    now = 0.0
    changes = dict(CASES[CASE]["changes"])
    expected_resets = []
    for index, (entry, row, observed) in enumerate(zip(history, rows, observations[1:], strict=True)):
        next_change = min([t for t in changes if t > now] + [horizon])
        duration = min(SETTINGS["output_step_s"], next_change - now, horizon - now)
        reset = now > 0 and now % RESET_INTERVAL_S == 0
        if reset:
            state = rows[index - 1]["state_sha256"]
            expected_resets.append(dict(time_s=now, state_sha256_before=state, state_sha256_after=state))
        expected_options = dict(OPTIONS, **{k: SETTINGS[k] for k in ("method", "rtol", "atol")})
        if (entry["duration_s"] != duration or entry["commands"] != changes.get(now, {})
            or entry["options"] != expected_options
            or entry["reset_solver_before_step"] is not (index == 0 or reset)):
            raise ValueError("Progress command/solver/reset journal differs")
        now += duration
        if row["time_s"] != now or observed["time_s"] != now or row["state_sha256"] != entry["state_sha256"]:
            raise ValueError("Progress clock/state differs")
    if now != clock or receipt["resets"] != expected_resets or receipt["state_sha256"] != history[-1]["state_sha256"]:
        raise ValueError("Progress final state/reset schedule differs")
    session = Session.restore(receipt["checkpoint"])
    if session.plant.time_s != clock or session.state_hash() != receipt["state_sha256"] or sampled_observation(session.plant) != observations[-1]:
        raise ValueError("Progress exact replay differs")
    return session, receipt, blobs, rows


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


def run(output, *, resume_from=None):
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output cannot be a symlink")
    if output.exists():
        raise FileExistsError(output)
    restored = restore_progress(resume_from) if resume_from is not None else None
    output.mkdir(parents=True, exist_ok=False)
    source, driver = identity(), sha(__file__)
    declared = protocol()
    publish(output / "protocol.json", declared)
    session = restored[0] if restored else Session(maximum_internal_step_s=SETTINGS["max_step_s"])
    p = session.plant
    horizon = CASES[CASE]["duration_s"]
    changes = dict(CASES[CASE]["changes"])
    started = time.monotonic()
    events, resets, rows = [], [], []
    failure = None
    maximum_quality, minimum_feed = 0.0, float("inf")
    trace, samples_path = output / "trace.jsonl", output / "observations.jsonl"
    if restored:
        _, receipt, blobs, rows = restored
        resets = deepcopy(receipt["resets"])
        events = [deepcopy(e) for e in session.history if e["events"] or e["protection_event_s"] is not None]
        maximum_quality = max(r["primary_quality"] for r in rows)
        minimum_feed = min(r["feed_flow_kg_s"] for r in rows)
        trace.write_bytes(blobs["trace"])
        samples_path.write_bytes(blobs["observations"])
        publish(output / "resume-origin.json", dict(receipt_sha256=sha(resume_from), time_s=p.time_s,
                scope="Exact source-bound prefix replay; original files unchanged"))
    try:
        with trace.open("a" if restored else "x") as log, samples_path.open("a" if restored else "x") as samples:
            if not restored:
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
                if p.time_s % RESET_INTERVAL_S == 0 or p.time_s == horizon:
                    save_progress(output, session, source=source, driver=driver,
                                  resets=resets, trace=trace, observations=samples_path)
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
    parser.add_argument("--resume-from", type=Path, help="Sealed window receipt; exact replay into a new output directory")
    args = parser.parse_args(argv)
    result = run(args.output, resume_from=args.resume_from)
    print(json.dumps(dict(accepted=result["full_horizon_accepted"], failure=result["failure"])))
    return 0 if result["full_horizon_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
