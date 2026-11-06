"""Predeclared full-horizon campaign; a failed prefix can never be accepted."""

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_workspace.common import digest

from .session import Session, identity

CASES = {
    "nominal": dict(duration_s=3600.0, changes=[]),
    "scram": dict(duration_s=60.0, changes=[(10.0, dict(scram=True))]),
    "coastdown": dict(duration_s=1800.0, changes=[(10.0, dict(scram=True, motor_off=True))]),
    "load_rejection": dict(duration_s=600.0, changes=[(10.0, dict(throttle=0.0))]),
    "feed_disturbance": dict(
        duration_s=120.0,
        changes=[(10.0, dict(feed_bias_kg_s=-10.0)), (30.0, dict(feed_bias_kg_s=0.0))],
    ),
    "sink_disturbance": dict(
        duration_s=120.0,
        changes=[(10.0, dict(sink_temperature_k=305.0)), (30.0, dict(sink_temperature_k=300.0))],
    ),
}
CRITERIA = dict(
    energy_relative=1e-6,
    mass_relative=1e-10,
    boron_relative=1e-10,
    nominal_power_drift=1e-5,
    scram_final_relative_fission=0.1,
    maximum_primary_quality=1e-4,
    minimum_feed_flow_kg_s=0.0,
)


def protocol():
    return dict(
        schema_version=1,
        source=identity(),
        cases=CASES,
        criteria=CRITERIA,
        status="predeclared; numerical qualification only",
        natural_circulation="Positive coastdown flow alone does not qualify zero-head natural circulation",
        models_compared="Separate reduced seven-bank model; frozen legacy failures remain unchanged",
        validated=False,
        plant_referenced=False,
    )


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(dict(payload=payload, sha256=digest(payload)), stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_case(
    name, *, output, rtol=1e-6, atol=1e-5, max_step_s=0.5, output_step_s=5.0, method="Radau"
):
    if name not in CASES:
        raise ValueError("Unknown campaign case")
    if not np.isfinite([rtol, atol, max_step_s, output_step_s]).all() or not (
        0 < rtol <= 1e-4
        and 0 < atol <= 1e-2
        and 0.01 <= max_step_s <= 2.0
        and 0 < output_step_s <= 5.0
    ):
        raise ValueError("Campaign solver/step domain")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    declared = protocol()
    publish(output / "protocol.json", declared)
    session = Session(maximum_internal_step_s=max_step_s)
    p = session.plant
    initial = p.pack().copy()
    events = []
    failed = None
    start = time.monotonic()
    changes = {float(t): cmd for t, cmd in CASES[name]["changes"]}
    horizon = CASES[name]["duration_s"]
    minimum_feed = float("inf")
    maximum_x = 0.0
    maximum_power_drift = 0.0
    trace = output / "trace.jsonl"
    try:
        with trace.open("x") as stream:
            while p.time_s < horizon:
                now = p.time_s
                next_change = min([t for t in changes if t > now] + [horizon])
                dt = min(output_step_s, next_change - now, horizon - now)
                result = session.advance(
                    dt, commands=changes.get(now, {}), rtol=rtol, atol=atol, method=method
                )
                transition = result["transition"]
                x = max(
                    float(p.network.volumes[n].thermo.quality or 0.0)
                    for n in ["core", "hot_leg", "sg_primary", "cold_leg"]
                )
                feed = p.network.junctions["feed"].mass_flow_kg_s
                minimum_feed = min(minimum_feed, feed)
                maximum_x = max(maximum_x, x)
                maximum_power_drift = max(
                    maximum_power_drift, abs(float(p.kinetics_state[0]) - initial[0])
                )
                row = dict(
                    time_s=p.time_s,
                    relative_fission=float(p.kinetics_state[0]),
                    primary_quality=x,
                    feed_flow_kg_s=feed,
                    primary_flow_kg_s=p.network.junctions["pump"].mass_flow_kg_s,
                    omega_rad_s=p.omega,
                    effective_throttle=p.effective_steam_throttle,
                    energy=deepcopy(p.last_energy_ledger),
                    state_sha256=transition["state_sha256"],
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if transition["events"] or transition["protection_event_s"] is not None:
                    events.append(transition)
                if int(p.time_s) % 100 == 0:
                    print(name, method, rtol, p.time_s, flush=True)
    except Exception as exc:
        failed = dict(type=type(exc).__name__, message=str(exc), last_accepted_time_s=p.time_s)
    ledger = session.ledger.report() if session.revision else None
    accepted = failed is None and p.time_s == horizon and ledger is not None
    checks = {}
    if ledger:
        checks = {
            k: ledger[k] <= CRITERIA[k]
            for k in ["energy_relative", "mass_relative", "boron_relative"]
        }
        checks["feed_nonnegative"] = minimum_feed >= CRITERIA["minimum_feed_flow_kg_s"]
        checks["primary_scope"] = maximum_x <= CRITERIA["maximum_primary_quality"]
        if name == "nominal":
            checks["stationary_fission"] = maximum_power_drift <= CRITERIA["nominal_power_drift"]
        if name in ("scram", "coastdown"):
            checks["shutdown_fission"] = (
                float(p.kinetics_state[0]) < CRITERIA["scram_final_relative_fission"]
            )
        accepted = accepted and all(checks.values())
    checkpoint = session.checkpoint()
    publish(output / "checkpoint-envelope.json", checkpoint)
    result = dict(
        schema_version=1,
        source=declared["source"],
        protocol_sha256=digest(declared),
        case=name,
        requested_horizon_s=horizon,
        completed_horizon_s=p.time_s,
        full_horizon_accepted=bool(accepted),
        checks={k: bool(v) for k, v in checks.items()},
        failure=failed,
        ledger=ledger,
        initialization=session.initialization,
        solver=dict(
            method=method, rtol=rtol, atol=atol, max_step_s=max_step_s, output_step_s=output_step_s
        ),
        extrema=dict(
            minimum_feed_flow_kg_s=minimum_feed if session.revision else None,
            maximum_primary_quality=maximum_x,
            maximum_fission_drift=maximum_power_drift,
        ),
        final_state=list(map(float, p.pack())),
        events=events,
        steps=session.revision,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint["sha256"],
        wall_seconds=time.monotonic() - start,
        validated=False,
        plant_referenced=False,
        natural_circulation_qualified=False,
    )
    publish(output / "result.json", result)
    return result
