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
    "natural": dict(duration_s=1800.0, changes=[]),
    "natural_perturbation": dict(
        duration_s=120.0, changes=[(10.0, dict(throttle=0.049)), (30.0, dict(throttle=0.05))]
    ),
    "nominal": dict(duration_s=3600.0, changes=[]),
    "scram": dict(duration_s=60.0, changes=[(10.0, dict(scram=True))]),
    "coastdown": dict(duration_s=1800.0, changes=[(10.0, dict(scram=True, motor_off=True))]),
    "load_rejection": dict(duration_s=600.0, changes=[(10.0, dict(throttle=0.0))]),
    "load_rejection_equilibrium": dict(duration_s=7200.0, changes=[(10.0, dict(throttle=0.0))]),
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
        load_rejection_tail_s=100.0,
        load_final_normalized_rate_limit_per_s=1e-5,
        load_tail_power_and_primary_flow_limit=1e-3,
        load_tail_feed_limit=1e-2,
        extended_load_horizon_reason="7200s covers7.2 times the longest declared decay-heat time constant1000s;600s all-state equilibrium failed and remains unaccepted",
        natural_circulation="Positive coastdown flow alone does not qualify zero-head natural circulation",
        models_compared="Separate bounded flow-controlled seven-bank model; legacy and uncontrolled outcomes remain unchanged",
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
    name, *, output, rtol=1e-6, atol=1e-5, max_step_s=2.0, output_step_s=5.0, method="Radau"
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
    session = Session(
        maximum_internal_step_s=max_step_s,
        initialization="natural" if name.startswith("natural") else "nominal",
    )
    p = session.plant
    initial = p.pack().copy()
    initial_heads = natural_heads(p) if name.startswith("natural") else None
    events = []
    failed = None
    start = time.monotonic()
    changes = {float(t): cmd for t, cmd in CASES[name]["changes"]}
    horizon = CASES[name]["duration_s"]
    minimum_feed = float("inf")
    minimum_primary = float("inf")
    all_zero_head = True
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
                minimum_primary = min(minimum_primary, p.network.junctions["pump"].mass_flow_kg_s)
                if name.startswith("natural"):
                    all_zero_head = (
                        all_zero_head
                        and p.omega == 0.0
                        and p.network.junctions["pump"].pump_pressure_pa == 0.0
                        and not p.motor_on
                    )
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
    if name.startswith("natural"):
        checks["zero_shaft_and_head"] = bool(all_zero_head)
        checks["positive_primary_flow"] = bool(minimum_primary > 0.0)
        checks["buoyancy_drives_friction"] = bool(
            initial_heads["buoyancy_head_pa"] > 0.0 and abs(initial_heads["head_defect_pa"]) < 0.01
        )
        accepted = accepted and all(checks.values())
    # Load rejection must settle, not merely finish its clock.
    settling = None
    if name.startswith("load_rejection") and failed is None and p.time_s == horizon:
        values = [json.loads(line) for line in trace.read_text().splitlines()]
        tail = [r for r in values if r["time_s"] >= horizon - 100.0]
        spans = {
            k: (max(r[k] for r in tail) - min(r[k] for r in tail))
            / max(1.0, abs(sum(r[k] for r in tail) / len(tail)))
            for k in ["relative_fission", "primary_flow_kg_s", "feed_flow_kg_s"]
        }
        final_rate = float(
            np.max(abs(p.derivatives(p.time_s, p.pack())) / np.maximum(abs(p.pack()), 1.0))
        )
        settling = dict(final_normalized_rate_per_s=final_rate, tail_relative_ranges=spans)
        checks["settled_final_rate"] = final_rate <= 1e-5
        checks["settled_power"] = spans["relative_fission"] <= 1e-3
        checks["settled_primary_flow"] = spans["primary_flow_kg_s"] <= 1e-3
        checks["settled_feed_flow"] = spans["feed_flow_kg_s"] <= 1e-2
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
        natural_circulation_qualified=bool(name.startswith("natural") and accepted),
        load_rejection_settling=settling,
        initial_natural_heads=initial_heads,
    )
    publish(output / "result.json", result)
    return result


def natural_heads(plant):
    from physics.hydraulics.network import G

    head = 0.0
    loss = 0.0
    for name in ["pump", "core_out", "to_sg", "to_cold"]:
        link = plant.network.junctions[name]
        up = plant.network.volumes[link.upstream]
        down = plant.network.volumes[link.downstream]
        head -= (
            0.5
            * (up.thermo.rho_kg_m3 + down.thermo.rho_kg_m3)
            * G
            * (down.elevation_m - up.elevation_m)
        )
        donor = up if link.mass_flow_kg_s >= 0 else down
        loss += link.friction_dp(donor.thermo.rho_kg_m3)
    return dict(
        buoyancy_head_pa=head,
        friction_loss_pa=loss,
        head_defect_pa=head - loss,
        pump_head_pa=plant.network.junctions["pump"].pump_pressure_pa,
    )
