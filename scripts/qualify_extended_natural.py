"""Source-bound zero-head heated natural-circulation qualification.

Initial fill is defined by core pressure16MPa and the charging solver. Algebraic
homotopy is not a simulated warm-up or a trajectory from the nominal inventory.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from threading import RLock

from atomic_model_plant_extension.pwr import SourceBoundPWR
from atomic_model_plant_extension.qualification import publish
from atomic_model_plant_extension.session import Session, identity
from atomic_model_workspace.common import digest
from physics.hydraulics.network import G
from plants.ledger import HorizonLedger
from scripts import extended_natural_initialization as charging


def driver_identity():
    return dict(
        plant=identity(),
        files={
            Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [__file__, charging.__file__]
        },
    )


class NaturalSession(Session):
    def __init__(
        self, *, height_m=10.0, maximum_internal_step_s=2.0, initialization_source_sha256=None
    ):
        source = digest(driver_identity())
        if initialization_source_sha256 is not None and initialization_source_sha256 != source:
            raise ValueError("Natural initialization source changed")
        if height_m != 10.0 or not 0.01 <= maximum_internal_step_s <= 2.0:
            raise ValueError("Natural configuration only height10m, cap.01..2s")
        self.config = dict(
            height_m=height_m,
            maximum_internal_step_s=maximum_internal_step_s,
            initialization_source_sha256=source,
        )
        self.plant = SourceBoundPWR(height_m)
        self.plant.maximum_internal_step_s = maximum_internal_step_s
        p = self.plant
        original_drive = p.drive_torque
        reports = []
        for k in range(101):
            fraction = k / 100.0
            power = 1 - 0.95 * fraction
            p.throttle = power
            p.drive_torque = original_drive * (1 - fraction)
            r = charging.charge(p, power)
            reports.append(dict(fraction=fraction, power=power, scaled_defect=r["scaled_defect"]))
        # Exact prescribed initial rest condition, not an integration projection.
        # The nonlinear root is already zero to roundoff; dimensional balances
        # are checked again with the prescribed zero angular velocity.
        if abs(p.omega) > 1e-8:
            raise ValueError("Charging shaft root not zero")
        state = p.pack()
        state[11] = 0.0
        p.motor_on = False
        p.drive_torque = original_drive
        p.unpack(state)
        p.refresh_algebraic(state)
        dot = p.derivatives(0.0, state)
        if (
            abs(dot[11]) > 1e-12
            or max(abs(dot[14 + 3 * i]) for i in range(len(p.network.volumes))) > 1.0
        ):
            raise ValueError("Natural initial stationarity failed")
        self.initialization = dict(
            final=r,
            homotopy=reports,
            warmup_steps=0,
            source=driver_identity(),
            primary_fill_mass_kg=sum(
                v.mass_kg for n, v in p.network.volumes.items() if n != "sg_secondary"
            ),
            exact_initial_shaft_speed_rad_s=0.0,
        )
        self.ledger = HorizonLedger()
        self.ledger.attach(p)
        self.history = []
        self.revision = 0
        self.lock = RLock()


def heads(plant):
    hydro = 0.0
    loss = 0.0
    for name in ["pump", "core_out", "to_sg", "to_cold"]:
        link = plant.network.junctions[name]
        up = plant.network.volumes[link.upstream]
        down = plant.network.volumes[link.downstream]
        hydro -= (
            0.5
            * (up.thermo.rho_kg_m3 + down.thermo.rho_kg_m3)
            * G
            * (down.elevation_m - up.elevation_m)
        )
        donor = up if link.mass_flow_kg_s >= 0 else down
        loss += link.friction_dp(donor.thermo.rho_kg_m3)
    return dict(
        buoyancy_head_pa=hydro,
        friction_loss_pa=loss,
        momentum_head_defect_pa=hydro - loss,
        pump_head_pa=plant.network.junctions["pump"].pump_pressure_pa,
    )


def run(output, *, method="Radau", perturb=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = driver_identity()
    duration = 120.0 if perturb else 1800.0
    declared = dict(
        source=source,
        duration_s=duration,
        method=method,
        criteria=dict(
            energy_relative=1e-6,
            mass_relative=1e-10,
            boron_relative=1e-10,
            positive_flow=True,
            exact_zero_pump_head=True,
        ),
        initial_fill="core pressure16MPa determines charged mass; no claim of transient from nominal fill",
        perturbation=perturb,
        validated=False,
    )
    publish(output / "protocol.json", declared)
    session = NaturalSession()
    p = session.plant
    minimum_flow = float("inf")
    failure = None
    start = time.monotonic()
    initial_heads = heads(p)
    try:
        with (output / "trace.jsonl").open("x") as stream:
            while p.time_s < duration:
                commands = {}
                if perturb and p.time_s == 10.0:
                    commands = {"throttle": 0.049}
                if perturb and p.time_s == 30.0:
                    commands = {"throttle": 0.05}
                session.advance(min(5.0, duration - p.time_s), commands=commands, method=method)
                flow = p.network.junctions["pump"].mass_flow_kg_s
                minimum_flow = min(minimum_flow, flow)
                h = heads(p)
                if p.omega != 0.0 or h["pump_head_pa"] != 0.0 or p.motor_on:
                    raise ValueError("Natural trajectory has nonzero shaft speed/head or motor on")
                stream.write(
                    json.dumps(
                        dict(
                            time_s=p.time_s,
                            flow_kg_s=flow,
                            relative_fission=float(p.kinetics_state[0]),
                            heads=h,
                        )
                    )
                    + "\n"
                )
                stream.flush()
                if int(p.time_s) % 100 == 0:
                    print("natural", method, p.time_s, flow, flush=True)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), time_s=p.time_s)
    ledger = session.ledger.report() if session.revision else None
    accepted = bool(
        failure is None
        and p.time_s == duration
        and minimum_flow > 0
        and initial_heads["buoyancy_head_pa"] > 0
        and abs(initial_heads["momentum_head_defect_pa"]) < 0.01
        and ledger
        and all(
            ledger[k] <= declared["criteria"][k]
            for k in ["energy_relative", "mass_relative", "boron_relative"]
        )
    )
    if source != driver_identity():
        raise ValueError("Natural driver changed during execution")
    publish(output / "checkpoint-envelope.json", session.checkpoint())
    result = dict(
        source=source,
        completed_horizon_s=p.time_s,
        requested_horizon_s=duration,
        full_horizon_accepted=accepted,
        failure=failure,
        ledger=ledger,
        minimum_flow_kg_s=minimum_flow if session.revision else None,
        initial_heads=initial_heads,
        final_heads=heads(p),
        initialization=session.initialization,
        wall_seconds=time.monotonic() - start,
        validated=False,
        plant_referenced=False,
    )
    publish(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--method", choices=["Radau", "BDF"], default="Radau")
    p.add_argument("--perturb", action="store_true")
    p.add_argument("--replay", type=Path)
    a = p.parse_args()
    if a.replay:
        saved = json.loads(a.replay.read_text())
        if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
            raise ValueError("Invalid natural checkpoint file")
        restored = NaturalSession.restore(saved["payload"])
        publish(
            a.output, dict(replay_equal=True, revision=restored.revision, source=driver_identity())
        )
        return 0
    result = run(a.output, method=a.method, perturb=a.perturb)
    print("Natural full horizon accepted:", result["full_horizon_accepted"], flush=True)
    return 0 if result["full_horizon_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
