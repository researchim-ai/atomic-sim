"""Three output clocks, priority commands and phase-specific full replay."""

import argparse
import hashlib
import json
from pathlib import Path

from atomic_model_plant_extension.qualification import publish
from atomic_model_plant_extension.session import Session, identity


def qualify():
    source = dict(
        plant=identity(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    runs = []
    for dt in [1.0, 0.5, 0.25]:
        s = Session(maximum_internal_step_s=0.5)
        events = []
        restarts = []
        points = [9.0, 10.5, 14.0, 20.5]
        while s.plant.time_s < 20.5:
            now = s.plant.time_s
            target = min([x for x in points + [10.0, 20.5] if x > now])
            commands = (
                dict(scram=True, motor_off=True, throttle=1.0, rod_command=-0.1)
                if now == 10.0
                else {}
            )
            s.advance(min(dt, target - now), commands=commands, rtol=1e-7, atol=1e-6)
            if s.plant.time_s > 10.0:
                if (
                    s.plant.effective_steam_throttle != 0.0
                    or not s.plant.scram_latched
                    or s.plant.motor_on
                ):
                    raise ValueError("Simultaneous-command protection priority failed")
            events.extend(s.plant.last_motion_events)
            if s.plant.time_s in points:
                saved = json.loads(json.dumps(s.checkpoint()))
                restored = Session.restore(saved)
                if restored.checkpoint() != saved:
                    raise ValueError("Phase checkpoint full replay differs")
                restarts.append(dict(time_s=s.plant.time_s, replay_equal=True))
        first = {}
        for e in events:
            first.setdefault(e["kind"], e["t"])
        if not {"feed_closed", "dump_source_opened", "rod_stop"} <= set(first):
            raise ValueError("Missing physical switch events")
        runs.append(
            dict(
                output_step_s=dt,
                first_events=first,
                events=events,
                restarts=restarts,
                final_relative_fission=float(s.plant.kinetics_state[0]),
            )
        )
        print("events output clock", dt, "passed", flush=True)
    differences = {}
    for name, tolerance in [
        ("rod_stop", 1e-10),
        ("feed_closed", 1e-3),
        ("dump_source_opened", 1e-3),
    ]:
        times = [r["first_events"][name] for r in runs]
        spread = max(times) - min(times)
        if spread > tolerance:
            raise ValueError(f"Event {name} timing spread {spread}>{tolerance}")
        differences[name] = dict(spread_s=spread, tolerance_s=tolerance)
    if source["plant"] != identity():
        raise ValueError("Event campaign plant source changed")
    return dict(
        source=source,
        runs=runs,
        first_event_spreads=differences,
        full_replays=12,
        scope="Three output clocks, SCRAM priority, simultaneous requests and restart in four transition phases",
        validated=False,
        plant_referenced=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    publish(a.output, qualify())


if __name__ == "__main__":
    main()
