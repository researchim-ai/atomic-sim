"""Independent sampled decisions and mixed-hydraulic IVPs, without production policy."""

import random
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.hysteresis_references import cases as mode_cases
from scripts.hysteresis_references import relay
from scripts.nonlinear_graph_references import IndependentSystem
from scripts.scheduled_graph_references import apply_changes
from scripts.supervision_references import calendar, hold, wrap


def make(
    controller,
    *,
    low=310.0373,
    high=310.1037,
    noise=0.01,
    seed=42,
    period=1.0,
    sensor="a",
    schedule=(),
    initial="heat",
):
    return dict(
        format="atomic-closed-loop-radau",
        schema_version=1,
        plant=wrap(controller, schedule, ["heat", "cool", "hold"]),
        policy=dict(
            kind="sampled_schmitt_v1",
            sensor_node=sensor,
            low_k=low,
            high_k=high,
            heat_mode="heat",
            cool_mode="cool",
            initial_decision=initial,
            sample_period_s=period,
            noise_amplitude_k=noise,
            seed=seed,
        ),
    )


def terminal(controller, *, pressure=None):
    controller = hold(controller, pressure)
    for mode in controller["modes"]:
        mode["transitions"] = []
    return controller


def cases():
    modes = dict(mode_cases())
    yield "idle_clean", make(terminal(relay(residence=0.1)), noise=0)
    yield "idle_seed42", make(terminal(relay(residence=0.1)))
    yield "idle_seed43", make(terminal(relay(residence=0.1)), seed=43)
    yield (
        "calendar_pause",
        make(terminal(relay(residence=0.1)), schedule=[calendar(10.5, "hold", "pause")]),
    )
    yield "flowing", make(terminal(modes["flowing_relay"]), low=310.25, high=311)
    yield "pressure", make(terminal(modes["pressure_relay"], pressure=199000), low=310.25, high=311)
    yield "bridge", make(terminal(modes["bridge_relay"]), low=300.5, high=302)
    eight = deepcopy(modes["eight_node_modes"])
    eight["modes"][1]["id"] = "heat"
    for mode in eight["modes"]:
        mode["transitions"] = []
    yield "eight_nodes", make(eight, low=352, high=353, sensor="node0", initial="cool", period=0.25)
    safety = hold(relay(residence=0.1, high=310.14575))
    safety["modes"][1]["transitions"] = []
    yield "autonomous_safety", make(safety, low=310.03, high=310.13, noise=0.001)


def reference(request, ticks, method):
    p = request["policy"]
    spec = request["plant"]["controller"]
    modes = {m["id"]: m for m in spec["modes"]}
    base = spec["model_input"]
    mode = spec["initial_mode"]
    system = IndependentSystem(apply_changes(base, modes[mode]["commands"]))
    state = np.zeros(len(system.names) + 1)
    indices = {name: i for i, name in enumerate(system.names)}
    rng = random.Random(p["seed"])
    memory = p["initial_decision"]
    time, entry, cursor = 0.0, 0.0, 0
    schedule = request["plant"]["schedule"]
    samples = []
    events = []

    def change(target, kind, eid):
        nonlocal mode, entry, system
        changed = target != mode
        if changed and time - entry < spec["minimum_residence_s"] - 1e-8:
            raise ValueError("Independent controller residence violation")
        events.append(dict(kind=kind, time_s=time, source_mode=mode, target_mode=target, id=eid))
        if changed:
            mode, entry = target, time
            system = IndependentSystem(apply_changes(base, modes[mode]["commands"]))

    for tick in range(ticks):
        temperature = system.initial + state[:-1]
        truth = float(temperature[indices[p["sensor_node"]]])
        unit = rng.random()
        measured = truth + p["noise_amplitude_k"] * (2 * unit - 1)
        decision = (
            p["heat_mode"]
            if measured <= p["low_k"]
            else p["cool_mode"]
            if measured >= p["high_k"]
            else memory
        )
        samples.append(
            dict(
                true_temperature_k=truth,
                uniform_draw=unit,
                measured_temperature_k=measured,
                decision=decision,
            )
        )
        change(decision, "online", f"controller:tick-{tick + 1}")
        memory = decision
        end = (tick + 1) * p["sample_period_s"]
        while time < end:
            bound = min(end, schedule[cursor]["time_s"]) if cursor < len(schedule) else end
            outgoing = modes[mode]["transitions"]
            functions = []
            for guard in outgoing:
                index = indices[guard["node"]]
                offset = guard["threshold_k"] - system.initial[index]

                def event(t, y, i=index, target=offset):
                    return y[i] - target

                event.direction, event.terminal = guard["direction"], True
                functions.append(event)
            sol = solve_ivp(
                system.rhs,
                (time, bound),
                state,
                method=method,
                rtol=1e-11,
                atol=1e-12,
                max_step=0.125,
                events=functions or None,
            )
            if not sol.success or not np.all(np.isfinite(sol.y)):
                raise ValueError("Independent closed-loop IVP failed")
            time, state = float(sol.t[-1]), sol.y[:, -1]
            roots = [(float(ts[0]), i) for i, ts in enumerate(sol.t_events or []) if len(ts)]
            if roots:
                _, i = min(roots)
                guard = outgoing[i]
                change(guard["to"], "state", guard["id"])
            elif cursor < len(schedule) and time == schedule[cursor]["time_s"]:
                declared = schedule[cursor]
                cursor += 1
                change(declared["to"], "calendar", declared["id"])
    temperature = system.initial + state[:-1]
    _, hyd = system.hydraulic(temperature)
    return dict(
        samples=samples,
        events=events,
        mode=mode,
        temperatures_k=dict(zip(system.names, map(float, temperature))),
        pressures_pa=hyd["pressures_pa"],
        flows_kg_s=hyd["flows_kg_s"],
        integrated_external_energy_j=float(state[-1] * system.scale),
    )
