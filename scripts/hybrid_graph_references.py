"""Independent segmented SciPy IVPs and scenario definitions for hybrid controls.

No production hybrid dispatch, event locator, residual or graph projection.
Hydraulics and energy equations use the separate established reference family.
"""

import numpy as np
from scipy.integrate import solve_ivp

from scripts.hysteresis_references import commands, relay
from scripts.nonlinear_graph_references import IndependentSystem, bridge_spec, pipe_spec
from scripts.scheduled_graph_references import apply_changes


def hybrid(model=None):
    value = relay(model=model, residence=0.1)
    value["format"] = "atomic-hybrid-thermal-graph"
    value.pop("minimum_residence_s")
    value["calendar"] = []
    for mode in value["modes"]:
        for guard in mode["transitions"]:
            guard.update(priority=10, arm_delay_s=0.0)
    return value


def hold(spec):
    spec["modes"].append(dict(id="hold", commands=commands(0.0), transitions=[]))
    return spec


def calendar(id, when, mode, priority=20):
    return dict(id=id, time_s=float(when), to=mode, priority=priority)


def calls():
    return [dict(duration_s=v, command=None) for v in [2.0, 7.0, 8.0, 13.0, 10.0]]


def cases():
    yield "relay", hybrid(), calls()
    r = hold(hybrid())
    r["calendar"] = [
        calendar("force_cool", 3.0, "cool"),
        calendar("release", 12.0, "heat"),
        calendar("stop", 31.0, "hold"),
    ]
    yield "calendar_relay", r, calls()
    r = hold(hybrid())
    r["calendar"] = [
        calendar("low_priority", 12.0, "heat", 1),
        calendar("high_priority", 12.0, "hold", 40),
    ]
    yield "calendar_priority", r, calls()
    r = hybrid()
    r["modes"][0]["transitions"][0]["arm_delay_s"] = 10.0
    yield "delayed_arming", r, calls()
    r = hold(hybrid())
    other = dict(r["modes"][0]["transitions"][0], id="low_priority_guard", to="hold", priority=1)
    r["modes"][0]["transitions"].append(other)
    yield "same_predicate_priority", r, calls()
    r = hold(hybrid())
    c = calls()
    c[1]["command"] = dict(
        mode="hold", expected_revision=1, controller_state={"integral": 0.5, "tick": 1}, priority=30
    )
    r["calendar"] = [calendar("restart", 17.5, "heat")]
    yield "explicit_online_state", r, c
    r = hybrid(pipe_spec())
    r["modes"][0]["commands"] = commands(100000.0, pressure=200000.0)
    r["modes"][1]["commands"] = commands(-100000.0, pressure=198000.0)
    r["modes"][0]["transitions"][0]["threshold_k"] = 311.0
    r["modes"][1]["transitions"][0]["threshold_k"] = 310.25
    r["calendar"] = [
        calendar("reverse_at_13", 13.0, "cool"),
        calendar("restore_at_23", 23.0, "heat"),
    ]
    yield "flow_reversal", r, calls()
    r = hybrid(bridge_spec())
    r["modes"][0]["commands"] = commands(600000.0)
    r["modes"][1]["commands"] = commands(-600000.0)
    # Branching test isolates scheduled reinitialization; no monotonicity assumed.
    r["modes"][0]["transitions"] = []
    r["modes"][1]["transitions"] = []
    r["calendar"] = [calendar("branch_cool", 7.0, "cool"), calendar("branch_heat", 21.0, "heat")]
    yield "branched_schedule", r, calls()


def reference(spec, program, method):
    modes = {m["id"]: m for m in spec["modes"]}
    origin = IndependentSystem(spec["model_input"])
    y = np.zeros(len(origin.names) + 1)
    mode, entry, now, cursor = spec["initial_mode"], 0.0, 0.0, 0
    timeline = sorted(spec["calendar"], key=lambda e: (e["time_s"], e["id"]))
    states, history = [], []

    def system():
        return IndependentSystem(apply_changes(spec["model_input"], modes[mode]["commands"]))

    def event(contenders, kind):
        nonlocal mode, entry
        contenders = sorted(contenders, key=lambda c: (-c["priority"], c["id"]))
        if len(contenders) > 1 and contenders[0]["priority"] == contenders[1]["priority"]:
            raise ValueError("Independent priority tie")
        winner = contenders[0]
        history.append(
            dict(
                time_s=now,
                winner=winner["id"],
                source_mode=mode,
                target_mode=winner["to"],
                kind=kind,
            )
        )
        mode, entry = winner["to"], now
        if len(history) > 4096:
            raise ValueError("Independent transition budget")

    for revision, call in enumerate(program):
        target = now + call["duration_s"]
        online = None
        if call["command"] is not None:
            online = dict(call["command"], id="online_" + str(revision), to=call["command"]["mode"])
        while True:
            due = []
            while cursor < len(timeline) and timeline[cursor]["time_s"] == now:
                due.append(timeline[cursor])
                cursor += 1
            active = [g for g in modes[mode]["transitions"] if entry + g["arm_delay_s"] <= now]
            due += [
                g
                for g in active
                if g["direction"]
                * (
                    origin.initial[origin.index[g["node"]]]
                    + y[origin.index[g["node"]]]
                    - g["threshold_k"]
                )
                >= 0
            ]
            if online:
                due.append(online)
                online = None
            if due:
                event(due, "boundary")
                continue
            if now >= target:
                break
            bound = min(
                [target]
                + ([timeline[cursor]["time_s"]] if cursor < len(timeline) else [])
                + [
                    entry + g["arm_delay_s"]
                    for g in modes[mode]["transitions"]
                    if entry + g["arm_delay_s"] > now
                ]
            )
            events = []
            for guard in active:

                def root(t, state, g=guard):
                    i = origin.index[g["node"]]
                    return origin.initial[i] + state[i] - g["threshold_k"]

                root.terminal = True
                root.direction = guard["direction"]
                events.append(root)
            solved = solve_ivp(
                system().rhs,
                (now, bound),
                y,
                method=method,
                rtol=1e-11,
                atol=1e-12,
                max_step=0.25,
                events=events or None,
            )
            if not solved.success:
                raise ValueError("Independent IVP failed")
            y, now = solved.y[:, -1], float(solved.t[-1])
            if solved.status == 1:
                i = next(i for i, v in enumerate(solved.t_events) if len(v))
                g = active[i]
                simultaneous = [
                    other
                    for other in active
                    if (other["node"], other["direction"], other["threshold_k"])
                    == (g["node"], g["direction"], g["threshold_k"])
                ]
                event(simultaneous, "root")
        ref = system()
        hyd = ref.hydraulic(origin.initial + y[:-1])[1]
        states.append(
            dict(
                time_s=now,
                mode=mode,
                temperatures_k=dict(zip(origin.names, map(float, origin.initial + y[:-1]))),
                pressures_pa=hyd["pressures_pa"],
                flows_kg_s=hyd["flows_kg_s"],
                integrated_external_energy_j=float(y[-1] * origin.scale),
            )
        )
    return dict(states=states, history=history)
