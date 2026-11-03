"""Separate mixed-hydraulic IVPs with a small explicit supervisory dispatcher."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.hysteresis_references import cases as mode_cases
from scripts.hysteresis_references import commands, relay
from scripts.hysteresis_references import request as mode_request
from scripts.nonlinear_graph_references import IndependentSystem
from scripts.scheduled_graph_references import apply_changes
from scripts.state_event_references import idle


def calendar(time, target, name):
    return dict(time_s=float(time), to=target, id=name)


def wrap(controller, schedule=(), online=()):
    return dict(
        format="atomic-supervised-mode-graph",
        schema_version=1,
        controller=deepcopy(controller),
        schedule=list(schedule),
        online_modes=list(online),
    )


def hold(controller, pressure=None):
    controller = deepcopy(controller)
    controller["modes"].append(
        dict(id="hold", commands=commands(0, pressure=pressure), transitions=[])
    )
    return controller


def cases():
    modes = dict(mode_cases())
    yield (
        "calendar_pause",
        wrap(hold(relay()), [calendar(20, "hold", "pause"), calendar(30, "cool", "resume")]),
        [],
    )
    yield (
        "calendar_noop",
        wrap(relay(), [calendar(3, "heat", "ack1"), calendar(10, "heat", "ack2")]),
        [],
    )
    yield "calendar_pressure", wrap(modes["pressure_relay"], [calendar(2, "cool", "override")]), []
    yield (
        "bridge_pause",
        wrap(
            hold(modes["bridge_relay"]),
            [calendar(0.5, "hold", "pause"), calendar(11, "heat", "resume")],
        ),
        [],
    )
    yield "eight_override", wrap(modes["eight_node_modes"], [calendar(1, "hold", "override")]), []
    yield (
        "online_interleaved",
        wrap(
            hold(relay()),
            [calendar(50, "hold", "pause"), calendar(70, "heat", "resume")],
            ["heat", "cool", "hold"],
        ),
        [(2, "hold", 10), (12, "heat", 10), (22, "heat", 5)],
    )
    yield (
        "online_pressure",
        wrap(hold(modes["pressure_relay"], 198000), online=["heat", "cool", "hold"]),
        [(2, "cool", 3), (5, "hold", 5), (10, "cool", 10)],
    )
    terminal = mode_request(
        idle(0),
        [
            dict(id=name, commands=commands(power), transitions=[])
            for name, power in [("warm", 1000), ("cool", -1000), ("hold", 0)]
        ],
        initial="warm",
        residence=1,
    )
    yield (
        "external_terminals",
        wrap(
            terminal,
            [
                calendar(10, "cool", "cool"),
                calendar(30, "hold", "pause"),
                calendar(50, "warm", "warm"),
                calendar(90, "hold", "stop"),
            ],
        ),
        [],
    )
    yield "autonomous_only", wrap(relay()), []


def reference(spec, horizon, online_events, method):
    """Independent physics and event dispatch; online decisions are supplied inputs."""
    controller = spec["controller"]
    modes = {mode["id"]: mode for mode in controller["modes"]}
    base = controller["model_input"]
    mode = controller["initial_mode"]
    system = IndependentSystem(apply_changes(base, modes[mode]["commands"]))
    state = np.zeros(len(system.names) + 1)
    indices = {name: i for i, name in enumerate(system.names)}
    forced = sorted(
        [(e["time_s"], "calendar", e["to"], e["id"]) for e in spec["schedule"]]
        + [(e["time_s"], "online", e["target_mode"], e["command_id"]) for e in online_events]
    )
    cursor, current, entry = 0, 0.0, 0.0
    history = []
    while current < horizon:
        bound = min(horizon, forced[cursor][0]) if cursor < len(forced) else horizon
        outgoing = modes[mode]["transitions"]
        functions = []
        for guard in outgoing:
            i = indices[guard["node"]]
            offset = guard["threshold_k"] - system.initial[i]

            def event(t, y, i=i, offset=offset):
                return y[i] - offset

            event.direction, event.terminal = guard["direction"], True
            functions.append(event)
        sol = solve_ivp(
            system.rhs,
            (current, bound),
            state,
            method=method,
            rtol=1e-11,
            atol=1e-12,
            max_step=0.5,
            events=functions or None,
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            raise ValueError("Independent supervised IVP failed")
        current, state = float(sol.t[-1]), sol.y[:, -1]
        roots = [(float(ts[0]), i) for i, ts in enumerate(sol.t_events or []) if len(ts)]
        if roots:
            _, i = min(roots)
            guard = outgoing[i]
            kind, target, eid = "state", guard["to"], guard["id"]
        elif cursor < len(forced) and current == forced[cursor][0]:
            _, kind, target, eid = forced[cursor]
            cursor += 1
        else:
            continue
        changed = target != mode
        if changed and current - entry < controller["minimum_residence_s"] - 1e-8:
            raise ValueError("Independent residence violation")
        history.append(
            dict(
                kind=kind,
                time_s=current,
                source_mode=mode,
                target_mode=target,
                id=eid,
                changed=changed,
            )
        )
        if changed:
            mode, entry = target, current
            system = IndependentSystem(apply_changes(base, modes[mode]["commands"]))
    temperature = system.initial + state[:-1]
    _, hyd = system.hydraulic(temperature)
    return dict(
        history=history,
        mode=mode,
        temperatures_k=dict(zip(system.names, map(float, temperature))),
        pressures_pa=hyd["pressures_pa"],
        flows_kg_s=hyd["flows_kg_s"],
        integrated_external_energy_j=float(state[-1] * system.scale),
    )
