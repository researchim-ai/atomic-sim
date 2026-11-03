"""Independent segmented IVPs with explicitly rebuilt command states."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.nonlinear_graph_references import (
    IndependentSystem,
    bridge_spec,
    pipe_spec,
)
from scripts.nonlinear_graph_references import cases as nonlinear_cases


def event(time, **changes):
    return dict(time_s=float(time), changes=changes)


def wrap(model, timeline):
    return dict(
        format="atomic-scheduled-thermal-graph",
        schema_version=1,
        model_input=model,
        timeline=timeline,
    )


def heater_spec():
    model = pipe_spec()
    model["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    return wrap(
        model,
        [
            event(0, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
            event(
                10, thermal_nodes=dict(a=dict(heat_power_w=2000.0), b=dict(heat_power_w=-1000.0))
            ),
            event(30, thermal_nodes=dict(a=dict(heat_power_w=0.0), b=dict(heat_power_w=500.0))),
            event(60, thermal_nodes=dict(a=dict(heat_power_w=0.0), b=dict(heat_power_w=0.0))),
            event(120, thermal_nodes=dict(b=dict(heat_power_w=1000.0))),
        ],
    )


def cases():
    yield "heater_pulse", heater_spec()
    yield (
        "pressure_reversal",
        wrap(
            pipe_spec(),
            [
                event(17.25, hydraulic_nodes=dict(a=dict(pressure_pa=198000.0))),
                event(43.75, hydraulic_nodes=dict(a=dict(pressure_pa=201000.0))),
                event(90, hydraulic_edges=dict(pipe=dict(pressure_gain_pa=-500.0))),
                event(120, hydraulic_nodes=dict(a=dict(pressure_pa=199000.0))),
            ],
        ),
    )
    model = bridge_spec()
    yield (
        "injection_batch",
        wrap(
            model,
            [
                event(
                    20,
                    hydraulic_nodes=dict(a=dict(injection_kg_s=0.25)),
                    thermal_nodes=dict(a=dict(supply_temperature_k=355.0)),
                ),
                event(40, hydraulic_nodes=dict(a=dict(injection_kg_s=-0.25))),
                event(70, hydraulic_nodes=dict(a=dict(injection_kg_s=0.0))),
                event(
                    90,
                    thermal_nodes=dict(a=dict(heat_power_w=200000.0, ambient_temperature_k=295.0)),
                ),
            ],
        ),
    )
    yield (
        "simultaneous_pressure_offset",
        wrap(
            bridge_spec(),
            [
                event(
                    25,
                    hydraulic_nodes=dict(
                        supply=dict(pressure_pa=201000.0), outlet=dict(pressure_pa=191000.0)
                    ),
                    thermal_nodes=dict(a=dict(heat_power_w=0.0)),
                ),
                event(60, thermal_nodes=dict(a=dict(heat_power_w=600000.0))),
            ],
        ),
    )
    _, model = next(c for c in nonlinear_cases() if c[0] == "8_mixed_negative")
    first = next(iter(model["steady_input"]["thermal"]["nodes"]))
    power = model["steady_input"]["thermal"]["nodes"][first]["heat_power_w"]
    edge = model["steady_input"]["hydraulic"]["edges"][0]
    yield (
        "endpoint_checkpoint",
        wrap(
            model,
            [
                event(30, thermal_nodes={first: dict(heat_power_w=power + 10000.0)}),
                event(37.5, thermal_nodes={first: dict(heat_power_w=power)}),
                event(
                    60,
                    hydraulic_edges={
                        edge["id"]: dict(pressure_gain_pa=edge["pressure_gain_pa"] + 100.0)
                    },
                ),
                event(120, thermal_nodes={first: dict(heat_power_w=power + 1000.0)}),
                event(150, thermal_nodes={first: dict(heat_power_w=power)}),
            ],
        ),
    )
    yield "no_commands", wrap(pipe_spec(-0.01), [])


def negative_cases():
    yield (
        "invalid_right_state",
        wrap(
            pipe_spec(),
            [
                event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
                event(5, hydraulic_nodes=dict(a=dict(pressure_pa=1e9))),
            ],
        ),
        "event_reinitialization",
    )
    yield (
        "late_domain_exit",
        wrap(
            pipe_spec(),
            [
                event(1, thermal_nodes=dict(a=dict(heat_power_w=1000.0))),
                event(5, thermal_nodes=dict(a=dict(heat_power_w=1e9))),
            ],
        ),
        "segment_failure",
    )
    model = bridge_spec()
    model["steady_input"]["thermal"]["nodes"]["a"].pop("supply_temperature_k")
    yield (
        "missing_new_inlet_temperature",
        wrap(model, [event(1, hydraulic_nodes=dict(a=dict(injection_kg_s=0.5)))]),
        "event_reinitialization",
    )


def apply_changes(model, changes):
    # Deliberately independent from production schedule parsing/patching.
    updated = deepcopy(model)
    for name, values in changes.get("thermal_nodes", {}).items():
        for field, value in values.items():
            updated["steady_input"]["thermal"]["nodes"][name][field] = value
    for name, values in changes.get("hydraulic_nodes", {}).items():
        node = next(n for n in updated["steady_input"]["hydraulic"]["nodes"] if n["id"] == name)
        for field, value in values.items():
            node[field] = value
    for name, values in changes.get("hydraulic_edges", {}).items():
        edge = next(e for e in updated["steady_input"]["hydraulic"]["edges"] if e["id"] == name)
        for field, value in values.items():
            edge[field] = value
    return updated


def reference(spec, method, times=(0.0, 10.0, 30.0, 60.0, 90.0, 120.0)):
    active = deepcopy(spec["model_input"])
    timeline = spec["timeline"]
    names = list(active["initial_temperatures_k"])
    state = np.zeros(len(names) + 1)
    cursor = 0
    observations = []
    switches = []

    def observe(system, time):
        temperature = system.initial + state[:-1]
        _, hyd = system.hydraulic(temperature)
        return dict(
            time_s=time,
            temperatures_k=dict(zip(system.names, map(float, temperature))),
            pressures_pa=hyd["pressures_pa"],
            flows_kg_s=hyd["flows_kg_s"],
            integrated_external_energy_j=float(state[-1] * system.scale),
            energy_residual_j=float(system.capacity @ state[:-1] - state[-1] * system.scale),
            active_event_count=cursor,
        )

    system = IndependentSystem(active)
    while cursor < len(timeline) and timeline[cursor]["time_s"] == 0:
        left = observe(system, 0.0)
        active = apply_changes(active, timeline[cursor]["changes"])
        system = IndependentSystem(active)
        cursor += 1
        switches.append(dict(time_s=0.0, left=left, right=observe(system, 0.0)))
    observations.append(observe(system, 0.0))
    boundaries = sorted(
        set(times) | {float(e["time_s"]) for e in timeline if 0 < e["time_s"] <= times[-1]}
    )
    start = 0.0
    for end in boundaries[1:]:
        solution = solve_ivp(
            system.rhs,
            (start, end),
            state,
            method=method,
            rtol=1e-11,
            atol=1e-12,
            max_step=0.5,
            t_eval=[end],
        )
        if not solution.success or solution.t[-1] != end or not np.all(np.isfinite(solution.y)):
            raise ValueError("Independent scheduled IVP failed")
        state = solution.y[:, -1]
        if cursor < len(timeline) and timeline[cursor]["time_s"] == end:
            left = observe(system, end)
            active = apply_changes(active, timeline[cursor]["changes"])
            system = IndependentSystem(active)
            cursor += 1
            switches.append(dict(time_s=end, left=left, right=observe(system, end)))
        if end in times:
            observations.append(observe(system, end))
        start = end
    return dict(states=observations, switches=switches, active_event_count=cursor)
