"""Independent switched IVPs for finite-mode thermal control, without production FSM."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.nonlinear_graph_references import IndependentSystem, bridge_spec, pipe_spec
from scripts.nonlinear_graph_references import cases as nonlinear_cases
from scripts.scheduled_graph_references import apply_changes
from scripts.state_event_references import idle


def transition(name, node, threshold, direction, target):
    return dict(id=name, node=node, threshold_k=float(threshold), direction=direction, to=target)


def commands(power, node="a", pressure=None):
    result = dict(thermal_nodes={node: dict(heat_power_w=float(power))})
    if pressure is not None:
        result["hydraulic_nodes"] = dict(a=dict(pressure_pa=float(pressure)))
    return result


def request(model, modes, initial="heat", residence=0.1):
    return dict(
        format="atomic-hysteretic-graph",
        schema_version=1,
        model_input=model,
        initial_mode=initial,
        minimum_residence_s=float(residence),
        modes=modes,
    )


def relay(
    model=None,
    high=310.1375,
    low=310.0125,
    heat=12500.0,
    cool=-25000.0,
    pressure=False,
    residence=1.0,
):
    return request(
        idle() if model is None else model,
        [
            dict(
                id="heat",
                commands=commands(heat, pressure=200000.0 if pressure else None),
                transitions=[transition("hot", "a", high, 1, "cool")],
            ),
            dict(
                id="cool",
                commands=commands(cool, pressure=198000.0 if pressure else None),
                transitions=[transition("cold", "a", low, -1, "heat")],
            ),
        ],
        residence=residence,
    )


def cases():
    yield "idle_relay", relay()
    yield "asymmetric_relay", relay(high=310.21, low=310.05, heat=20000.0, cool=-10000.0)
    yield (
        "flowing_relay",
        relay(pipe_spec(), high=311.0, low=310.25, heat=100000.0, cool=-100000.0, residence=0.1),
    )
    yield (
        "pressure_relay",
        relay(
            pipe_spec(),
            high=311.0,
            low=310.25,
            heat=100000.0,
            cool=-100000.0,
            pressure=True,
            residence=0.1,
        ),
    )
    yield (
        "bridge_relay",
        relay(bridge_spec(), high=302.0, low=300.5, heat=600000.0, cool=-600000.0, residence=0.1),
    )
    model = deepcopy(dict(nonlinear_cases())["8_mixed_negative"])
    yield (
        "eight_node_modes",
        request(
            model,
            [
                dict(
                    id="cool",
                    commands=commands(-5000000.0, "node0"),
                    transitions=[transition("low", "node0", 352.0, -1, "warm")],
                ),
                dict(
                    id="warm",
                    commands=commands(20000000.0, "node0"),
                    transitions=[transition("high", "node0", 353.0, 1, "hold")],
                ),
                dict(id="hold", commands=commands(0.0, "node0"), transitions=[]),
            ],
            initial="cool",
        ),
    )
    yield (
        "terminal_mode",
        request(
            idle(),
            [
                dict(
                    id="heat",
                    commands=commands(12500.0),
                    transitions=[transition("stop", "a", 310.1375, 1, "hold")],
                ),
                dict(id="hold", commands=commands(0.0), transitions=[]),
            ],
        ),
    )
    yield (
        "standby",
        request(
            idle(0.0),
            [dict(id="standby", commands=commands(0.0), transitions=[])],
            initial="standby",
        ),
    )
    model = idle()
    model["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] = 20000.0
    yield (
        "branch_selection",
        request(
            model,
            [
                dict(
                    id="ramp",
                    commands=dict(
                        thermal_nodes=dict(
                            a=dict(heat_power_w=12500.0), b=dict(heat_power_w=20000.0)
                        )
                    ),
                    transitions=[
                        transition("a_first", "a", 310.1375, 1, "finish_a"),
                        transition("b_first", "b", 320.12, 1, "finish_b"),
                    ],
                ),
                dict(
                    id="finish_a",
                    commands=dict(
                        thermal_nodes=dict(a=dict(heat_power_w=0.0), b=dict(heat_power_w=0.0))
                    ),
                    transitions=[],
                ),
                dict(
                    id="finish_b",
                    commands=dict(
                        thermal_nodes=dict(a=dict(heat_power_w=-1000.0), b=dict(heat_power_w=0.0))
                    ),
                    transitions=[],
                ),
            ],
            initial="ramp",
        ),
    )


def negative_cases():
    yield "too_short_residence", relay(residence=10.0), "minimum_residence"
    bad = relay(pressure=True)
    bad["modes"][1]["commands"]["hydraulic_nodes"]["a"]["pressure_pa"] = 1e9
    yield "invalid_destination", bad, None
    model = idle()
    model["initial_temperatures_k"]["b"] = 310.0
    model["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] = 12500.0
    ambiguous = request(
        model,
        [
            dict(
                id="ramp",
                commands=dict(
                    thermal_nodes=dict(a=dict(heat_power_w=12500.0), b=dict(heat_power_w=12500.0))
                ),
                transitions=[
                    transition("left", "a", 310.1375, 1, "hold"),
                    transition("right", "b", 310.1375, 1, "hold"),
                ],
            ),
            dict(
                id="hold",
                commands=dict(
                    thermal_nodes=dict(a=dict(heat_power_w=0.0), b=dict(heat_power_w=0.0))
                ),
                transitions=[],
            ),
        ],
        initial="ramp",
    )
    yield "ambiguous_branches", ambiguous, "ambiguous_roots"
    zero_gap = relay()
    zero_gap["modes"][1]["transitions"][0]["threshold_k"] = 310.1375
    yield "zero_gap_entry", zero_gap, "entry_guard"


def reference(spec, method, times):
    base = deepcopy(spec["model_input"])
    modes = {m["id"]: m for m in spec["modes"]}
    mode = spec["initial_mode"]
    active = apply_changes(base, modes[mode]["commands"])
    system = IndependentSystem(active)
    state = np.zeros(len(system.names) + 1)
    names = list(system.names)
    indices = {name: i for i, name in enumerate(names)}
    current = 0.0
    entry = 0.0
    history = []
    observations = []

    def observe():
        temperature = system.initial + state[:-1]
        _, hyd = system.hydraulic(temperature)
        return dict(
            time_s=current,
            mode=mode,
            transition_count=len(history),
            temperatures_k=dict(zip(names, map(float, temperature))),
            pressures_pa=hyd["pressures_pa"],
            flows_kg_s=hyd["flows_kg_s"],
            integrated_external_energy_j=float(state[-1] * system.scale),
            energy_residual_j=float(system.capacity @ state[:-1] - state[-1] * system.scale),
        )

    observations.append(observe())
    for end in times[1:]:
        while current < end:
            outgoing = modes[mode]["transitions"]
            functions = []
            for guard in outgoing:
                index = indices[guard["node"]]
                offset = guard["threshold_k"] - system.initial[index]

                def event(time, value, i=index, target=offset):
                    return value[i] - target

                event.direction = guard["direction"]
                event.terminal = True
                functions.append(event)
            solution = solve_ivp(
                system.rhs,
                (current, end),
                state,
                method=method,
                rtol=1e-11,
                atol=1e-12,
                max_step=0.5,
                events=functions if functions else None,
            )
            if not solution.success or not np.all(np.isfinite(solution.y)):
                raise ValueError("Independent hysteretic IVP failed")
            previous = current
            current = float(solution.t[-1])
            state = solution.y[:, -1]
            roots = [(float(ts[0]), i) for i, ts in enumerate(solution.t_events or []) if len(ts)]
            if roots:
                when, index = min(roots)
                guard = outgoing[index]
                target = guard["to"]
                if (
                    when <= previous
                    or len(history) > 4096
                    or when - entry < spec["minimum_residence_s"] - 1e-8
                ):
                    raise ValueError("Independent hysteretic mode timing invalid")
                history.append(
                    dict(guard_id=guard["id"], time_s=when, source_mode=mode, target_mode=target)
                )
                mode = target
                entry = when
                active = apply_changes(base, modes[mode]["commands"])
                system = IndependentSystem(active)
            elif current != end:
                raise ValueError("Independent hysteretic IVP did not reach sample")
        observations.append(observe())
    return dict(states=observations, history=history, method=method)
