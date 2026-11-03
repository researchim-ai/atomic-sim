"""Independent event-located IVPs using mixed hydraulic roots and SciPy events."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.nonlinear_graph_references import IndependentSystem, bridge_spec, pipe_spec
from scripts.nonlinear_graph_references import cases as nonlinear_cases
from scripts.scheduled_graph_references import apply_changes


def guard(name, node, threshold, direction, **changes):
    return dict(
        id=name, node=node, threshold_k=float(threshold), direction=direction, changes=changes
    )


def wrap(model, guards):
    return dict(
        format="atomic-state-event-graph", schema_version=1, model_input=model, guards=guards
    )


def idle(power=12500.0):
    model = pipe_spec()
    model["steady_input"]["hydraulic"]["nodes"][1]["pressure_pa"] = 200000.0
    model["steady_input"]["thermal"]["nodes"]["a"]["heat_power_w"] = power
    return model


def cases():
    yield (
        "idle_rising",
        wrap(
            idle(), [guard("rise", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))]
        ),
    )
    yield (
        "idle_falling",
        wrap(
            idle(-12500.0),
            [guard("fall", "a", 309.875, -1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))],
        ),
    )
    yield (
        "heat_then_cool",
        wrap(
            idle(),
            [
                guard("hot", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=-25000.0))),
                guard("cool", "a", 310.0625, -1, thermal_nodes=dict(a=dict(heat_power_w=0.0))),
            ],
        ),
    )
    yield (
        "ignored_direction",
        wrap(
            idle(),
            [guard("fall_only", "a", 310.125, -1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))],
        ),
    )
    model = pipe_spec()
    model["steady_input"]["thermal"]["nodes"]["a"]["heat_power_w"] = 100000.0
    yield (
        "pressure_reversal",
        wrap(
            model,
            [
                guard(
                    "reverse",
                    "a",
                    311.0,
                    1,
                    thermal_nodes=dict(a=dict(heat_power_w=0.0)),
                    hydraulic_nodes=dict(a=dict(pressure_pa=198000.0)),
                )
            ],
        ),
    )
    model = bridge_spec()
    yield (
        "injection_batch",
        wrap(
            model,
            [
                guard(
                    "feed",
                    "a",
                    301.0,
                    1,
                    thermal_nodes=dict(a=dict(heat_power_w=0.0, supply_temperature_k=350.0)),
                    hydraulic_nodes=dict(a=dict(injection_kg_s=0.25)),
                )
            ],
        ),
    )
    model = deepcopy(dict(nonlinear_cases())["8_mixed_negative"])
    model["steady_input"]["thermal"]["nodes"]["node0"]["heat_power_w"] = -5000000.0
    yield (
        "eight_node_cooling",
        wrap(
            model,
            [guard("cool", "node0", 352.0, -1, thermal_nodes=dict(node0=dict(heat_power_w=0.0)))],
        ),
    )
    model = idle()
    model["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] = 25000.0
    yield (
        "ordered_guards",
        wrap(
            model,
            [
                guard("later", "b", 320.5, 1, thermal_nodes=dict(b=dict(heat_power_w=0.0))),
                guard("earlier", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0))),
            ],
        ),
    )
    yield (
        "no_crossing",
        wrap(
            idle(0.0), [guard("never", "a", 311.0, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0)))]
        ),
    )


def negative_cases():
    model = idle()
    model["initial_temperatures_k"]["b"] = 310.0
    model["steady_input"]["thermal"]["nodes"]["b"]["heat_power_w"] = 12500.0
    yield (
        "coincident_roots",
        wrap(
            model,
            [
                guard("left", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=0.0))),
                guard("right", "b", 310.125, 1, thermal_nodes=dict(b=dict(heat_power_w=0.0))),
            ],
        ),
        "ambiguous_roots",
    )
    yield (
        "invalid_command",
        wrap(
            idle(), [guard("bad", "a", 310.125, 1, hydraulic_nodes=dict(a=dict(pressure_pa=1e9)))]
        ),
        None,
    )
    yield (
        "post_event_domain_exit",
        wrap(
            idle(), [guard("bad", "a", 310.125, 1, thermal_nodes=dict(a=dict(heat_power_w=1e12)))]
        ),
        None,
    )


def reference(spec, method, times):
    active = deepcopy(spec["model_input"])
    system = IndependentSystem(active)
    state = np.zeros(len(system.names) + 1)
    current = 0.0
    fired = []
    observations = []
    names = list(system.names)
    indices = {name: i for i, name in enumerate(names)}

    def observe():
        temperature = system.initial + state[:-1]
        _, hyd = system.hydraulic(temperature)
        return dict(
            time_s=current,
            temperatures_k=dict(zip(names, map(float, temperature))),
            pressures_pa=hyd["pressures_pa"],
            flows_kg_s=hyd["flows_kg_s"],
            integrated_external_energy_j=float(state[-1] * system.scale),
            energy_residual_j=float(system.capacity @ state[:-1] - state[-1] * system.scale),
        )

    observations.append(observe())
    for end in times[1:]:
        while current < end:
            consumed = {r["guard_id"] for r in fired}
            pending = [g for g in spec["guards"] if g["id"] not in consumed]
            functions = []
            for g in pending:
                index = indices[g["node"]]
                offset = g["threshold_k"] - system.initial[index]

                def event(time, value, i=index, target=offset):
                    return value[i] - target

                event.direction = g["direction"]
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
                raise ValueError("Independent state-event IVP failed")
            previous = current
            current = float(solution.t[-1])
            state = solution.y[:, -1]
            roots = [(float(ts[0]), i) for i, ts in enumerate(solution.t_events or []) if len(ts)]
            if roots:
                root, index = min(roots)
                g = pending[index]
                if root <= previous:
                    raise ValueError("Independent zero-time root")
                fired.append(dict(guard_id=g["id"], time_s=root))
                active = apply_changes(active, g["changes"])
                system = IndependentSystem(active)
            elif current != end:
                raise ValueError("Independent event IVP did not reach observation")
        observations.append(observe())
    return dict(states=observations, fired=fired, method=method)
