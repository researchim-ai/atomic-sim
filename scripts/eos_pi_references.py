"""Independent sampled control recurrence over absolute-energy reference plant."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from scripts.controlled_assembly_references import IndependentAssembly
from scripts.controlled_assembly_references import request as plant_request


def request(*, mode="pi"):
    plant = plant_request()
    policy = dict(
        kind="sampled_pi_conditional_v1",
        sensor="wall.w0",
        actuator="wall.w0",
        setpoint_k=301.0,
        sample_period_s=0.01,
        filter_tau_s=0.0,
        kp_w_k=100.0,
        ki_w_k_s=50.0,
        bias_w=0.0,
        output_min_w=-2000.0,
        output_max_w=2000.0,
        slew_w_s=1e6,
        initial_integral_w=0.0,
        initial_output_w=0.0,
    )
    if mode == "fixed":
        policy.update(kp_w_k=0.0, ki_w_k_s=0.0, bias_w=100.0)
    elif mode == "filtered":
        policy["filter_tau_s"] = 0.02
    elif mode == "upper":
        policy.update(setpoint_k=310.0, kp_w_k=1000.0, output_max_w=1000.0)
    elif mode == "lower":
        policy.update(setpoint_k=290.0, kp_w_k=1000.0, output_min_w=-1000.0)
    elif mode == "slew":
        policy.update(
            setpoint_k=310.0,
            kp_w_k=1000.0,
            output_min_w=-5000.0,
            output_max_w=5000.0,
            slew_w_s=10000.0,
        )
    elif mode == "chamber":
        policy.update(
            sensor="primary.chamber.right", actuator="primary.chamber.right", setpoint_k=350.1
        )
    elif mode == "gravity":
        plant = plant_request(height=10.0)
    elif mode != "pi":
        raise ValueError(mode)
    return dict(format="atomic-closed-loop-eos-pi", schema_version=1, plant=plant, policy=policy)


def cases():
    for name in ["pi", "fixed", "filtered", "upper", "lower", "slew", "chamber", "gravity"]:
        yield name, request(mode=name)


def independent_sample(p, state, truth):
    previous_filter, integral, previous_output = state
    fraction = (
        1.0 if p["filter_tau_s"] == 0 else -np.expm1(-p["sample_period_s"] / p["filter_tau_s"])
    )
    measurement = (
        truth if p["filter_tau_s"] == 0 else previous_filter + fraction * (truth - previous_filter)
    )
    error = p["setpoint_k"] - measurement
    proposal = integral + p["ki_w_k_s"] * p["sample_period_s"] * error
    lower = max(p["output_min_w"], previous_output - p["sample_period_s"] * p["slew_w_s"])
    upper = min(p["output_max_w"], previous_output + p["sample_period_s"] * p["slew_w_s"])
    command = p["bias_w"] + p["kp_w_k"] * error + proposal
    clipped = np.clip(command, lower, upper)
    held = bool(
        ((command > clipped and error > 0) or (command < clipped and error < 0))
        and proposal != integral
    )
    if not held:
        integral = proposal
    raw = p["bias_w"] + p["kp_w_k"] * error + integral
    output = float(np.clip(raw, lower, upper))
    return (float(measurement), float(integral), output), held


class IndependentPI:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.plant = IndependentAssembly(spec["plant"])

    def temperatures(self, y):
        result = {}
        e = self.plant
        for i, name in enumerate(["primary", "secondary"]):
            z = y[e.offsets[i] : e.offsets[i + 1]]
            terminal = e.circuits[i]
            cm, cP, cE, cr, cu, cT, cp, cs = terminal.channel.primitive(z[: terminal.start])
            m, P, E, r, u, T, p, s = terminal.primitive(z)
            for j, value in enumerate(cT):
                result[f"{name}.channel.c{j}"] = float(value)
            for side, value in zip(["left", "right"], T):
                result[f"{name}.chamber.{side}"] = float(value)
        for w, value in zip(self.spec["plant"]["walls"], y[e.wall_start : -5] / e.capacity):
            result[f"wall.{w['id']}"] = float(value)
        return result

    def integrate(self, ticks, method):
        e = self.plant
        p = self.spec["policy"]
        y = e.initial.copy()
        controls = e.heaters.copy()
        state = (self.temperatures(y)[p["sensor"]], p["initial_integral_w"], p["initial_output_w"])
        rows = []
        evaluations = 0
        for k in range(ticks):
            truth = self.temperatures(y)[p["sensor"]]
            state, held = independent_sample(p, state, truth)
            controls[p["actuator"]] = state[2]
            base = y.copy()
            start = k * p["sample_period_s"]
            end = (k + 1) * p["sample_period_s"]

            def fun(t, z):
                nonlocal evaluations
                evaluations += 1
                if evaluations > 400000:
                    raise ValueError("Independent PI RHS budget")
                return e.rhs(t, base + z.astype(np.longdouble), controls)

            sol = solve_ivp(
                fun,
                (start, end),
                np.zeros(len(y)),
                method=method,
                rtol=2e-10,
                atol=1e-9,
                max_step=0.001,
            )
            if not sol.success:
                raise ValueError("Independent PI integration failed")
            y = base + sol.y[:, -1].astype(np.longdouble)
            external = np.longdouble(0)
            for i, terminal in enumerate(e.circuits):
                local = y[e.offsets[i] : e.offsets[i + 1]]
                external += local[3 * terminal.channel.n] + local[-4]
            external += -y[-5] + y[-3]
            rows.append(
                dict(
                    time_s=end,
                    external_heat_j=float(external),
                    truth_k=truth,
                    filtered_k=state[0],
                    integral_w=state[1],
                    output_w=state[2],
                    integral_held=held,
                    port_temperatures_k=self.temperatures(y),
                )
            )
        return rows
