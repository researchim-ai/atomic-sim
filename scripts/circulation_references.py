"""Independent scalar Riccati/work and Fourier thermal references for fixed cases."""

import numpy as np
from scipy.linalg import expm


def flow_and_work(time_s, initial, head, inertia, resistance, density):
    if head == 0:
        return initial / (1 + resistance * abs(initial) * time_s / inertia), 0.0, None
    sign = float(np.sign(head))
    h = abs(head)
    scale = np.sqrt(h / resistance)
    frequency = np.sqrt(h * resistance) / inertia
    oriented = sign * initial
    factor = h * inertia / (density * resistance)
    if oriented < 0:
        angle = np.arctan(oriented / scale)
        zero = -angle / frequency
        if time_s <= zero:
            theta = angle + frequency * time_s
            return (
                sign * scale * np.tan(theta),
                factor * np.log(np.cos(angle) / np.cos(theta)),
                float(zero),
            )
        x = frequency * (time_s - zero)
        work = factor * (np.log(np.cos(angle)) + np.logaddexp(x, -x) - np.log(2))
        return sign * scale * np.tanh(x), work, float(zero)
    x = frequency * time_s
    ratio = oriented / scale
    flow = sign * scale * (ratio + np.tanh(x)) / (1 + ratio * np.tanh(x))
    # Stable log(cosh(x)+r*sinh(x)), for x>=0 and r>=0.
    work = factor * (x + np.log((1 + ratio) + (1 - ratio) * np.exp(-2 * x)) - np.log(2))
    return flow, work, None


def switched_reference(time_s, initial, head, inertia, resistance, density, commands):
    flow = initial
    work = 0.0
    stamp = 0.0
    zeros = []
    active = head
    for command_time, new_head in list(commands) + [(time_s, None)]:
        if command_time > time_s:
            break
        dt = command_time - stamp
        flow, increment, zero = flow_and_work(dt, flow, active, inertia, resistance, density)
        if zero is not None and zero <= dt:
            zeros.append(stamp + zero)
        work += increment
        stamp = command_time
        if new_head is not None:
            active = new_head
        else:
            return float(flow), float(work), zeros
    if stamp < time_s:
        flow, increment, zero = flow_and_work(
            time_s - stamp, flow, active, inertia, resistance, density
        )
        if zero is not None and zero <= time_s - stamp:
            zeros.append(stamp + zero)
        work += increment
    return float(flow), float(work), zeros


def modal_temperatures(
    time_s,
    fluid_initial,
    wall_initial,
    flow,
    density,
    cp,
    area,
    length,
    loss_coefficient,
    wall_capacity,
    ua,
    ambient_ua,
    ambient_temperature,
):
    """Uniform segments, constant equilibrium flow, zero wall heaters."""
    count = len(fluid_initial)
    capacity = density * area * length * cp
    resistance = loss_coefficient / (2 * density * area**2)
    dissipation = resistance * abs(flow) ** 3 / density
    wall_equilibrium = ambient_temperature + dissipation / ambient_ua
    fluid_equilibrium = wall_equilibrium + dissipation / ua
    modes = np.fft.fft(
        np.column_stack(
            (np.array(fluid_initial) - fluid_equilibrium, np.array(wall_initial) - wall_equilibrium)
        ),
        axis=0,
    )
    result = np.zeros_like(modes)
    for k in range(count):
        theta = 2 * np.pi * k / count
        adv = abs(flow) / (density * area * length) * (np.exp(-1j * np.sign(flow) * theta) - 1)
        matrix = np.array(
            [
                [adv - ua / capacity, ua / capacity],
                [ua / wall_capacity, -(ua + ambient_ua) / wall_capacity],
            ],
            dtype=complex,
        )
        result[k] = expm(matrix * time_s) @ modes[k]
    temperature = np.fft.ifft(result, axis=0).real
    return temperature[:, 0] + fluid_equilibrium, temperature[:, 1] + wall_equilibrium


def adiabatic_temperatures(
    time_s, initial_temperature, initial_flow, head, density, cp, area, lengths, losses
):
    """Exact semidiscrete solution via transported mass s=integral(abs(m) dt).

    On each fixed-direction phase d(m^2)/ds=2*(sign(m)*H-R*m^2)/I,
    so temperature advection plus cubic dissipation becomes a linear augmented
    system in s. No production RHS, Jacobian or numerical time solver is used.
    """
    lengths = np.asarray(lengths, dtype=float)
    losses = np.asarray(losses, dtype=float)
    inertia = float(sum(lengths) / area)
    resistance = float(sum(losses) / (2 * density * area**2))
    _, _, zero = flow_and_work(time_s, initial_flow, head, inertia, resistance, density)
    if zero is not None and 0 < zero < time_s:
        at_zero = adiabatic_temperatures(
            zero, initial_temperature, initial_flow, head, density, cp, area, lengths, losses
        )
        return adiabatic_temperatures(
            time_s - zero, at_zero, 0.0, head, density, cp, area, lengths, losses
        )
    sign = float(np.sign(initial_flow) if initial_flow != 0 else np.sign(head))
    if sign == 0:
        return np.array(initial_temperature, dtype=float)
    if head == 0:
        transported = (
            inertia / resistance * np.log1p(resistance * abs(initial_flow) * time_s / inertia)
        )
    else:
        _, work, _ = flow_and_work(time_s, initial_flow, head, inertia, resistance, density)
        transported = abs(density * work / head)
    count = len(lengths)
    capacity = density * area * lengths * cp
    matrix = np.zeros((count + 2, count + 2))
    donor = np.roll(np.arange(count), 1 if sign > 0 else -1)
    for i in range(count):
        matrix[i, i] -= cp / capacity[i]
        matrix[i, donor[i]] += cp / capacity[i]
        matrix[i, count] = (losses[i] / (2 * density * area**2)) / (density * capacity[i])
    matrix[count, count] = -2 * resistance / inertia
    matrix[count, -1] = 2 * sign * head / inertia
    return (expm(matrix * transported) @ np.r_[initial_temperature, initial_flow**2, 1.0])[:count]
