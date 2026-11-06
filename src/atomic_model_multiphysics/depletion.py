"""Bounded Metzler chain, CE/CM and energy-augmented CRAM, explicit input data."""

import numpy as np
from scipy.linalg import expm

from physics.depletion.cram import cram16


def operator(decay_per_s, capture_m2, flux_per_m2_s):
    decay = np.asarray(decay_per_s, float)
    capture = np.asarray(capture_m2, float)
    if (
        decay.ndim != 1
        or capture.shape != decay.shape
        or not 2 <= len(decay) <= 64
        or not np.isfinite(decay).all()
        or not np.isfinite(capture).all()
        or np.any(decay < 0)
        or np.any(capture < 0)
        or not np.isfinite(flux_per_m2_s)
        or flux_per_m2_s < 0
    ):
        raise ValueError("Invalid explicitly supplied chain data")
    rate = decay + capture * flux_per_m2_s
    if rate[-1] != 0:
        raise ValueError("Declared chain terminates in a stable accounting sink")
    a = np.diag(-rate)
    for i in range(len(rate) - 1):
        a[i + 1, i] = rate[i]
    return a


def advance(matrix, composition, dt_s, energy_per_transition_j):
    a = np.asarray(matrix, float)
    n = np.asarray(composition, float)
    q = np.asarray(energy_per_transition_j, float)
    size = len(n)
    if (
        a.shape != (size, size)
        or q.shape != (size,)
        or not np.isfinite(a).all()
        or not np.isfinite(n).all()
        or not np.isfinite(q).all()
        or np.any(n < 0)
        or np.any(q < 0)
        or type(dt_s) not in (int, float)
        or not np.isfinite(dt_s)
        or dt_s <= 0
    ):
        raise ValueError("Invalid depletion state/interval")
    off = a - np.diag(np.diag(a))
    if np.any(off < 0) or np.any(np.diag(a) > 0) or np.max(abs(a.sum(axis=0))) > 1e-14:
        raise ValueError("Only conservative nonnegative transition chains admitted")
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = a
    augmented[-1, :size] = (-np.diag(a)) * q
    initial = np.r_[n, 0.0]
    result = cram16(augmented, initial, float(dt_s))
    reference = expm(augmented * dt_s) @ initial
    relative = float(np.max(abs(result - reference) / np.maximum(1.0, abs(reference))))
    inventory = abs(sum(result[:-1]) - sum(n)) / max(1.0, sum(n))
    if np.any(result[:-1] < -1e-12 * max(1.0, max(n))) or relative > 1e-8 or inventory > 1e-10:
        raise ValueError("Depletion positivity/reference/inventory criterion failed")
    if np.any(result[:-1] < 0):
        raise ValueError("Negative trace inventory is reported, never silently projected")
    return dict(
        composition=result[:-1].tolist(),
        released_energy_j=float(result[-1]),
        reference_relative_error=relative,
        inventory_relative=inventory,
        physical_scope="Explicit closed capture/decay test chain, not evaluated fuel depletion library",
    )


def cecm(composition, dt_s, operator_at):
    n = np.asarray(composition, float)
    if not np.isfinite(n).all() or np.any(n < 0) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Invalid CE/CM state")

    def admitted(state):
        matrix = np.asarray(operator_at(state), float)
        if (
            matrix.shape != (len(n), len(n))
            or not np.isfinite(matrix).all()
            or np.any(matrix - np.diag(np.diag(matrix)) < 0)
            or np.any(np.diag(matrix) > 0)
            or np.max(abs(matrix.sum(axis=0))) > 1e-14
        ):
            raise ValueError("Invalid CE/CM transition operator")
        return matrix

    first = admitted(n)
    half = cram16(first, n, dt_s / 2)
    midpoint = admitted(half)
    result = cram16(midpoint, n, dt_s)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Invalid CE/CM accepted state")
    return result
