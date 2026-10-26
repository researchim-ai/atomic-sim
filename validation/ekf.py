"""Расширенный фильтр Калмана для амплитуды точечной кинетики на синтетических данных."""

from __future__ import annotations

import numpy as np

from physics.neutronics.point_kinetics import PointKinetics


def ekf_track_amplitude(reactivity: float, dt_s: float, steps: int, measurement_std: float, seed: int) -> dict:
    kinetics = PointKinetics.thermal_u235()
    state = kinetics.initial_state(1.0)
    truth = [state.copy()]
    for _ in range(steps):
        state = kinetics.reference_step(state, reactivity, dt_s)
        truth.append(state.copy())
    truth_n = np.array([item[0] for item in truth])
    rng = np.random.default_rng(seed)
    measurements = truth_n + rng.normal(0.0, measurement_std, size=truth_n.shape)
    estimate = np.zeros(steps + 1)
    variance = np.zeros(steps + 1)
    estimate[0] = measurements[0]
    variance[0] = measurement_std**2
    process_std = 1.0e-4
    # Линеаризация dn/dt ≈ ((rho-beta)/Lambda) n + sum lambda C, C берём из истины только как
    # известную динамику запаздывающих групп этой синтетической задачи наблюдения амплитуды.
    for index in range(steps):
        precursors = truth[index][1:]
        derivative = kinetics.derivatives(estimate[index], precursors, reactivity, 0.0)[0]
        slope = (reactivity - kinetics.beta_total) / kinetics.generation_time_s
        predicted = estimate[index] + dt_s * derivative
        predicted_var = (1.0 + dt_s * slope) ** 2 * variance[index] + process_std**2
        gain = predicted_var / (predicted_var + measurement_std**2)
        estimate[index + 1] = predicted + gain * (measurements[index + 1] - predicted)
        variance[index + 1] = (1.0 - gain) * predicted_var
    residual = estimate - truth_n
    rmse = float(np.sqrt(np.mean(residual**2)))
    measurement_rmse = float(np.sqrt(np.mean((measurements - truth_n) ** 2)))
    coverage = float(np.mean(np.abs(residual) <= 2.0 * np.sqrt(variance)))
    holdout = slice(steps // 2, None)
    holdout_rmse = float(np.sqrt(np.mean(residual[holdout] ** 2)))
    return {
        "rmse": rmse,
        "measurement_rmse": measurement_rmse,
        "holdout_rmse": holdout_rmse,
        "coverage": coverage,
    }
