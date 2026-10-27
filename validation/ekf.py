"""Линейный фильтр Калмана для точечной кинетики по измерению амплитуды.

Предшественники не читаются из истины. Их оценка сравнивается с истиной только
после прогона, как независимый holdout скрытого состояния.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from physics.neutronics.point_kinetics import PointKinetics


def ekf_track_amplitude(
    reactivity: float, dt_s: float, steps: int, measurement_std: float, seed: int
) -> dict:
    kinetics = PointKinetics.thermal_u235()
    state = kinetics.initial_state(1.0)
    truth = [state.copy()]
    for _ in range(steps):
        state = kinetics.reference_step(state, reactivity, dt_s)
        truth.append(state.copy())
    truth_arr = np.vstack(truth)
    truth_n = truth_arr[:, 0]
    rng = np.random.default_rng(seed)
    measurements = truth_n + rng.normal(0.0, measurement_std, size=truth_n.shape)

    dim = truth_arr.shape[1]
    sensitivity = np.concatenate(
        [[1.0], kinetics.beta / (kinetics.generation_time_s * kinetics.decay)]
    )
    estimate = sensitivity * measurements[0]
    covariance = np.outer(sensitivity, sensitivity) * measurement_std**2
    transition = expm(kinetics.matrix(reactivity) * dt_s)
    observation = np.zeros(dim)
    observation[0] = 1.0
    process = np.eye(dim) * 1.0e-18
    noise = measurement_std**2
    estimates = np.zeros((steps + 1, dim))
    variance = np.zeros(steps + 1)
    estimates[0] = estimate
    variance[0] = covariance[0, 0]
    identity = np.eye(dim)
    for index in range(steps):
        predicted = transition @ estimate
        predicted_cov = transition @ covariance @ transition.T + process
        innovation_var = float(observation @ predicted_cov @ observation + noise)
        gain = predicted_cov @ observation / innovation_var
        estimate = predicted + gain * (measurements[index + 1] - float(observation @ predicted))
        innovation_gain = identity - np.outer(gain, observation)
        covariance = (
            innovation_gain @ predicted_cov @ innovation_gain.T + noise * np.outer(gain, gain)
        )
        covariance = 0.5 * (covariance + covariance.T)
        estimates[index + 1] = estimate
        variance[index + 1] = covariance[0, 0]

    residual = estimates[:, 0] - truth_n
    rmse = float(np.sqrt(np.mean(residual**2)))
    measurement_rmse = float(np.sqrt(np.mean((measurements - truth_n) ** 2)))
    coverage = float(np.mean(np.abs(residual) <= 2.0 * np.sqrt(np.maximum(variance, 0.0))))
    holdout = slice(steps // 2, None)
    holdout_rmse = float(np.sqrt(np.mean(residual[holdout] ** 2)))
    precursor_error = estimates[holdout, 1:] - truth_arr[holdout, 1:]
    measured_precursors = np.vstack(
        [kinetics.equilibrium_precursors(float(value)) for value in measurements[holdout]]
    )
    precursor_baseline = measured_precursors - truth_arr[holdout, 1:]
    return {
        "rmse": rmse,
        "measurement_rmse": measurement_rmse,
        "holdout_rmse": holdout_rmse,
        "coverage": coverage,
        "precursor_holdout_rmse": float(np.sqrt(np.mean(precursor_error**2))),
        "precursor_measurement_rmse": float(np.sqrt(np.mean(precursor_baseline**2))),
    }
