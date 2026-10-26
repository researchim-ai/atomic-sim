"""Линейный фильтр Калмана для демонстрации оценивания на синтетических данных модели.

Это не цифровой двойник блока: измерения порождены той же моделью.
"""

from __future__ import annotations

import numpy as np


def kalman_track(
    truth: np.ndarray, measurement_std: float, process_std: float, seed: int = 1
) -> dict:
    rng = np.random.default_rng(seed)
    measurements = truth + rng.normal(0.0, measurement_std, size=truth.shape)
    estimate = np.zeros_like(truth)
    variance = np.zeros_like(truth)
    estimate[0] = measurements[0]
    variance[0] = measurement_std**2
    for i in range(1, truth.size):
        predicted = estimate[i - 1]
        predicted_var = variance[i - 1] + process_std**2
        gain = predicted_var / (predicted_var + measurement_std**2)
        estimate[i] = predicted + gain * (measurements[i] - predicted)
        variance[i] = (1.0 - gain) * predicted_var
    residual = estimate - truth
    rmse = float(np.sqrt(np.mean(residual**2)))
    open_loop = float(np.sqrt(np.mean((measurements - truth) ** 2)))
    band = 2.0 * np.sqrt(variance)
    coverage = float(np.mean(np.abs(residual) <= band))
    return {"rmse": rmse, "measurement_rmse": open_loop, "coverage": coverage, "estimate": estimate}
