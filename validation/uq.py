"""Чувствительность и распространение неопределённости без закрытых бенчмарков."""

from __future__ import annotations

import numpy as np


def equilibrium_amplitude(reactivity: float, generation_time_s: float, source: float) -> float:
    """Подкритическая амплитуда n = -S * Lambda / rho."""
    return -source * generation_time_s / reactivity


def monte_carlo_amplitude(reactivity_mean: float, reactivity_std: float, generation_time_s: float, source: float, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values = reactivity_mean + reactivity_std * rng.standard_normal(samples)
    values = values[values < -1.0e-6]
    amplitudes = equilibrium_amplitude(values, generation_time_s, source)
    return {
        "mean": float(np.mean(amplitudes)),
        "p05": float(np.quantile(amplitudes, 0.05)),
        "p95": float(np.quantile(amplitudes, 0.95)),
        "samples": int(amplitudes.size),
    }


def finite_difference_sensitivity(reactivity: float, generation_time_s: float, source: float, step: float = 1.0e-5) -> float:
    high = equilibrium_amplitude(reactivity + step, generation_time_s, source)
    low = equilibrium_amplitude(reactivity - step, generation_time_s, source)
    return (high - low) / (2.0 * step)
