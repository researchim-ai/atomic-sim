"""Векторизованная точечная кинетика. Уравнения те же, что у скалярного модуля."""

from __future__ import annotations

import numpy as np

from physics.neutronics.point_kinetics import PointKinetics, batch_derivatives


def torch_batch_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return True


def integrate_batch_euler(reactivity: np.ndarray, dt_s: float, steps: int, kinetics: PointKinetics | None = None) -> np.ndarray:
    model = kinetics or PointKinetics.thermal_u235()
    batch = reactivity.shape[0]
    amplitudes = np.ones(batch)
    precursors = np.vstack([model.equilibrium_precursors(1.0) for _ in range(batch)])
    step = dt_s / steps
    for _ in range(steps):
        dn, dc_dt = batch_derivatives(amplitudes, precursors, reactivity, model)
        amplitudes = amplitudes + step * dn
        precursors = precursors + step * dc_dt
    return amplitudes


def torch_batch_derivative(reactivity: np.ndarray, amplitudes: np.ndarray, precursors: np.ndarray, kinetics: PointKinetics) -> tuple:
    import torch

    beta = torch.tensor(kinetics.beta, dtype=torch.float64)
    decay = torch.tensor(kinetics.decay, dtype=torch.float64)
    rho = torch.tensor(reactivity, dtype=torch.float64)
    n = torch.tensor(amplitudes, dtype=torch.float64)
    precursor = torch.tensor(precursors, dtype=torch.float64)
    beta_total = float(kinetics.beta_total)
    generation = kinetics.generation_time_s
    dn = ((rho - beta_total) / generation) * n + precursor @ decay
    dc_dt = n[:, None] * beta / generation - precursor * decay
    return dn.numpy(), dc_dt.numpy()
