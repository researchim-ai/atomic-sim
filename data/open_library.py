"""Открытая тестовая двухгрупповая библиотека. Не является библиотекой ВВЭР."""

from __future__ import annotations

import numpy as np

from physics.neutronics.xs import CrossSectionProvider


def open_two_group_provider() -> CrossSectionProvider:
    """Две температуры, сечения подобраны так, чтобы макросечение совпало с N*sigma в тесте."""
    temperatures = np.array([600.0, 1200.0])
    # Группы: быстрая, тепловая. Размерности СИ.
    diffusion = np.array([[0.012, 0.004], [0.013, 0.0045]])
    sigma_a = np.array([[0.8, 8.0], [0.7, 7.0]])
    sigma_f = np.array([[0.2, 6.0], [0.18, 5.5]])
    nu_sigma_f = np.array([[0.5, 15.0], [0.45, 13.5]])
    kappa_sigma_f = 3.2e-11 * sigma_f
    scatter = np.zeros((2, 2, 2))
    # scatter[temperature, g_to, g_from]
    scatter[0, 1, 0] = 5.0
    scatter[1, 1, 0] = 4.6
    return CrossSectionProvider(
        temperatures,
        {
            "D": diffusion,
            "sigma_a": sigma_a,
            "sigma_f": sigma_f,
            "nu_sigma_f": nu_sigma_f,
            "kappa_sigma_f": kappa_sigma_f,
            "scatter": scatter,
        },
    )
