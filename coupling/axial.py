"""Осевая связка одногрупповой диффузии и сосредоточенного топлива.

На каждом узле UA (T - T_cool) = q'. Интеграл q' равен заданной мощности.
Радиальный профиль твэла этой моделью не заменяется: он считается отдельно.
"""

from __future__ import annotations

import numpy as np

from core.mesh import Mesh1D
from core.status import DomainError, ModelPassport, ModelStatus
from physics.neutronics.diffusion import diffusion_operator, fission_operator, k_eigenvalue

PASSPORT = ModelPassport(
    model_id="axial-lumped-fuel",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Стационарная форма мощности и температура топлива вдоль канала",
    approximations=(
        "одна группа",
        "мощность задана, собственное значение определяет только форму и k",
        "топливо в узле — одна теплоёмкость с проводимостью UA на метр",
        "поглощение линейно по sqrt(T)",
    ),
    applicability="Sigma_a остаётся положительным; итерация температуры сходится",
    data_sources=("Конечно-объёмная диффузия и баланс q' = UA (T - T_cool)",),
)


def steady_axial_power(
    height_m: float,
    n_cells: int,
    power_w: float,
    diffusion_m: float,
    sigma_a_ref_per_m: float,
    nu_sigma_f_per_m: float,
    doppler_per_sqrt_k: float,
    temperature_ref_k: float,
    coolant_temperature_k: float,
    linear_conductance_w_k_m: float,
    iterations: int = 40,
    tolerance_k: float = 1.0e-4,
) -> dict:
    if power_w <= 0.0 or linear_conductance_w_k_m <= 0.0 or n_cells < 4:
        raise DomainError("Мощность, проводимость и число узлов осевой связки заданы неверно")
    mesh = Mesh1D.uniform(height_m, n_cells)
    temperature = np.full(n_cells, temperature_ref_k)
    k_value = 1.0
    flux = np.ones(n_cells)
    for _ in range(iterations):
        sigma_a = sigma_a_ref_per_m + doppler_per_sqrt_k * (
            np.sqrt(temperature) - np.sqrt(temperature_ref_k)
        )
        if np.any(sigma_a <= 0.0):
            raise DomainError("Доплер увёл поглощение из положительной области")
        removal = diffusion_operator(mesh, np.full(n_cells, diffusion_m), sigma_a)
        fission = fission_operator(mesh, np.full(n_cells, nu_sigma_f_per_m))
        k_value, flux, converged = k_eigenvalue(removal, fission, tol=1.0e-10)
        if not converged:
            raise DomainError("k-задача осевой связки не сошлась")
        weight = flux * mesh.dx
        heating = power_w * weight / float(np.sum(weight))
        updated = coolant_temperature_k + heating / (linear_conductance_w_k_m * mesh.dx)
        if float(np.max(np.abs(updated - temperature))) < tolerance_k:
            temperature = updated
            break
        temperature = updated
    else:
        raise DomainError("Температура топлива и поток не сошлись")
    if abs(float(np.sum(heating)) - power_w) > 1.0e-6 * power_w:
        raise DomainError("Осевая мощность не сохранила интеграл")
    return {
        "k": float(k_value),
        "flux": flux,
        "temperature_k": temperature,
        "power_w": heating,
        "mesh": mesh,
    }
