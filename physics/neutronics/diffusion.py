"""Многогрупповая конечно-объёмная диффузия на одномерной сетке.

Для группы g баланс ячейки:
поток утечки + Sigma_r,g * phi * V = источники рассеяния, деления и предшественников.
Внутригрупповое рассеяние входит в Sigma_r и не вычитается повторно.
Скорость нейтронов — физический параметр задачи; для замедления шага она не уменьшается.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.mesh import Mesh1D
from core.status import DomainError, ModelPassport, ModelStatus

THERMAL_NEUTRON_SPEED_M_S = 2200.0
PASSPORT = ModelPassport(
    model_id="fv-diffusion-1d",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Одномерная многогрупповая диффузия, стационарная k-задача",
    approximations=(
        "скалярная диффузия",
        "конечно-объёмные ячейки и гармонический интерфейс D",
        "вакуум в форме нулевого потока на грани объёма",
        "отражение — нулевой ток",
    ),
    applicability="Задачи, где диффузионное приближение согласовано с данными сечений; транспортный расчёт не подменяется",
    data_sources=("Сечения передаются вызывающей стороной",),
)


@dataclass
class GroupData:
    diffusion_m: np.ndarray
    sigma_removal_per_m: np.ndarray
    nu_sigma_fission_per_m: np.ndarray
    chi: np.ndarray
    speed_m_s: np.ndarray
    scatter_to: np.ndarray  # [g_to, g_from, cell]


def removal_with_scatter(sigma_absorption: np.ndarray, scatter: np.ndarray) -> np.ndarray:
    """Sigma_r,g = Sigma_a,g + sum_{g'!=g} Sigma_s,g->g'.

    `scatter[g_to, g_from, cell]`. Внутригрупповой член в сумму выхода не входит.
    """
    groups = scatter.shape[0]
    removal = np.array(sigma_absorption, dtype=float, copy=True)
    for group in range(groups):
        for destination in range(groups):
            if destination == group:
                continue
            removal[group] += scatter[destination, group]
    return removal


def macroscopic_from_micro(
    number_density_per_m3: dict, micro_per_m2: dict, reaction: str
) -> np.ndarray:
    """Sigma = sum N_i * sigma_i. Плотности в 1/м^3, микросечения в м^2."""
    total = None
    for isotope, density in number_density_per_m3.items():
        sigma = np.asarray(micro_per_m2[isotope][reaction], dtype=float)
        term = density * sigma
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Нет изотопов для макросечения")
    return total


def _face_conductance(d_left: float, d_right: float, dx_left: float, dx_right: float) -> float:
    resistance = 0.5 * dx_left / d_left + 0.5 * dx_right / d_right
    return 1.0 / resistance


def diffusion_operator(
    mesh: Mesh1D,
    diffusion: np.ndarray,
    sigma_removal: np.ndarray,
    left: str = "dirichlet0",
    right: str = "dirichlet0",
) -> np.ndarray:
    """Оператор утечки и съёма на одну группу. Уравнение A phi = source, source включает деление/V."""
    n = mesh.n_cells
    matrix = np.zeros((n, n))
    for i, dx in enumerate(mesh.dx):
        matrix[i, i] += sigma_removal[i] * dx
        if i == 0:
            if left == "dirichlet0":
                matrix[i, i] += diffusion[i] / (0.5 * dx)
            elif left != "reflective":
                raise DomainError("Граничное условие %s не поддержано" % left)
        else:
            g = _face_conductance(diffusion[i - 1], diffusion[i], mesh.dx[i - 1], dx)
            matrix[i, i] += g
            matrix[i, i - 1] -= g
        if i == n - 1:
            if right == "dirichlet0":
                matrix[i, i] += diffusion[i] / (0.5 * dx)
            elif right != "reflective":
                raise DomainError("Граничное условие %s не поддержано" % right)
        else:
            g = _face_conductance(diffusion[i], diffusion[i + 1], dx, mesh.dx[i + 1])
            matrix[i, i] += g
            matrix[i, i + 1] -= g
    return matrix


def fission_operator(mesh: Mesh1D, nu_sigma_fission: np.ndarray) -> np.ndarray:
    return np.diag(nu_sigma_fission * mesh.dx)


def k_eigenvalue(
    removal: np.ndarray, fission: np.ndarray, tol: float = 1e-10, max_iter: int = 200
) -> tuple:
    """Степенные итерации A phi = (1/k) F phi. Возвращает k и поток, нормированный на максимум 1."""
    n = removal.shape[0]
    phi = np.ones(n)
    k_value = 1.0
    for _ in range(max_iter):
        source = fission @ phi
        phi_new = np.linalg.solve(removal, source)
        source_new = fission @ phi_new
        denom = float(np.sum(source))
        if denom <= 0.0:
            raise DomainError("Источник деления нулевой, k-задача не определена")
        k_new = float(np.sum(source_new) / denom)
        scale = np.max(np.abs(phi_new))
        if scale == 0.0:
            raise DomainError("Поток схлопнулся")
        phi_new = phi_new / scale
        flux_err = np.linalg.norm(phi_new - phi) / max(np.linalg.norm(phi_new), 1e-30)
        k_err = abs(k_new - k_value)
        phi = phi_new
        k_value = k_new
        if flux_err < tol and k_err < tol:
            return k_value, phi, True
    return k_value, phi, False


def analytical_slab_k(
    diffusion_m: float, sigma_a: float, nu_sigma_f: float, height_m: float
) -> float:
    """Однородная пластина с нулевым потоком на концах. B^2 = (pi/H)^2."""
    buckling = (np.pi / height_m) ** 2
    return nu_sigma_f / (sigma_a + diffusion_m * buckling)


def rod_absorption(
    base_sigma_a: np.ndarray, inserted_fraction: float, rod_delta_sigma: float
) -> np.ndarray:
    """Доля поглотителя в ячейках при вводе сверху. inserted_fraction=1 заполняет всю высоту.

    Скачок целого индекса не используется: граница стержня делит ячейку пропорционально объёму.
    """
    if inserted_fraction < 0.0 or inserted_fraction > 1.0:
        raise DomainError("Доля ввода стержня должна быть в [0, 1]")
    n = base_sigma_a.size
    depth = inserted_fraction * n
    sigma = np.array(base_sigma_a, dtype=float, copy=True)
    full = int(np.floor(depth))
    if full > 0:
        sigma[:full] += rod_delta_sigma
    if full < n:
        fraction = depth - full
        sigma[full] += fraction * rod_delta_sigma
    return sigma


def two_group_infinite_k(
    sigma_a: np.ndarray, scatter_12: float, nu_sigma_f: np.ndarray, chi: np.ndarray
) -> float:
    """Аналитический k_inf для двух групп без утечки и без обратного рассеяния.

    Группа 0 — быстрая, рассеяние только 0→1. chi[0] должен быть 1.
    """
    if chi[0] != 1.0 or abs(chi[1]) > 0.0:
        raise DomainError("Эта аналитическая формула задана для chi=(1, 0)")
    removal_0 = sigma_a[0] + scatter_12
    return float(nu_sigma_f[1] * scatter_12 / (sigma_a[1] * removal_0) + nu_sigma_f[0] / removal_0)
