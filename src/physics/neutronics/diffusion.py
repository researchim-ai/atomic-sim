"""Многогрупповая конечно-объёмная диффузия на одномерной сетке.

Для группы g баланс ячейки:
поток утечки + Sigma_r,g * phi * V = источники рассеяния, деления и предшественников.
Внутригрупповое рассеяние входит в Sigma_r и не вычитается повторно.
Скорость нейтронов — физический параметр задачи; для замедления шага она не уменьшается.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.mesh import Mesh1D
from core.status import DomainError, ModelPassport, ModelStatus

THERMAL_NEUTRON_SPEED_M_S = 2200.0


def flux_weighted_sigma(sigma_per_m, flux, volumes_m3) -> float:
    """Σ = Σ (φ V Σ) / Σ (φ V). Плоский поток совпадает со средним по объёму."""
    sigma = np.asarray(sigma_per_m, dtype=float)
    weight_flux = np.asarray(flux, dtype=float)
    volumes = np.asarray(volumes_m3, dtype=float)
    if sigma.shape != weight_flux.shape or sigma.shape != volumes.shape:
        raise DomainError("Поток, объём и сечение заданы на разных зонах")
    if np.any(volumes <= 0.0) or np.any(weight_flux < 0.0) or np.any(sigma < 0.0):
        raise DomainError("Взвешивание сечения задано при отрицательном потоке или объёме")
    weight = weight_flux * volumes
    total = float(np.sum(weight))
    if total <= 0.0:
        raise DomainError("Вес потока для гомогенизации нулевой")
    return float(np.sum(sigma * weight) / total)


def mean_free_path_m(sigma_per_m: float) -> float:
    """Средняя длина свободного пробега 1/Σ."""
    if sigma_per_m <= 0.0:
        raise DomainError("Макросечение длины пробега должно быть положительным")
    return 1.0 / sigma_per_m


def uncollided_fraction(sigma_per_m: float, distance_m: float) -> float:
    """Доля частиц без столкновения на пути x: exp(−Σ x)."""
    if sigma_per_m < 0.0 or distance_m < 0.0:
        raise DomainError("Ослабление задано при отрицательном сечении или пути")
    return math.exp(-sigma_per_m * distance_m)


def absorption_reactivity_delta(delta_sigma_per_m: float, nu_sigma_f: float) -> float:
    """Δρ = −ΔΣ / νΣf для одногруппового ρ = (k−1)/k. Тождество не требует k = 1."""
    if nu_sigma_f <= 0.0:
        raise DomainError("νΣf реактивности поглощения должно быть положительным")
    return -delta_sigma_per_m / nu_sigma_f


def reactivity_from_k(k_eigenvalue: float) -> float:
    if k_eigenvalue <= 0.0:
        raise DomainError("Реактивность по k задана при неположительном k")
    return (k_eigenvalue - 1.0) / k_eigenvalue


def critical_buckling_m2(diffusion_m: float, sigma_a: float, nu_sigma_f: float) -> float:
    """B² = (νΣf − Σa) / D, при котором одногрупповой k равен единице."""
    if diffusion_m <= 0.0 or sigma_a <= 0.0 or nu_sigma_f <= sigma_a:
        raise DomainError("Критический баклинг задан для надкритической бесконечной среды и D > 0")
    return (nu_sigma_f - sigma_a) / diffusion_m


def generation_time_s(speed_m_s: float, nu_sigma_f_per_m: float) -> float:
    """Λ = 1 / (v νΣf) однородной зоны без утечки."""
    if speed_m_s <= 0.0 or nu_sigma_f_per_m <= 0.0:
        raise DomainError("Время генерации задано при неположительной скорости или νΣf")
    return 1.0 / (speed_m_s * nu_sigma_f_per_m)


def diffusion_length_m(diffusion_m: float, sigma_a: float) -> float:
    """Длина диффузии L = sqrt(D / Σa)."""
    if diffusion_m <= 0.0 or sigma_a <= 0.0:
        raise DomainError("Длина диффузии задана при неположительных D или Σa")
    return math.sqrt(diffusion_m / sigma_a)


def critical_slab_height_m(diffusion_m: float, sigma_a: float, nu_sigma_f: float) -> float:
    """Высота пластины с нулевым потоком на торцах, при которой одногрупповой k равен единице."""
    return math.pi / math.sqrt(critical_buckling_m2(diffusion_m, sigma_a, nu_sigma_f))


def buckling_k(diffusion_m: float, sigma_a: float, nu_sigma_f: float, buckling2: float) -> float:
    if min(diffusion_m, sigma_a, nu_sigma_f) <= 0.0 or buckling2 < 0.0:
        raise DomainError("k(B²) задано при неположительных сечениях или отрицательном баклинге")
    return nu_sigma_f / (sigma_a + diffusion_m * buckling2)


def buckling_k_slope(
    diffusion_m: float, sigma_a: float, nu_sigma_f: float, buckling2: float
) -> float:
    """dk/dB² = -D νΣf / (Σa + D B²)²."""
    if min(diffusion_m, sigma_a, nu_sigma_f) <= 0.0 or buckling2 < 0.0:
        raise DomainError("Наклон k(B²) задан вне области")
    denominator = sigma_a + diffusion_m * buckling2
    return -diffusion_m * nu_sigma_f / denominator**2


def homogenize_parallel(volumes_m3, sigma_per_m, diffusion_m) -> tuple:
    """Параллельные зоны при общем потоке: Σ и D — средние по объёму."""
    volumes = np.asarray(volumes_m3, dtype=float)
    sigma = np.asarray(sigma_per_m, dtype=float)
    diffusion = np.asarray(diffusion_m, dtype=float)
    if (
        volumes.shape != sigma.shape
        or volumes.shape != diffusion.shape
        or np.any(volumes <= 0.0)
        or np.any(sigma < 0.0)
        or np.any(diffusion <= 0.0)
    ):
        raise DomainError("Параллельная гомогенизация задана вне области")
    total = float(np.sum(volumes))
    return float(np.sum(volumes * sigma) / total), float(np.sum(volumes * diffusion) / total)


def homogenize_series(lengths_m, sigma_per_m, diffusion_m) -> tuple:
    """Последовательные слои: Σ по длине, D — гармоническое среднее."""
    lengths = np.asarray(lengths_m, dtype=float)
    sigma = np.asarray(sigma_per_m, dtype=float)
    diffusion = np.asarray(diffusion_m, dtype=float)
    if (
        lengths.shape != sigma.shape
        or lengths.shape != diffusion.shape
        or np.any(lengths <= 0.0)
        or np.any(sigma < 0.0)
        or np.any(diffusion <= 0.0)
    ):
        raise DomainError("Последовательная гомогенизация задана вне области")
    total = float(np.sum(lengths))
    harmonic = total / float(np.sum(lengths / diffusion))
    return float(np.sum(lengths * sigma) / total), harmonic


def sine_slab_peak_to_average(n_cells: int) -> float:
    """Пик к среднему по ячейкам sin(πx) на [0, 1]. Нечётная сетка содержит центральную ячейку."""
    if n_cells < 1 or n_cells % 2 == 0:
        raise DomainError("Синус пластины считается на положительной нечётной сетке")
    dx = 1.0 / n_cells
    averages = []
    for index in range(n_cells):
        left = index * dx
        right = left + dx
        averages.append((math.cos(math.pi * left) - math.cos(math.pi * right)) / (math.pi * dx))
    return max(averages) / (sum(averages) / n_cells)


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


def outscatter(scatter: np.ndarray) -> np.ndarray:
    """Увод из группы. Внутригрупповой член scatter[g, g] не входит."""
    matrix = np.asarray(scatter, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DomainError("Матрица рассеяния должна быть квадратной")
    if np.any(matrix < 0.0):
        raise DomainError("Сечение рассеяния отрицательно")
    groups = matrix.shape[0]
    leaving = np.zeros(groups)
    for group in range(groups):
        for destination in range(groups):
            if destination != group:
                leaving[group] += matrix[destination, group]
    return leaving


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


def diffusion_matvec(
    mesh: Mesh1D,
    diffusion: np.ndarray,
    sigma_removal: np.ndarray,
    phi: np.ndarray,
    left: str = "dirichlet0",
    right: str = "dirichlet0",
) -> np.ndarray:
    """Разреженное применение того же оператора, без плотной матрицы."""
    n = mesh.n_cells
    out = np.asarray(sigma_removal, dtype=float) * mesh.dx * np.asarray(phi, dtype=float)
    for i, dx in enumerate(mesh.dx):
        if i == 0:
            if left == "dirichlet0":
                out[i] += diffusion[i] / (0.5 * dx) * phi[i]
            elif left != "reflective":
                raise DomainError("Граничное условие %s не поддержано" % left)
        else:
            g = _face_conductance(diffusion[i - 1], diffusion[i], mesh.dx[i - 1], dx)
            out[i] += g * (phi[i] - phi[i - 1])
        if i == n - 1:
            if right == "dirichlet0":
                out[i] += diffusion[i] / (0.5 * dx) * phi[i]
            elif right != "reflective":
                raise DomainError("Граничное условие %s не поддержано" % right)
        else:
            g = _face_conductance(diffusion[i], diffusion[i + 1], dx, mesh.dx[i + 1])
            out[i] += g * (phi[i] - phi[i + 1])
    return out


def adjoint_generation_time_s(
    phi: np.ndarray,
    phi_star: np.ndarray,
    nu_sigma_f_per_m: np.ndarray,
    speed_m_s: float,
    dx_m: np.ndarray,
) -> float:
    """Lambda = <phi*, phi/v> / <phi*, nuSigma_f phi>. Одногрупповой вес, не beta_eff всех установок."""
    if speed_m_s <= 0.0:
        raise DomainError("Скорость для времени генерации должна быть положительной")
    numerator = float(np.sum(phi_star * phi / speed_m_s * dx_m))
    denominator = float(np.sum(phi_star * nu_sigma_f_per_m * phi * dx_m))
    if numerator <= 0.0 or denominator <= 0.0:
        raise DomainError("Вес времени генерации не положительный")
    return numerator / denominator


def adjoint_delayed_fraction(
    phi: np.ndarray,
    phi_star: np.ndarray,
    beta,
    nu_sigma_f_per_m: np.ndarray,
    dx_m: np.ndarray,
) -> float:
    """beta_eff = <phi*, beta nuSigma_f phi> / <phi*, nuSigma_f phi>.

    Постоянная по зоне доля даёт саму себя. Это не beta_eff паспорта ВВЭР.
    """
    flux = np.asarray(phi, dtype=float)
    weight = np.asarray(phi_star, dtype=float)
    fission = np.asarray(nu_sigma_f_per_m, dtype=float)
    widths = np.asarray(dx_m, dtype=float)
    delayed = np.asarray(beta, dtype=float)
    if delayed.ndim == 0:
        delayed = np.full(flux.shape, float(delayed))
    if flux.shape != weight.shape or flux.shape != fission.shape or flux.shape != widths.shape:
        raise DomainError("Поля beta_eff должны быть заданы на одних ячейках")
    if delayed.shape != flux.shape:
        raise DomainError("Доля запаздывающих нейтронов задана не на сетке потока")
    if np.any(delayed < 0.0) or np.any(delayed >= 1.0):
        raise DomainError("Доля запаздывающих нейтронов должна лежать в [0, 1)")
    numerator = float(np.sum(weight * delayed * fission * flux * widths))
    denominator = float(np.sum(weight * fission * flux * widths))
    if denominator <= 0.0:
        raise DomainError("Вес деления для beta_eff не положительный")
    return numerator / denominator


def adjoint_delayed_group_fractions(
    phi: np.ndarray,
    phi_star: np.ndarray,
    beta_groups: np.ndarray,
    nu_sigma_f_per_m: np.ndarray,
    dx_m: np.ndarray,
) -> np.ndarray:
    """beta_eff каждой группы. Сумма групп совпадает с beta_eff их суммы."""
    groups = np.asarray(beta_groups, dtype=float)
    cells = np.asarray(phi, dtype=float).size
    if groups.ndim == 1:
        groups = np.broadcast_to(groups[:, None], (groups.size, cells)).copy()
    if groups.ndim != 2 or groups.shape[1] != cells:
        raise DomainError("Групповые доли запаздывающих нейтронов заданы не на ячейках")
    if np.any(np.sum(groups, axis=0) >= 1.0):
        raise DomainError("Сумма групповых долей должна быть меньше единицы")
    return np.array(
        [
            adjoint_delayed_fraction(phi, phi_star, groups[index], nu_sigma_f_per_m, dx_m)
            for index in range(groups.shape[0])
        ]
    )


def advance_fixed_flux_precursors(
    precursor: np.ndarray,
    flux: np.ndarray,
    beta: float,
    nu_sigma_f_per_m: np.ndarray,
    decay_per_s: float,
    dt_s: float,
) -> np.ndarray:
    """Предшественники при зафиксированном потоке: C → C_eq + (C-C_eq) exp(-λ t)."""
    concentration = np.asarray(precursor, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fission = np.asarray(nu_sigma_f_per_m, dtype=float)
    if decay_per_s <= 0.0 or dt_s < 0.0 or beta < 0.0 or beta >= 1.0:
        raise DomainError("Распад, шаг или доля предшественников заданы вне области")
    if concentration.shape != flux.shape or fission.shape != flux.shape:
        raise DomainError("Предшественники и поток заданы на разных ячейках")
    equilibrium = beta * fission * flux / decay_per_s
    return equilibrium + (concentration - equilibrium) * math.exp(-decay_per_s * dt_s)


def fission_operator(mesh: Mesh1D, nu_sigma_fission: np.ndarray) -> np.ndarray:
    return np.diag(nu_sigma_fission * mesh.dx)


def adjoint_k_eigenvalue(
    removal: np.ndarray, fission: np.ndarray, tol: float = 1e-10, max_iter: int = 200
) -> tuple:
    """Сопряжённая задача A^T phi* = (1/k) F^T phi*. То же k, что у прямой задачи."""
    return k_eigenvalue(
        np.array(removal, dtype=float).T, np.array(fission, dtype=float).T, tol, max_iter
    )


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


def modal_delayed_step(
    mesh: Mesh1D,
    diffusion_m: np.ndarray,
    sigma_a_per_m: np.ndarray,
    nu_sigma_f_per_m: np.ndarray,
    speed_m_s: float,
    beta: float,
    decay_per_s: float,
    flux: np.ndarray,
    precursor: np.ndarray,
    dt_s: float,
    left: str = "dirichlet0",
    right: str = "dirichlet0",
) -> tuple:
    """Шаг одногрупповой диффузии с одной группой запаздывающих нейтронов.

    Пространственный оператор тот же, что у стационарной k-задачи. На собственной
    моде амплитуда совпадает с точечной кинетикой при Lambda = 1/(v nuSigma_f).
    """
    from scipy.linalg import expm

    if speed_m_s <= 0.0 or decay_per_s <= 0.0:
        raise DomainError(
            "Скорость и постоянная распада модальной кинетики должны быть положительными"
        )
    removal = diffusion_operator(mesh, diffusion_m, sigma_a_per_m, left=left, right=right)
    n_cells = mesh.n_cells
    matrix = np.zeros((2 * n_cells, 2 * n_cells))
    for index in range(n_cells):
        matrix[index, :n_cells] = -speed_m_s / mesh.dx[index] * removal[index]
        matrix[index, index] += speed_m_s * (1.0 - beta) * nu_sigma_f_per_m[index]
        matrix[index, n_cells + index] = speed_m_s * decay_per_s
        matrix[n_cells + index, index] = beta * nu_sigma_f_per_m[index]
        matrix[n_cells + index, n_cells + index] = -decay_per_s
    state = expm(matrix * dt_s) @ np.concatenate([flux, precursor])
    return state[:n_cells], state[n_cells:]


def uniform_delayed_step(
    flux: float,
    precursor: float,
    sigma_a_per_m: float,
    nu_sigma_f_per_m: float,
    speed_m_s: float,
    beta: float,
    decay_per_s: float,
    dt_s: float,
) -> np.ndarray:
    """Однородная зона без утечки, одна группа запаздывающих нейтронов.

    Возвращает (phi, C). C связан с точечной кинетикой как C_hat = v C.
    """
    from scipy.linalg import expm

    if speed_m_s <= 0.0 or nu_sigma_f_per_m <= 0.0:
        raise DomainError("Скорость и nuSigma_f однородной кинетики должны быть положительными")
    matrix = np.array(
        [
            [
                speed_m_s * ((1.0 - beta) * nu_sigma_f_per_m - sigma_a_per_m),
                speed_m_s * decay_per_s,
            ],
            [beta * nu_sigma_f_per_m, -decay_per_s],
        ]
    )
    return expm(matrix * dt_s) @ np.array([flux, precursor], dtype=float)


def uniform_multigroup_delayed_step(
    flux: float,
    precursors: np.ndarray,
    sigma_a_per_m: float,
    nu_sigma_f_per_m: float,
    speed_m_s: float,
    beta: np.ndarray,
    decay_per_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Однородная зона без утечки и несколько групп запаздывающих нейтронов.

    Возвращает (phi, C). С точечной кинетикой связь C_hat = v C, Lambda = 1/(v nuSigma_f).
    """
    from scipy.linalg import expm

    delayed = np.asarray(beta, dtype=float)
    decay = np.asarray(decay_per_s, dtype=float)
    concentration = np.asarray(precursors, dtype=float)
    if delayed.ndim != 1 or delayed.shape != decay.shape or concentration.shape != delayed.shape:
        raise DomainError("Группы запаздывающих нейтронов заданы разной длины")
    if (
        speed_m_s <= 0.0
        or nu_sigma_f_per_m <= 0.0
        or dt_s < 0.0
        or np.any(delayed < 0.0)
        or np.any(decay <= 0.0)
        or float(np.sum(delayed)) >= 1.0
    ):
        raise DomainError("Многогрупповая однородная кинетика задана вне области")
    groups = delayed.size
    matrix = np.zeros((groups + 1, groups + 1))
    matrix[0, 0] = speed_m_s * ((1.0 - float(np.sum(delayed))) * nu_sigma_f_per_m - sigma_a_per_m)
    matrix[0, 1:] = speed_m_s * decay
    matrix[1:, 0] = delayed * nu_sigma_f_per_m
    matrix[1:, 1:] = np.diag(-decay)
    state = np.concatenate([[flux], concentration])
    return expm(matrix * dt_s) @ state


def two_region_reflector_k(
    diffusion_core_m: float,
    sigma_core_per_m: float,
    nu_sigma_f_per_m: float,
    diffusion_reflector_m: float,
    sigma_reflector_per_m: float,
    core_half_width_m: float,
    reflector_m: float,
) -> float:
    """k пластины с отражателем: ноль тока на оси, ноль потока на внешнем краю.

    На границе непрерывны поток и ток. Это одногрупповая задача, не многогрупповой ВВЭР.
    """
    if (
        min(
            diffusion_core_m,
            diffusion_reflector_m,
            sigma_core_per_m,
            sigma_reflector_per_m,
            nu_sigma_f_per_m,
            core_half_width_m,
            reflector_m,
        )
        <= 0.0
    ):
        raise DomainError("Параметры двухзонной пластины должны быть положительными")
    k_infinity = nu_sigma_f_per_m / sigma_core_per_m
    kappa = math.sqrt(sigma_reflector_per_m / diffusion_reflector_m)
    edge = math.pi / (2.0 * core_half_width_m)
    k_at_quarter_wave = nu_sigma_f_per_m / (sigma_core_per_m + diffusion_core_m * edge**2)

    def residual(k_value: float) -> float:
        buckling2 = (nu_sigma_f_per_m / k_value - sigma_core_per_m) / diffusion_core_m
        if buckling2 <= 0.0:
            return -1.0
        buckling = math.sqrt(buckling2)
        argument = buckling * core_half_width_m
        if argument >= 0.5 * math.pi:
            return 1.0
        left = diffusion_core_m * buckling * math.tan(argument)
        right = diffusion_reflector_m * kappa / math.tanh(kappa * reflector_m)
        return left - right

    from scipy.optimize import brentq

    low = min(k_infinity, k_at_quarter_wave) * 1.0000001
    high = k_infinity * 0.999999
    if low >= high or residual(low) * residual(high) > 0.0:
        raise DomainError("Корень k двухзонной пластины не локализован")
    return float(brentq(residual, low, high))


def analytical_slab_k(
    diffusion_m: float, sigma_a: float, nu_sigma_f: float, height_m: float
) -> float:
    """Однородная пластина с нулевым потоком на концах. B^2 = (pi/H)^2."""
    buckling = (np.pi / height_m) ** 2
    return nu_sigma_f / (sigma_a + diffusion_m * buckling)


def macroscopic_at_density(diffusion, sigma_a, nu_sigma_f, density_ratio: float):
    """Sigma ~ плотность, D ~ 1/плотность. Отношение nuSigma_f / Sigma_a не меняется."""
    if density_ratio <= 0.0:
        raise DomainError("Относительная плотность макросечений должна быть положительной")
    return (
        np.asarray(diffusion, dtype=float) / density_ratio,
        np.asarray(sigma_a, dtype=float) * density_ratio,
        np.asarray(nu_sigma_f, dtype=float) * density_ratio,
    )


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


def rod_absorption_staircase(
    base_sigma_a: np.ndarray, inserted_fraction: float, rod_delta_sigma: float, snap: str = "floor"
) -> np.ndarray:
    """Целые ячейки. Дробная ячейка не делится: floor оставляет её без поглотителя, ceil заполняет целиком."""
    if inserted_fraction < 0.0 or inserted_fraction > 1.0:
        raise DomainError("Доля ввода стержня должна быть в [0, 1]")
    if snap not in ("floor", "ceil"):
        raise DomainError("Скачок стержня задаётся как floor или ceil")
    sigma = np.array(base_sigma_a, dtype=float, copy=True)
    depth = inserted_fraction * sigma.size
    full = int(np.floor(depth))
    if snap == "ceil" and full < sigma.size and depth > full:
        full += 1
    if full > 0:
        sigma[:full] += rod_delta_sigma
    return sigma


def rod_absorption_2d(
    base_sigma_a: np.ndarray, inserted_fraction: float, rod_delta_sigma: float
) -> np.ndarray:
    """Ввод стержня вдоль первой оси. Граница делит строку, интеграл добавки точен."""
    if base_sigma_a.ndim != 2:
        raise DomainError("Двумерный стержень задаётся на поле ячеек")
    if inserted_fraction < 0.0 or inserted_fraction > 1.0:
        raise DomainError("Доля ввода стержня должна быть в [0, 1]")
    sigma = np.array(base_sigma_a, dtype=float, copy=True)
    rows = sigma.shape[0]
    depth = inserted_fraction * rows
    full = int(np.floor(depth))
    if full > 0:
        sigma[:full, :] += rod_delta_sigma
    if full < rows:
        sigma[full, :] += (depth - full) * rod_delta_sigma
    return sigma


def analytical_two_group_slab_k(
    diffusion_m: np.ndarray,
    sigma_a: np.ndarray,
    scatter_down: float,
    nu_sigma_f: np.ndarray,
    height_m: float,
) -> float:
    """Две группы, chi=(1, 0), рассеяние только вниз, нулевой поток на торцах пластины.

    Геометрический баклинг (pi/H)^2. Это не вакуумное условие Маршака.
    """
    buckling = (np.pi / height_m) ** 2
    removal_fast = sigma_a[0] + scatter_down + diffusion_m[0] * buckling
    removal_thermal = sigma_a[1] + diffusion_m[1] * buckling
    if removal_fast <= 0.0 or removal_thermal <= 0.0:
        raise DomainError("Съём двухгрупповой пластины неположителен")
    return float(
        nu_sigma_f[0] / removal_fast
        + nu_sigma_f[1] * scatter_down / (removal_fast * removal_thermal)
    )


def _as_cells(value, n_cells: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_cells, float(array))
    if array.shape == (n_cells,):
        return array
    raise DomainError("Поле ячеек должно быть скаляром или длиной сетки")


def two_group_slab_k(
    mesh: Mesh1D,
    diffusion_m: np.ndarray,
    sigma_a: np.ndarray,
    scatter_down,
    nu_sigma_f: np.ndarray,
    left: str = "dirichlet0",
    right: str = "dirichlet0",
    tol: float = 1e-10,
) -> tuple:
    """Стационарная k-задача двух групп. chi=(1, 0), обратного рассеяния нет."""
    n_cells = mesh.n_cells
    diffusion_m = np.asarray(diffusion_m, dtype=float)
    sigma_a = np.asarray(sigma_a, dtype=float)
    nu_sigma_f = np.asarray(nu_sigma_f, dtype=float)
    if diffusion_m.shape[0] != 2 or sigma_a.shape[0] != 2 or nu_sigma_f.shape[0] != 2:
        raise DomainError("Двухгрупповая задача ждёт по два значения D, Sigma_a и nuSigma_f")
    scatter = _as_cells(scatter_down, n_cells)
    fast_d = _as_cells(diffusion_m[0], n_cells)
    thermal_d = _as_cells(diffusion_m[1], n_cells)
    fast_a = _as_cells(sigma_a[0], n_cells)
    thermal_a = _as_cells(sigma_a[1], n_cells)
    fast_f = _as_cells(nu_sigma_f[0], n_cells)
    thermal_f = _as_cells(nu_sigma_f[1], n_cells)
    removal = np.zeros((2 * n_cells, 2 * n_cells))
    fission = np.zeros((2 * n_cells, 2 * n_cells))
    removal[:n_cells, :n_cells] = diffusion_operator(
        mesh, fast_d, fast_a + scatter, left=left, right=right
    )
    removal[n_cells:, n_cells:] = diffusion_operator(
        mesh, thermal_d, thermal_a, left=left, right=right
    )
    for index, dx in enumerate(mesh.dx):
        removal[n_cells + index, index] -= scatter[index] * dx
        fission[index, index] = fast_f[index] * dx
        fission[index, n_cells + index] = thermal_f[index] * dx
    return k_eigenvalue(removal, fission, tol=tol)


def two_group_critical_buckling(
    sigma_a: np.ndarray, scatter_12: float, nu_sigma_f: np.ndarray, diffusion_m: np.ndarray
) -> float:
    """Положительный B², при котором двухгрупповой k с chi=(1, 0) равен единице.

    Обратного рассеяния нет. Подкритическая бесконечная среда корня не имеет.
    """
    absorption = np.asarray(sigma_a, dtype=float)
    fission = np.asarray(nu_sigma_f, dtype=float)
    diffusion = np.asarray(diffusion_m, dtype=float)
    if (
        scatter_12 < 0.0
        or absorption.shape != (2,)
        or fission.shape != (2,)
        or diffusion.shape != (2,)
        or np.any(absorption <= 0.0)
        or np.any(diffusion <= 0.0)
        or np.any(fission < 0.0)
    ):
        raise DomainError("Критический баклинг двух групп задан вне области")
    fast_removal = absorption[0] + scatter_12
    if fast_removal <= 0.0:
        raise DomainError("Съём быстрой группы неположителен")
    infinite = fission[0] / fast_removal + fission[1] * scatter_12 / (fast_removal * absorption[1])
    if infinite <= 1.0:
        raise DomainError("Двухгрупповая бесконечная среда не надкритична")
    quadratic = diffusion[0] * diffusion[1]
    linear = fast_removal * diffusion[1] + absorption[1] * diffusion[0] - fission[0] * diffusion[1]
    constant = fast_removal * absorption[1] - fission[0] * absorption[1] - fission[1] * scatter_12
    discriminant = linear**2 - 4.0 * quadratic * constant
    if discriminant < 0.0:
        raise DomainError("Дискриминант двухгруппового баклинга отрицателен")
    buckling = (-linear + math.sqrt(discriminant)) / (2.0 * quadratic)
    if buckling <= 0.0:
        raise DomainError("Положительный критический баклинг двух групп не найден")
    return float(buckling)


def two_group_buckling_k(
    sigma_a: np.ndarray,
    scatter_12: float,
    nu_sigma_f: np.ndarray,
    chi: np.ndarray,
    diffusion_m: np.ndarray,
    buckling2: float,
) -> float:
    """k_inf с утечкой D B², добавленной к съёму каждой группы. Больший баклинг снижает k."""
    if buckling2 < 0.0:
        raise DomainError("Баклинг двух групп отрицателен")
    leaked = np.asarray(sigma_a, dtype=float) + np.asarray(diffusion_m, dtype=float) * buckling2
    return two_group_infinite_k(leaked, scatter_12, nu_sigma_f, chi)


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
