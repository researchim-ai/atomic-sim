"""Двух- и трёхмерная диффузия и одномерный дискретный транспорт S2."""

from __future__ import annotations

import math

import numpy as np

from core.status import DomainError, ModelPassport, ModelStatus

ND_PASSPORT = ModelPassport(
    model_id="fv-diffusion-nd",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Многомерная одногрупповая диффузия на прямоугольной сетке",
    approximations=("постоянный D", "нулевой поток на внешней грани", "однородная среда"),
    applicability="Прямоугольный параллелепипед",
    data_sources=("Аналитический баклинг (pi/L)^2 по каждой оси",),
)

TRANSPORT_PASSPORT = ModelPassport(
    model_id="sn-s2-1d",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Одномерный дискретный ординат S2 с изотропным источником",
    approximations=("две дискретные ординаты Гаусса", "изотропное рассеяние не отделено от источника"),
    applicability="Плоский слой; вакуум или отражение на гранях",
    data_sources=("Дискретная ордината S2, веса 1 и 1, mu = ±1/sqrt(3)",),
)


def analytical_rectangle_k(diffusion_m: float, sigma_a: float, nu_sigma_f: float, lengths_m: tuple) -> float:
    buckling = sum((math.pi / length) ** 2 for length in lengths_m)
    return nu_sigma_f / (sigma_a + diffusion_m * buckling)


def _laplacian_1d(n_cells: int, dx: float, diffusion: float) -> np.ndarray:
    matrix = np.zeros((n_cells, n_cells))
    face = diffusion / dx
    boundary = diffusion / (0.5 * dx)
    for i in range(n_cells):
        matrix[i, i] += sigma_placeholder(face, boundary, i, n_cells)
        if i > 0:
            matrix[i, i - 1] -= face
        if i + 1 < n_cells:
            matrix[i, i + 1] -= face
    return matrix


def sigma_placeholder(face: float, boundary: float, index: int, n_cells: int) -> float:
    value = 0.0
    value += boundary if index == 0 else face
    value += boundary if index == n_cells - 1 else face
    return value


def diffusion_k_nd(shape: tuple, lengths_m: tuple, diffusion_m: float, sigma_a: float, nu_sigma_f: float, tol: float = 1e-9) -> tuple:
    """k-задача Кронecker-суммой одномерных операторов утечки."""
    if len(shape) != len(lengths_m) or len(shape) not in (2, 3):
        raise DomainError("Сетка диффузии поддерживает 2 или 3 измерения")
    operators = []
    for n_cells, length in zip(shape, lengths_m):
        dx = length / n_cells
        operators.append(_laplacian_1d(n_cells, dx, diffusion_m) / dx)
    # Уравнение (L + Sa) phi = nuSf / k phi. L уже делён на dx, то есть на объём ячейки в 1D.
    # Для тензорного произведения однородный источник деления не зависит от объёма, если L — оператор на значение.
    size = int(np.prod(shape))
    removal = np.zeros((size, size))
    identity_factors = []
    for axis, operator in enumerate(operators):
        eye = [np.eye(n) for n in shape]
        eye[axis] = operator
        removal += _kron(eye)
        identity_factors.append(np.eye(shape[axis]))
    removal += sigma_a * np.eye(size)
    fission = nu_sigma_f * np.eye(size)
    phi = np.ones(size)
    k_value = 1.0
    converged = False
    for _ in range(80):
        source = fission @ phi
        phi_new = np.linalg.solve(removal, source)
        k_new = float(np.sum(fission @ phi_new) / np.sum(source))
        phi_new /= np.max(np.abs(phi_new))
        if abs(k_new - k_value) < tol:
            converged = True
            k_value = k_new
            phi = phi_new
            break
        k_value = k_new
        phi = phi_new
    return k_value, phi, converged


def _kron(factors: list) -> np.ndarray:
    result = np.array([[1.0]])
    for factor in factors:
        result = np.kron(result, factor)
    return result


MU_S2 = 1.0 / math.sqrt(3.0)


def sn_infinite_medium(sigma_per_m: float, isotropic_source: float) -> float:
    """Угловой поток в бесконечной среде: Sigma psi = S/2."""
    if sigma_per_m <= 0.0:
        raise DomainError("Сечение съёма транспорта должно быть положительным")
    return isotropic_source / (2.0 * sigma_per_m)


def _sweep_direction(
    psi_size: int,
    dx: float,
    mu: float,
    sigma_per_m: float,
    source_half: float,
    incoming: float,
    forward: bool,
) -> np.ndarray:
    psi = np.zeros(psi_size)
    indices = range(psi_size) if forward else range(psi_size - 1, -1, -1)
    denom = mu / dx + sigma_per_m
    for index in indices:
        psi[index] = (mu / dx * incoming + source_half) / denom
        incoming = psi[index]
    return psi


def sn_s2_slab(
    width_m: float,
    n_cells: int,
    sigma_per_m: float,
    isotropic_source: float,
    left: str = "vacuum",
    right: str = "vacuum",
    iterations: int = 200,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Скалярный поток S2. Отражающие границы итерируются до сходимости токов."""
    if left not in ("vacuum", "reflective") or right not in ("vacuum", "reflective"):
        raise DomainError("Граничное условие транспорта не поддержано")
    dx = width_m / n_cells
    mu = MU_S2
    incoming_left = 0.0
    incoming_right = 0.0
    psi_plus = np.zeros(n_cells)
    psi_minus = np.zeros(n_cells)
    for _ in range(iterations):
        half = isotropic_source / 2.0
        psi_plus = _sweep_direction(n_cells, dx, mu, sigma_per_m, half, incoming_left, True)
        right_in = psi_plus[-1] if right == "reflective" else 0.0
        psi_minus = _sweep_direction(n_cells, dx, mu, sigma_per_m, half, right_in, False)
        left_in = psi_minus[0] if left == "reflective" else 0.0
        if abs(left_in - incoming_left) + abs(right_in - incoming_right) < tolerance:
            incoming_left = left_in
            incoming_right = right_in
            break
        incoming_left = left_in
        incoming_right = right_in
    else:
        raise DomainError("Токи на отражающих границах S2 не сошлись")
    return psi_plus + psi_minus


def sn_particle_balance(width_m: float, n_cells: int, sigma_per_m: float, isotropic_source: float) -> tuple:
    """Поглощение, утечка и источник для вакуумного слоя. Их сумма совпадает."""
    dx = width_m / n_cells
    mu = MU_S2
    psi_plus = np.zeros(n_cells)
    incoming = 0.0
    for i in range(n_cells):
        denom = mu / dx + sigma_per_m
        psi_plus[i] = (mu / dx * incoming + isotropic_source / 2.0) / denom
        incoming = psi_plus[i]
    psi_minus = np.zeros(n_cells)
    incoming = 0.0
    for i in range(n_cells - 1, -1, -1):
        denom = mu / dx + sigma_per_m
        psi_minus[i] = (mu / dx * incoming + isotropic_source / 2.0) / denom
        incoming = psi_minus[i]
    phi = psi_plus + psi_minus
    absorption = float(sigma_per_m * np.sum(phi) * dx)
    leakage = float(mu * psi_plus[-1] + mu * psi_minus[0])
    source = isotropic_source * width_m
    return absorption, leakage, source
