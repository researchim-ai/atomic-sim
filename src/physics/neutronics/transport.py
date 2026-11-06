"""Двух- и трёхмерная диффузия и одномерный дискретный транспорт S2."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core.status import DomainError, ModelPassport, ModelStatus

ND_PASSPORT = ModelPassport(
    model_id="fv-diffusion-nd",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Многомерная одногрупповая диффузия на прямоугольной сетке",
    approximations=(
        "нулевой поток на внешней грани, если грань не объявлена отражающей",
        "отражающая грань ставится только на нижней стороне оси и даёт нулевой ток",
        "разреженная сборка совпадает с плотной на однородной среде",
        "D и Sigma могут задаваться по ячейкам",
    ),
    applicability="Прямоугольный параллелепипед",
    data_sources=("Аналитический баклинг (pi/L)^2 по каждой оси",),
)

TRANSPORT_PASSPORT = ModelPassport(
    model_id="sn-s2-1d",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Одномерный дискретный ординат S2 с изотропным источником",
    approximations=(
        "две дискретные ординаты Гаусса",
        "изотропное рассеяние не отделено от источника",
    ),
    applicability="Плоский слой; вакуум или отражение на гранях",
    data_sources=("Дискретная ордината S2, веса 1 и 1, mu = ±1/sqrt(3)",),
)


def analytical_reflective_k(
    diffusion_m: float,
    sigma_a: float,
    nu_sigma_f: float,
    lengths_m: tuple,
    lower_reflective: tuple,
) -> float:
    """Баклинг (pi/(2L))^2 на отражающей нижней грани и (pi/L)^2 на паре нулевых потоков."""
    if len(lengths_m) != len(lower_reflective) or len(lengths_m) not in (2, 3):
        raise DomainError("Отражающий баклинг задан для двух или трёх длин")
    if min(diffusion_m, sigma_a, nu_sigma_f) <= 0.0 or any(length <= 0.0 for length in lengths_m):
        raise DomainError("Параметры отражающей пластины должны быть положительными")
    buckling = 0.0
    for length, reflected in zip(lengths_m, lower_reflective):
        half = 0.5 if reflected else 1.0
        buckling += (math.pi * half / length) ** 2
    return nu_sigma_f / (sigma_a + diffusion_m * buckling)


def recover_manufactured_flux(shape: tuple, lengths_m: tuple, diffusion, sigma_a, phi: np.ndarray):
    """Решение A psi = A phi обязано вернуть заданный поток. Это проверка дискретного оператора."""
    matrix, _fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, 1.0)
    flux = np.asarray(phi, dtype=float).reshape(-1)
    if flux.size != matrix.shape[0]:
        raise DomainError("Изготовленный поток не совпадает с сеткой")
    solved = spla.spsolve(matrix, matrix @ flux)
    return np.asarray(solved, dtype=float).reshape(shape)


def analytical_rectangle_k(
    diffusion_m: float, sigma_a: float, nu_sigma_f: float, lengths_m: tuple
) -> float:
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


def diffusion_k_nd(
    shape: tuple,
    lengths_m: tuple,
    diffusion_m: float,
    sigma_a: float,
    nu_sigma_f: float,
    tol: float = 1e-9,
) -> tuple:
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


def _sparse_system(
    shape: tuple, lengths_m: tuple, diffusion, sigma_a, nu_sigma_f, lower_reflective=None
):
    if len(shape) != len(lengths_m) or len(shape) not in (2, 3):
        raise DomainError("Разреженная диффузия поддерживает 2 или 3 измерения")
    dims = [int(n) for n in shape]
    if any(n < 2 for n in dims) or any(length <= 0.0 for length in lengths_m):
        raise DomainError("В каждом измерении нужно не меньше двух ячеек и положительная длина")
    diffusion_grid = np.broadcast_to(np.asarray(diffusion, dtype=float), dims)
    removal_grid = np.broadcast_to(np.asarray(sigma_a, dtype=float), dims)
    fission_grid = np.broadcast_to(np.asarray(nu_sigma_f, dtype=float), dims)
    if np.any(diffusion_grid <= 0.0) or np.any(removal_grid <= 0.0):
        raise DomainError("D и сечение съёма в разреженной диффузии должны быть положительными")
    if lower_reflective is None:
        reflective = (False,) * len(dims)
    else:
        reflective = tuple(bool(flag) for flag in lower_reflective)
    if len(reflective) != len(dims):
        raise DomainError("Отражение задаётся по одному флагу на ось")
    steps = [length / n for length, n in zip(lengths_m, dims)]
    stride = [1] * len(dims)
    for axis in range(len(dims) - 2, -1, -1):
        stride[axis] = stride[axis + 1] * dims[axis + 1]
    size = int(np.prod(dims))

    def flat(coords: tuple) -> int:
        return sum(coord * stride[axis] for axis, coord in enumerate(coords))

    rows: list = []
    cols: list = []
    values: list = []
    fission = np.zeros(size)
    for coords in product(*[range(n) for n in dims]):
        point = flat(coords)
        diagonal = float(removal_grid[coords])
        fission[point] = float(fission_grid[coords])
        for axis, count in enumerate(dims):
            step = steps[axis]
            here = float(diffusion_grid[coords])
            index = coords[axis]
            if index == 0:
                if not reflective[axis]:
                    diagonal += 2.0 * here / step**2
            else:
                neighbor = list(coords)
                neighbor[axis] = index - 1
                other = float(diffusion_grid[tuple(neighbor)])
                couple = 2.0 * here * other / ((here + other) * step**2)
                diagonal += couple
                rows.append(point)
                cols.append(flat(tuple(neighbor)))
                values.append(-couple)
            if index == count - 1:
                diagonal += 2.0 * here / step**2
            else:
                neighbor = list(coords)
                neighbor[axis] = index + 1
                other = float(diffusion_grid[tuple(neighbor)])
                couple = 2.0 * here * other / ((here + other) * step**2)
                diagonal += couple
                rows.append(point)
                cols.append(flat(tuple(neighbor)))
                values.append(-couple)
        rows.append(point)
        cols.append(point)
        values.append(diagonal)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(size, size))
    return matrix, fission


def axis_strides(shape: tuple) -> tuple:
    """Шаг плоского индекса по каждой оси. Последняя ось имеет шаг 1."""
    dims = [int(n) for n in shape]
    if any(n < 1 for n in dims):
        raise DomainError("Шаг индекса задан для положительных размеров")
    stride = [1] * len(dims)
    for axis in range(len(dims) - 2, -1, -1):
        stride[axis] = stride[axis + 1] * dims[axis + 1]
    return tuple(stride)


def couplings_are_orthogonal(matrix, shape: tuple) -> bool:
    """Вне диагонали разреженная диффузия связывает только соседей по осям, не по диагонали ячейки."""
    strides = set(axis_strides(shape))
    coo = matrix.tocoo()
    for row, col, value in zip(coo.row, coo.col, coo.data):
        if row == col or value == 0.0:
            continue
        if abs(int(row) - int(col)) not in strides:
            return False
    return True


class FactorizedDiffusion:
    """Одна факторизация разреженного оператора на несколько источников."""

    def __init__(self, shape: tuple, lengths_m: tuple, diffusion, sigma_a):
        matrix, _fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, 1.0)
        self.shape = tuple(shape)
        self._factor = spla.splu(matrix.tocsc())
        self.factorizations = 1
        self.solves = 0

    def solve(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=float).reshape(-1)
        if values.size != int(np.prod(self.shape)):
            raise DomainError("Источник не совпадает с факторизованной сеткой")
        self.solves += 1
        return np.asarray(self._factor.solve(values), dtype=float).reshape(self.shape)


def cmfd_ratio(coarse_flux: np.ndarray, reference_flux: np.ndarray) -> np.ndarray:
    """Поправка coarse/reference. Нулевой coarse-поток и отсутствующий референс не заменяются."""
    coarse = np.asarray(coarse_flux, dtype=float)
    reference = np.asarray(reference_flux, dtype=float)
    if coarse.shape != reference.shape or coarse.size == 0:
        raise DomainError("CMFD требует одинаковые непустые поля")
    if np.any(coarse <= 0.0) or np.any(reference < 0.0):
        raise DomainError("CMFD не делит на неположительный coarse-поток")
    return reference / coarse


def apply_cmfd(coarse_flux: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    coarse = np.asarray(coarse_flux, dtype=float)
    factor = np.asarray(ratio, dtype=float)
    if coarse.shape != factor.shape:
        raise DomainError("Поправка CMFD не совпадает с полем")
    return coarse * factor


def diffusion_k_sparse(
    shape: tuple,
    lengths_m: tuple,
    diffusion,
    sigma_a,
    nu_sigma_f,
    tol: float = 1e-9,
    lower_reflective=None,
) -> tuple:
    """Разреженная k-задача. На однородной среде k совпадает с плотной сборкой."""
    matrix, fission = _sparse_system(
        shape, lengths_m, diffusion, sigma_a, nu_sigma_f, lower_reflective
    )
    solver = spla.splu(matrix.tocsc())
    phi = np.ones(fission.size)
    k_value = 1.0
    converged = False
    for _ in range(80):
        source = fission * phi
        source_sum = float(np.sum(source))
        if source_sum <= 0.0:
            raise DomainError("Источник деления нулевой, k-задача не определена")
        phi_new = solver.solve(source)
        k_new = float(np.sum(fission * phi_new) / source_sum)
        phi_new /= np.max(np.abs(phi_new))
        if abs(k_new - k_value) < tol:
            return k_new, phi_new, True
        k_value = k_new
        phi = phi_new
    return k_value, phi, converged


def operator_leakage(
    shape: tuple, lengths_m: tuple, diffusion, sigma_a, phi: np.ndarray
) -> np.ndarray:
    """Утечка ячеек: оператор без сечения съёма."""
    matrix, _fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, 0.0)
    removal = np.broadcast_to(np.asarray(sigma_a, dtype=float), shape).reshape(-1)
    flux = np.asarray(phi, dtype=float).reshape(-1)
    return matrix @ flux - removal * flux


def face_currents_2d(shape: tuple, lengths_m: tuple, diffusion, phi: np.ndarray) -> tuple:
    """Ток через внутреннюю грань. Положительный ток выходит из клетки с меньшим индексом."""
    if len(shape) != 2 or len(lengths_m) != 2:
        raise DomainError("Токи граней заданы для двумерной сетки")
    flux = np.asarray(phi, dtype=float).reshape(shape)
    conductivity = np.broadcast_to(np.asarray(diffusion, dtype=float), shape)
    nx, ny = shape
    dx = lengths_m[0] / nx
    dy = lengths_m[1] / ny
    current_x = np.zeros((nx - 1, ny))
    current_y = np.zeros((nx, ny - 1))
    for i in range(nx - 1):
        for j in range(ny):
            left = float(conductivity[i, j])
            right = float(conductivity[i + 1, j])
            couple = 2.0 * left * right / ((left + right) * dx**2)
            current_x[i, j] = couple * (flux[i, j] - flux[i + 1, j])
    for i in range(nx):
        for j in range(ny - 1):
            bottom = float(conductivity[i, j])
            top = float(conductivity[i, j + 1])
            couple = 2.0 * bottom * top / ((bottom + top) * dy**2)
            current_y[i, j] = couple * (flux[i, j] - flux[i, j + 1])
    return current_x, current_y


def divergence_nd(shape: tuple, lengths_m: tuple, diffusion, phi: np.ndarray) -> np.ndarray:
    """Дивергенция токов в 2D и 3D. Внутренние грани сокращаются, внешняя грань — нулевой поток."""
    if len(shape) not in (2, 3) or len(shape) != len(lengths_m):
        raise DomainError("Дивергенция токов задана для двух или трёх измерений")
    flux = np.asarray(phi, dtype=float).reshape(shape)
    conductivity = np.broadcast_to(np.asarray(diffusion, dtype=float), shape)
    if np.any(conductivity <= 0.0):
        raise DomainError("Коэффициент диффузии для токов должен быть положительным")
    steps = [length / count for length, count in zip(lengths_m, shape)]
    divergence = np.zeros(shape, dtype=float)
    for axis, step in enumerate(steps):
        for index in range(shape[axis] - 1):
            left = [slice(None)] * len(shape)
            right = [slice(None)] * len(shape)
            left[axis] = index
            right[axis] = index + 1
            left_t = tuple(left)
            right_t = tuple(right)
            here = conductivity[left_t]
            other = conductivity[right_t]
            couple = 2.0 * here * other / ((here + other) * step**2)
            current = couple * (flux[left_t] - flux[right_t])
            divergence[left_t] += current
            divergence[right_t] -= current
        for edge in (0, shape[axis] - 1):
            sl = [slice(None)] * len(shape)
            sl[axis] = edge
            key = tuple(sl)
            divergence[key] += 2.0 * conductivity[key] / step**2 * flux[key]
    return divergence


def divergence_from_currents(
    shape: tuple, lengths_m: tuple, diffusion, phi: np.ndarray
) -> np.ndarray:
    """Дивергенция токов плюс утечка в нулевой поток на внешней грани."""
    current_x, current_y = face_currents_2d(shape, lengths_m, diffusion, phi)
    flux = np.asarray(phi, dtype=float).reshape(shape)
    conductivity = np.broadcast_to(np.asarray(diffusion, dtype=float), shape)
    dx = lengths_m[0] / shape[0]
    dy = lengths_m[1] / shape[1]
    divergence = np.zeros(shape)
    divergence[:-1, :] += current_x
    divergence[1:, :] -= current_x
    divergence[:, :-1] += current_y
    divergence[:, 1:] -= current_y
    divergence[0, :] += 2.0 * conductivity[0, :] / dx**2 * flux[0, :]
    divergence[-1, :] += 2.0 * conductivity[-1, :] / dx**2 * flux[-1, :]
    divergence[:, 0] += 2.0 * conductivity[:, 0] / dy**2 * flux[:, 0]
    divergence[:, -1] += 2.0 * conductivity[:, -1] / dy**2 * flux[:, -1]
    return divergence


def sparse_fixed_source(
    shape: tuple, lengths_m: tuple, diffusion, sigma_a, source: np.ndarray
) -> np.ndarray:
    """Решение (утечка + съём) phi = источник."""
    matrix, _fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, 0.0)
    return spla.spsolve(matrix, np.asarray(source, dtype=float))


def rayleigh_k(
    shape: tuple, lengths_m: tuple, diffusion, sigma_a, nu_sigma_f, phi: np.ndarray
) -> float:
    """k = <phi, F phi> / <phi, A phi> на заданном векторе."""
    matrix, fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, nu_sigma_f)
    flux = np.asarray(phi, dtype=float).reshape(-1)
    denominator = float(flux @ (matrix @ flux))
    numerator = float(flux @ (fission * flux))
    if denominator <= 0.0 or numerator <= 0.0:
        raise DomainError("Рэлеевский частный не определён")
    return numerator / denominator


def require_trained_mode(mode: np.ndarray, vector: np.ndarray, tolerance: float) -> float:
    """Доля вектора вне одной обученной моды. Выше допуска модель не применяется."""
    if tolerance < 0.0:
        raise DomainError("Допуск выхода из моды отрицателен")
    direction = np.asarray(mode, dtype=float).reshape(-1)
    sample = np.asarray(vector, dtype=float).reshape(-1)
    direction = direction / np.linalg.norm(direction)
    sample_norm = float(np.linalg.norm(sample))
    if sample_norm == 0.0:
        raise DomainError("Нулевой вектор не лежит в обученной моде")
    coefficient = float(np.dot(direction, sample))
    error = float(np.linalg.norm(sample - coefficient * direction) / sample_norm)
    if error > tolerance:
        raise DomainError("Вектор вне обученной моды")
    return error


def leakage_balance(
    shape: tuple, lengths_m: tuple, diffusion, sigma_a, nu_sigma_f, k_value: float, phi: np.ndarray
) -> tuple:
    """Сумма (утечка + съём) против источника деления, делённого на k."""
    matrix, fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, nu_sigma_f)
    removal = matrix @ np.asarray(phi, dtype=float)
    source = fission * np.asarray(phi, dtype=float) / k_value
    return float(np.sum(removal)), float(np.sum(source))


def modal_delayed_step_nd(
    shape: tuple,
    lengths_m: tuple,
    diffusion,
    sigma_a,
    nu_sigma_f,
    speed_m_s: float,
    beta: float,
    decay_per_s: float,
    flux: np.ndarray,
    precursor: np.ndarray,
    dt_s: float,
) -> tuple:
    """Одна группа запаздывающих нейтронов на декартовой сетке. Утечка берётся из разреженного оператора."""
    from scipy.linalg import expm

    if speed_m_s <= 0.0 or decay_per_s <= 0.0 or dt_s < 0.0:
        raise DomainError("Скорость, распад и шаг модальной кинетики заданы неверно")
    removal, fission = _sparse_system(shape, lengths_m, diffusion, sigma_a, nu_sigma_f)
    dense = removal.toarray()
    flux = np.asarray(flux, dtype=float)
    precursor = np.asarray(precursor, dtype=float)
    size = flux.size
    matrix = np.zeros((2 * size, 2 * size))
    matrix[:size, :size] = -speed_m_s * dense
    for index in range(size):
        matrix[index, index] += speed_m_s * (1.0 - beta) * fission[index]
        matrix[index, size + index] = speed_m_s * decay_per_s
        matrix[size + index, index] = beta * fission[index]
        matrix[size + index, size + index] = -decay_per_s
    state = expm(matrix * dt_s) @ np.concatenate([flux, precursor])
    return state[:size], state[size:]


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


def sn_particle_balance(
    width_m: float, n_cells: int, sigma_per_m: float, isotropic_source: float
) -> tuple:
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
