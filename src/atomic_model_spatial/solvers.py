"""Residual-certified sparse fixed-source and equilibrium k solves."""

import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, splu

from .model import array


def factor(matrix):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            return splu(matrix.tocsc())
    except (RuntimeError, MatrixRankWarning) as exc:
        raise ValueError("Singular diffusion operator") from exc


def relative_residual(matrix, x, rhs):
    denominator = max(float(np.max(abs(matrix) @ abs(x) + abs(rhs))), 1e-300)
    return float(np.max(abs(matrix @ x - rhs)) / denominator)


def eigenvalue(model, tolerance=1e-10, max_iterations=10000):
    if not np.isfinite(tolerance) or not 1e-13 <= tolerance <= 1e-6:
        raise ValueError("Invalid eigen tolerance")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 100000:
        raise ValueError("Invalid eigen iteration budget")
    if not np.any(model.nu_fission_per_m):
        raise ValueError("No fission eigenproblem")
    solve = factor(model.loss).solve
    flux = np.ones(model.g * model.n)
    flux /= float(np.sum(model.fission @ flux))
    for iteration in range(1, max_iterations + 1):
        raw = solve(model.fission @ flux)
        k = float(np.sum(model.fission @ raw))
        if not np.isfinite(k) or k <= 0 or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Invalid positive eigenpair")
        new = raw / k
        residual = relative_residual(model.loss, new, model.fission @ new / k)
        change = float(np.max(abs(new - flux)) / max(np.max(abs(new)), 1e-300))
        flux = new
        if residual <= tolerance and change <= tolerance:
            return dict(
                k=k,
                flux=flux.reshape(model.g, model.n).tolist(),
                relative_residual=residual,
                iterations=iteration,
                normalization="volume-integrated fission production = 1",
            )
    raise ValueError("Eigen iteration budget exhausted; no accepted result")


def fixed_source(model, source):
    q = array(source, (model.g, model.n), "source")
    if np.any(model.nu_fission_per_m) and eigenvalue(model)["k"] >= 1 - 1e-8:
        raise ValueError("Fixed source requires strictly subcritical multiplication")
    matrix = model.loss - model.fission
    rhs = (q * model.width).ravel()
    flux = factor(matrix).solve(rhs)
    residual = relative_residual(matrix, flux, rhs)
    if not np.isfinite(flux).all() or np.any(flux < 0) or residual > 1e-10:
        raise ValueError("Invalid fixed-source solution")
    phi = flux.reshape(model.g, model.n)
    production = float(np.sum(model.nu_fission_per_m * phi * model.width))
    absorption = float(np.sum(model.absorption_per_m * phi * model.width))
    leakage = float(np.sum(model.leakage * phi))
    external = float(np.sum(q * model.width))
    defect = production + external - absorption - leakage
    scale = max(production + external + absorption + leakage, 1e-300)
    if abs(defect) / scale > 1e-9:
        raise ValueError("Fixed-source global balance failed")
    return dict(
        flux=phi.tolist(),
        relative_residual=residual,
        fission_rate=production,
        absorption_rate=absorption,
        outward_leakage_rate=leakage,
        external_rate=external,
        relative_balance_defect=abs(defect) / scale,
    )
