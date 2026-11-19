"""Weighted least-change initialization with wall, divergence and flux constraints.

This discrete projection is an opt-in change to the initial data. It does not
establish continuum axis regularity, pressure compatibility or DNS validity.
"""
import numpy as np
from scipy.linalg import svd
from atomic_model_station_spectral.modes import derivative_matrices, layout, divergence


def constraint_matrix(radius, weights, k, m, alpha, target_bulk):
    """Independent unknowns exclude the three fixed zero wall values."""
    r = np.asarray(radius); w = np.asarray(weights); n = len(r)
    positive, ghost = derivative_matrices(r)
    radial = (positive + (-1 if (m + 1) % 2 else 1) * ghost).toarray()
    radial += np.diag(1 / r)
    eye = np.eye(n)[:, :-1]
    matrix = np.hstack((radial[:, :-1], 1j * m * eye / r[:, None],
                        1j * alpha * k * eye))
    rhs = np.zeros(n, complex)
    if k == 0 and m == 0:
        flux = np.r_[np.zeros(2 * (n - 1)), 2 * w[:-1]]
        matrix = np.vstack((matrix, flux))
        rhs = np.r_[rhs, target_bulk - 2 * w @ (1 - r * r)]
    return matrix, rhs


def project_velocity(ur, ut, uz, radius, weights, k_count, m_count, *,
                     alpha, maximum_relative_correction, target_bulk=.5, mp=1):
    """Return separate projected arrays and a receipt, without changing inputs.

    Minimize sum(w*|delta_u|²) per mode, subject to discrete div=0,
    no-slip and prescribed mean bulk velocity. SVD handles the redundant
    mean-mode continuity row. Large corrections are rejected before return.
    """
    r = np.asarray(radius, dtype=float); w = np.asarray(weights, dtype=float)
    # Validate before constructing any mode-sized allocation.
    derivative_matrices(r)
    if len(r) > 128 or w.shape != r.shape or not np.all(np.isfinite(w)) or np.any(w <= 0) or abs(w.sum() - .5) > 1e-12:
        raise ValueError('Positive normalized quadrature on at most128 radial nodes required')
    for value in (alpha, maximum_relative_correction, target_bulk):
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError('Finite real initialization policy required')
    if alpha <= 0 or not 0 < maximum_relative_correction <= .1 or target_bulk <= 0:
        raise ValueError('Positive alpha/bulk and explicit correction budget in(0,0.1] required')
    k, m, factors = layout(k_count, m_count, mp)
    inputs = [np.asarray(v) for v in (ur, ut, uz)]
    if any(v.shape != (len(k), len(r)) or not np.iscomplexobj(v) or not np.all(np.isfinite(v)) for v in inputs):
        raise ValueError('Complete finite complex velocity fields required')
    if any(np.max(abs(v[0].imag)) > 1e-12 for v in inputs):
        raise ValueError('Real mean Fourier mode required')
    output = [np.zeros_like(v, dtype=complex) for v in inputs]
    sqrt_w = np.sqrt(np.tile(w[:-1], 3)); maximum_constraint = 0.; ranks = []
    maximum_stationarity = 0.
    for index, (axial, azimuthal) in enumerate(zip(k, m)):
        a, b = constraint_matrix(r, w, int(axial), int(azimuthal), alpha, target_bulk)
        scaled = a / sqrt_w
        left, singular, right = svd(scaled, full_matrices=False, check_finite=False)
        keep = singular > singular[0] * 1e-12
        x = np.concatenate([v[index, :-1] for v in inputs])
        residual = a @ x - b
        multiplier = left[:, keep].conj().T @ residual / singular[keep]
        correction = right[keep].conj().T @ multiplier
        candidate = x - correction / sqrt_w
        error = float(np.max(abs(a @ candidate - b)))
        if error > 1e-9:
            raise ValueError('Discrete initialization constraints not resolved')
        # Orthogonality of the weighted correction to the constraint nullspace.
        projected = right[keep].conj().T @ (right[keep] @ correction)
        maximum_stationarity = max(maximum_stationarity, float(np.max(abs(correction - projected))))
        maximum_constraint = max(maximum_constraint, error); ranks.append(int(sum(keep)))
        for component, values in zip(output, np.split(candidate, 3)):
            component[index, :-1] = values
    def norm(fields):
        return float(2 * sum(np.sum((abs(v) ** 2) @ w * factors) for v in fields))
    delta = [new - old for new, old in zip(output, inputs)]
    correction_norm = np.sqrt(norm(delta))
    physical = [v.copy() for v in inputs]
    physical[2][0] += 1 - r * r
    original_norm = np.sqrt(norm(physical))
    relative = float(correction_norm / max(original_norm, 1e-30))
    if relative > maximum_relative_correction:
        raise ValueError('Initial-state change exceeds explicit correction budget')
    def div_norm(fields):
        d = divergence(*fields, r, k, m, alpha)
        return float(np.sqrt(2 * np.sum((abs(d) ** 2) @ w * factors)))
    receipt = dict(policy='explicit-weighted-discrete-velocity-initialization/v1',
        modes=len(k), radial_nodes=len(r), relative_correction=relative,
        maximum_relative_correction=maximum_relative_correction,
        correction_rms=float(correction_norm), velocity_rms_before=float(original_norm),
        correction_normalization='RMS of full velocity including the axial base profile',
        divergence_l2_before=div_norm(inputs), divergence_l2_after=div_norm(output),
        wall_maximum_after=float(max(np.max(abs(v[:, -1])) for v in output)),
        bulk_velocity_after=float(2 * w @ (1 - r * r + output[2][0].real)),
        maximum_constraint_residual=maximum_constraint,
        maximum_stationarity_residual=maximum_stationarity,minimum_constraint_rank=min(ranks),
        initial_data_changed=True,continuum_axis_regularity_qualified=False,
        pressure_compatibility_qualified=False,station_usable=False,validated=False)
    return tuple(output), receipt
