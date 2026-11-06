"""Independent dense incidence assembly and modal/continuum references.

Does not read production sparse matrices. Limited to small verification cases.
"""

import numpy as np
from scipy.linalg import eigvals, expm


def dense_operators(model):
    g, n, j = model.g, model.n, model.j
    if (g + j) * n > 256:
        raise ValueError("Dense reference budget exceeded")
    size = g * n
    incidence = np.zeros((n - 1, n))
    for c in range(n - 1):
        incidence[c, c : c + 2] = [1, -1]
    loss = np.zeros((size, size))
    fiss = np.zeros_like(loss)
    prompt = np.zeros_like(loss)
    beta = model.beta.sum(axis=0)
    for group in range(g):
        d = model.diffusion_m[group]
        conductance = 2 / (model.width[:-1] / d[:-1] + model.width[1:] / d[1:])
        block = incidence.T @ np.diag(conductance) @ incidence
        if model.boundaries[0] == "zero_flux":
            block[0, 0] += 2 * d[0] / model.width[0]
        if model.boundaries[1] == "zero_flux":
            block[-1, -1] += 2 * d[-1] / model.width[-1]
        sl = slice(group * n, (group + 1) * n)
        loss[sl, sl] = block + np.diag(
            (model.absorption_per_m[group] + model.scatter[:, group].sum(axis=0)) * model.width
        )
    for to in range(g):
        for origin in range(g):
            rows = slice(to * n, (to + 1) * n)
            cols = slice(origin * n, (origin + 1) * n)
            loss[rows, cols] -= np.diag(model.scatter[to, origin] * model.width)
            p = (1 - beta) * model.chi_prompt[to] * model.nu_fission_per_m[origin]
            f = p + beta * model.chi_delayed[to] * model.nu_fission_per_m[origin]
            fiss[rows, cols] = np.diag(f * model.width)
            prompt[rows, cols] = np.diag(p * model.width)
    operator = np.zeros(((g + j) * n, (g + j) * n))
    operator[:size, :size] = np.diag((model.speed_m_s / model.width).ravel()) @ (prompt - loss)
    for c in range(n):
        for family in range(j):
            index = size + family * n + c
            operator[index, index] = -model.decay[family]
            for group in range(g):
                operator[index, group * n + c] = (
                    model.beta[family, c] * model.nu_fission_per_m[group, c]
                )
                operator[group * n + c, index] = (
                    model.speed_m_s[group, c] * model.chi_delayed[group, c] * model.decay[family]
                )
    return loss, fiss, operator


def dense_k(model):
    loss, fiss, _ = dense_operators(model)
    return float(max(eigvals(fiss, loss).real))


def exact_evolution(model, flux, precursors, source, time_s):
    _, _, generator = dense_operators(model)
    initial = np.concatenate((np.ravel(flux), np.ravel(precursors), [1.0]))
    augmented = np.zeros((len(initial), len(initial)))
    augmented[:-1, :-1] = generator
    augmented[: model.g * model.n, -1] = (np.asarray(source) * model.speed_m_s).ravel()
    return (expm(augmented * time_s) @ initial)[:-1]


def continuum_k(model):
    """Homogeneous slab zero-face-flux fundamental mode, no spatial discretisation."""
    length = model.faces[-1] - model.faces[0]
    b = model.beta[:, 0].sum()
    removal = (
        np.diag(model.absorption_per_m[:, 0] + model.scatter[:, :, 0].sum(axis=0))
        - model.scatter[:, :, 0]
    )
    removal += np.diag(model.diffusion_m[:, 0] * (np.pi / length) ** 2)
    chi = (1 - b) * model.chi_prompt[:, 0] + b * model.chi_delayed[:, 0]
    fiss = np.outer(chi, model.nu_fission_per_m[:, 0])
    return float(max(eigvals(fiss, removal).real))


def two_layer_source(x, left_d=0.01, right_d=0.04):
    """Exact -d(D dphi/dx)/dx=1, phi(0)=phi(1)=0, D jump at1/2."""
    weighted_x = 0.125 / left_d + 0.375 / right_d
    resistance = 0.5 / left_d + 0.5 / right_d
    current_constant = -weighted_x / resistance
    x = np.asarray(x)
    left = np.minimum(x, 0.5)
    right = np.maximum(x - 0.5, 0)
    return (
        -(left**2 / 2 + current_constant * left) / left_d
        - ((right**2 + right) / 2 + current_constant * right) / right_d
    )
