"""Explicit synthetic coefficients for numerical verification, not evaluated XS."""

import numpy as np


def slab(cells=16, heterogeneous=False, boundaries=("zero_flux", "zero_flux")):
    if type(cells) is not int or not 1 <= cells <= 512:
        raise ValueError("Invalid case mesh")
    n = cells
    d = np.tile([[0.014], [0.006]], (1, n))
    a = np.tile([[0.09], [0.12]], (1, n))
    f = np.tile([[0.035], [0.1]], (1, n))
    scatter = np.zeros((2, 2, n))
    scatter[1, 0] = 0.04
    scatter[0, 1] = 0.008
    if heterogeneous:
        d[:, n // 2 :] *= 0.65
        a[:, n // 2 :] *= 1.35
        f[:, n // 2 :] *= 0.8
        scatter[0, 1, n // 2 :] *= 1.3
    return dict(
        faces_m=np.linspace(0, 1, n + 1).tolist(),
        diffusion_m=d.tolist(),
        absorption_per_m=a.tolist(),
        nu_fission_per_m=f.tolist(),
        scatter_per_m=scatter.tolist(),
        speed_m_s=np.tile([[100.0], [20.0]], (1, n)).tolist(),
        chi_prompt=np.tile([[1.0], [0.0]], (1, n)).tolist(),
        chi_delayed=np.tile([[0.8], [0.2]], (1, n)).tolist(),
        beta=np.tile([[0.004], [0.002]], (1, n)).tolist(),
        decay_per_s=[0.08, 0.3],
        boundaries=list(boundaries),
        provenance="Synthetic SI coefficients for numerical verification only; not a nuclear-data library or plant model",
    )
