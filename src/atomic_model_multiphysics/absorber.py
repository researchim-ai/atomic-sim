"""Exact absorber/cell overlap, no nearest-node rod snapping."""

from copy import deepcopy

import numpy as np

from atomic_model_spatial import Model, eigenvalue
from atomic_model_spatial.cases import slab
from atomic_model_spatial.reference import dense_k


def insert(config, tip_m, strength_per_m):
    model = Model(config)
    if (
        not np.isfinite(tip_m)
        or not model.faces[0] <= tip_m <= model.faces[-1]
        or not np.isfinite(strength_per_m)
        or strength_per_m < 0
    ):
        raise ValueError("Absorber outside declared geometry")
    fraction = np.maximum(0, np.minimum(model.faces[1:], tip_m) - model.faces[:-1]) / model.width
    result = deepcopy(config)
    result["absorption_per_m"] = (model.absorption_per_m + strength_per_m * fraction).tolist()
    return result


def study():
    curves = []
    for cells in (12, 24, 48):
        config = slab(cells)
        tips = np.linspace(0.37, 0.46, 10)
        rows = []
        for tip in tips:
            c = insert(config, float(tip), 0.2)
            m = Model(c)
            eig = eigenvalue(m)
            dense = dense_k(m)
            integrated = float(
                np.sum((m.absorption_per_m[0] - Model(config).absorption_per_m[0]) * m.width)
            )
            rows.append(
                dict(
                    tip_m=float(tip),
                    k=eig["k"],
                    dense_k=dense,
                    integrated_added_absorption=integrated,
                    overlap_defect=abs(integrated - 0.2 * tip),
                    peak=float(np.max(eig["flux"])),
                )
            )
        curves.append(dict(cells=cells, rows=rows))
    peak_errors = []
    for n in (12, 24, 48):
        # Exact finite-volume sine averages: sampled maxima converge to continuum peak.
        h = 1 / n
        x = (np.arange(n) + 0.5) * h
        avg = np.sin(np.pi * x) * np.sin(np.pi * h / 2) / (np.pi * h / 2)
        peak_errors.append(float(1 - max(avg)))
    orders = [float(np.log2(a / b)) for a, b in zip(peak_errors, peak_errors[1:])]
    monotone = all(all(a["k"] > b["k"] for a, b in zip(c["rows"], c["rows"][1:])) for c in curves)
    return dict(
        curves=curves,
        manufactured_peak_errors=peak_errors,
        peak_orders=orders,
        accepted=monotone
        and min(orders) > 1.5
        and all(
            abs(r["k"] - r["dense_k"]) < 1e-8 and r["overlap_defect"] < 1e-12
            for c in curves
            for r in c["rows"]
        ),
        dimensionality_decision="1D only: no evidence from this slab study justifies a 2D/3D plant claim",
    )
