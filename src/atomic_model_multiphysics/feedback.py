"""Tensor tables with complete axes and fail-closed interpolation."""

from copy import deepcopy
from itertools import product

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from atomic_model_training.spec import exact
from atomic_model_workspace.common import digest

AXES = (
    "fuel_temperature_k",
    "coolant_temperature_k",
    "density_kg_m3",
    "boron_ppm",
    "composition_fraction",
)


class FeedbackTable:
    def __init__(self, specification):
        exact(specification, {"axes", "values", "quantity", "provenance", "uncertainty", "family"})
        exact(specification["axes"], AXES)
        if specification["family"] not in ("numerical", "evaluated"):
            raise ValueError("Unknown nuclear coefficient family")
        if specification["quantity"] != "absorption multiplier":
            raise ValueError("Unsupported feedback quantity")
        if (
            not isinstance(specification["provenance"], str)
            or not specification["provenance"]
            or not isinstance(specification["uncertainty"], str)
            or not specification["uncertainty"]
        ):
            raise ValueError("Provenance and uncertainty semantics required")
        self.specification = deepcopy(specification)
        axes = []
        for name in AXES:
            a = np.array(specification["axes"][name], float)
            if (
                a.ndim != 1
                or not 2 <= len(a) <= 16
                or not np.isfinite(a).all()
                or np.any(np.diff(a) <= 0)
            ):
                raise ValueError("Invalid feedback axis " + name)
            if (
                name in ("fuel_temperature_k", "coolant_temperature_k", "density_kg_m3")
                and a[0] <= 0
                or name == "boron_ppm"
                and a[0] < 0
                or name == "composition_fraction"
                and (a[0] < 0 or a[-1] > 1)
            ):
                raise ValueError("Feedback axis outside physical domain")
            axes.append(a)
        values = np.array(specification["values"], float)
        if (
            values.shape != tuple(len(a) for a in axes)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError("Invalid positive feedback tensor")
        self.interpolator = RegularGridInterpolator(axes, values, bounds_error=True)
        self.sha256 = digest(specification)

    def __call__(self, **state):
        exact(state, AXES)
        point = np.array([state[k] for k in AXES], float)
        if not np.isfinite(point).all():
            raise ValueError("Nonfinite feedback state")
        return float(self.interpolator(point[None, :])[0])


def numerical_table():
    axes = dict(
        zip(AXES, [[300.0, 1600.0], [300.0, 650.0], [600.0, 1100.0], [0.0, 2000.0], [0.0, 1.0]])
    )
    values = np.empty((2,) * 5)
    for ix in product(range(2), repeat=5):
        tf, tc, rho, b, c = [axes[k][j] for k, j in zip(AXES, ix)]
        values[ix] = (
            1
            + 0.00015 * (tf - 600)
            + 0.00005 * (tc - 560)
            + 0.00002 * (750 - rho)
            + b * 1e-5
            + c * 0.01
        )
    return dict(
        axes=axes,
        values=values.tolist(),
        quantity="absorption multiplier",
        family="numerical",
        provenance="Explicit synthetic affine feedback verification fixture, not evaluated transport responses",
        uncertainty="No physical uncertainty model; only interpolation and coupling verification",
    )
