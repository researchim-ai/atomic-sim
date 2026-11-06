"""Physical-face group currents, positive in the increasing-x direction."""

import numpy as np

from .model import array


def face_currents(model, flux):
    phi = array(flux, (model.g, model.n), "flux")
    currents = np.zeros((model.g, model.n + 1))
    resistance = model.width[:-1] / (2 * model.diffusion_m[:, :-1]) + model.width[1:] / (
        2 * model.diffusion_m[:, 1:]
    )
    currents[:, 1:-1] = (phi[:, :-1] - phi[:, 1:]) / resistance
    if model.boundaries[0] == "zero_flux":
        currents[:, 0] = -2 * model.diffusion_m[:, 0] * phi[:, 0] / model.width[0]
    if model.boundaries[1] == "zero_flux":
        currents[:, -1] = 2 * model.diffusion_m[:, -1] * phi[:, -1] / model.width[-1]
    return currents
