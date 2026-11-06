"""Analytic IF97 region-1 derivatives in conserved M/U coordinates."""

from dataclasses import dataclass

import numpy as np

from core.status import DomainError
from physics.materials.iapws import _region1_raw


@dataclass(frozen=True)
class LiquidDerivatives:
    pressure: np.ndarray  # Pa per [kg, J]
    temperature: np.ndarray  # K per [kg, J]
    enthalpy: np.ndarray  # (J/kg) per [kg, J]
    density: np.ndarray  # (kg/m3) per [kg, J]


def conserved_derivatives(state, mass_kg, energy_j, volume_m3):
    """Inverse-function theorem for (v,u) with exact IF97 Gibbs derivatives.

    p in raw derivatives is MPa, u is kJ/kg; public outputs are all SI.
    Caller supplies the liquid EOS solution for these same inventories.
    """
    if (
        state.region != 1
        or not np.isfinite([mass_kg, energy_j, volume_m3]).all()
        or mass_kg <= 0
        or volume_m3 <= 0
    ):
        raise DomainError("Conserved derivatives require a positive liquid inventory")
    raw = _region1_raw(state.T_k, state.p_pa / 1e6)
    matrix = np.array([[raw["dv_dp_mpa"], raw["dv_dt"]], [raw["du_dp_mpa"], raw["du_dt"]]])
    target = np.array(
        [[-volume_m3 / mass_kg**2, 0.0], [-energy_j / (1000 * mass_kg**2), 1 / (1000 * mass_kg)]]
    )
    try:
        inverse = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError as exc:
        raise DomainError("Singular liquid thermodynamic Jacobian") from exc
    dp = inverse[0] * 1e6
    dt = inverse[1]
    v = volume_m3 / mass_kg
    h = energy_j / mass_kg + state.p_pa * v
    dh = v * dp + np.array([-h / mass_kg, 1 / mass_kg])
    rho = np.array([1 / volume_m3, 0.0])
    if not np.isfinite(np.r_[dp, dt, dh, rho]).all():
        raise DomainError("Nonfinite liquid derivatives")
    return LiquidDerivatives(dp, dt, dh, rho)
