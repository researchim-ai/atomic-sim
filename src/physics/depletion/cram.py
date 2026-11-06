"""CRAM 16-го порядка в форме неполной частичной факторизации Пусы."""

from __future__ import annotations

import numpy as np

from core.status import ModelPassport, ModelStatus

PASSPORT = ModelPassport(
    model_id="cram16-ipf",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Эволюция линейной цепочки exp(A t) N методом CRAM16",
    approximations=("постоянная матрица скоростей на шаге", "порядок 16"),
    applicability="Жёсткие линейные цепочки умеренной размерности, где частичные факторизации устойчивы",
    data_sources=("Pusa M., Nucl. Sci. Eng. 182 (2016), коэффициенты IPF CRAM16",),
)

_ALPHA = np.array(
    [
        5.464930576870210e3 - 3.797983575308356e4j,
        9.045112476907548e1 - 1.115537522430261e3j,
        2.344818070467641e2 - 4.228020157070496e2j,
        9.453304067358312e1 - 2.951294291446048e2j,
        7.283792954673409e2 - 1.205646080220011e5j,
        3.648229059594851e1 - 1.155509621409682e2j,
        2.547321630156819e1 - 2.639500283021502e1j,
        2.394538338734709e1 - 5.650522971778156e0j,
    ],
    dtype=np.complex128,
)
_THETA = np.array(
    [
        3.509103608414918 + 8.436198985884374j,
        5.948152268951177 + 3.587457362018322j,
        -5.264971343442647 + 16.22022147316793j,
        1.419375897185666 + 10.92536348449672j,
        6.416177699099435 + 1.194122393370139j,
        4.993174737717997 + 5.996881713603942j,
        -1.413928462488886 + 13.49772569889275j,
        -10.84391707869699 + 19.27744616718165j,
    ],
    dtype=np.complex128,
)
_ALPHA0 = 2.124853710495224e-16


def cram16(matrix: np.ndarray, concentration: np.ndarray, dt_s: float) -> np.ndarray:
    """N(dt) для системы dN/dt = A N."""
    operator = np.asarray(matrix, dtype=float) * dt_s
    state = np.asarray(concentration, dtype=float).copy()
    identity = np.eye(operator.shape[0])
    for alpha, theta in zip(_ALPHA, _THETA):
        solved = np.linalg.solve(operator - theta * identity, state.astype(np.complex128))
        state = state + 2.0 * np.real(alpha * solved)
    return _ALPHA0 * state
