"""Explicit branches of IAEA-TECDOC-1746 Eq.(6.9), printed p.178.

r = |1 +/- 1e5 Bo_q r**(-2.1)|**0.46.
Upward heating uses minus, downward heating plus. Constants are the stated
estimated constants of the equation, not fitted to its schematic Fig.6.8.
This is arithmetic qualification of a semi-empirical framework, not a closure
qualified for high-pressure PWR water. No station routing is provided here.
"""
from dataclasses import dataclass
from math import expm1, isfinite, log
from numbers import Real
from scipy.optimize import brentq
from core.status import DomainError, ModelLimitError, SolverError

ALPHA = .46
POWER = 2.1
COEFFICIENT = 1e5
SOURCE = 'https://www-pub.iaea.org/MTCD/Publications/PDF/TE-1746_web.pdf#page=191'
FOLD_RATIO = (POWER / (POWER + 1 / ALPHA)) ** ALPHA


def _upper_function(r):
    if r == 0.:
        return 0.
    return r ** POWER * (-expm1(log(r) / ALPHA)) / COEFFICIENT


FOLD_BO = _upper_function(FOLD_RATIO)


def _plus_function(r, inverted):
    return r ** POWER * (r ** (1 / ALPHA) + (1 if inverted else -1)) / COEFFICIENT


@dataclass(frozen=True)
class RatioResult:
    ratio: float
    buoyancy_parameter: float
    direction: str
    branch: str
    algebraic_residual: float
    original_equation_residual: float
    tangent: bool
    derivative_per_bo: float | None
    station_usable: bool = False
    validated: bool = False


def solve_ratio(buoyancy_parameter, *, direction, branch):
    """Solve one named branch; never select a branch or infer hysteresis.

    Numerical study domain: Bo=0 or 1e-10<=Bo<=1e-3. These are NOT claimed
    experimentally validated parameter bounds. Middle/inverted at Bo=0 have
    no positive root of the original equation and are rejected.
    """
    bo = buoyancy_parameter
    if not isinstance(bo, Real) or isinstance(bo, bool) or not isfinite(bo):
        raise ValueError('Finite real buoyancy parameter required')
    if bo != 0 and not 1e-10 <= bo <= 1e-3:
        raise DomainError('Outside selected numerical study parameter window')
    if type(direction) is not str or direction not in ('upward', 'downward'):
        raise ValueError('Explicit vertical direction required')
    allowed = ('upper', 'middle', 'inverted') if direction == 'upward' else ('unique',)
    if type(branch) is not str or branch not in allowed:
        raise ValueError('Explicit branch for the selected direction required')
    if bo == 0:
        if branch in ('middle', 'inverted'):
            raise DomainError('No positive zero-buoyancy root on this branch')
        return RatioResult(1., 0., direction, branch, 0., 0., False,
                           (-1 if direction == 'upward' else 1) * ALPHA * COEFFICIENT)
    tangent = False
    if branch in ('upper', 'middle'):
        if bo > FOLD_BO:
            raise ModelLimitError('Requested mixed-convection branch has ended',
                                  criterion='mixed_branch_fold', node='vertical_tube', observed=bo, limit=FOLD_BO)
        if bo == FOLD_BO:
            r = FOLD_RATIO
            tangent = True
        else:
            bounds = (FOLD_RATIO, 1.) if branch == 'upper' else (0., FOLD_RATIO)
            r = brentq(lambda y: _upper_function(y) - bo, *bounds, xtol=5e-324, rtol=1e-15)
        value = _upper_function(r)
        derivative = None if tangent else COEFFICIENT / (
            POWER * r ** (POWER - 1) * (1 - r ** (1 / ALPHA))
            - r ** (POWER + 1 / ALPHA - 1) / ALPHA)
    else:
        inverted = branch == 'inverted'
        low = 0. if inverted else 1.
        high = 2.
        while _plus_function(high, inverted) < bo:
            high *= 2
        r = brentq(lambda y: _plus_function(y, inverted) - bo, low, high, xtol=5e-324, rtol=1e-15)
        value = _plus_function(r, inverted)
        sign = 1 if inverted else -1
        derivative = COEFFICIENT / (
            POWER * r ** (POWER - 1) * (r ** (1 / ALPHA) + sign)
            + r ** (POWER + 1 / ALPHA - 1) / ALPHA)
    residual = abs(value - bo) / bo
    original = abs(r - abs(1 + (-1 if direction == 'upward' else 1) * COEFFICIENT * bo * r ** -POWER) ** ALPHA)
    if not isfinite(r) or not r > 0 or residual > 1e-10 or original > 1e-8:
        raise SolverError('Mixed-convection branch residual failure')
    return RatioResult(float(r), float(bo), direction, branch, float(residual), float(original), tangent,
                       None if derivative is None else float(derivative))
