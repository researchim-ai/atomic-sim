"""Exact overlap norms of piecewise constant fields on different meshes.

These are discrepancies between reconstructions, not estimates of continuum
error. No coarse averaging is performed before taking absolute differences.
"""
import numpy as np

from core.status import DomainError
from .conservative import ConservativeMap, vector


def profile_error(faces, values, reference_faces, reference_values, *, kind, scale):
    """Compare cell averages or cell integrals using a declared density scale.

    For cell integrals, values are divided by their own cell widths first.
    Scale therefore has units of field/length for integrals, or field units
    for averages. All three normalized norms use the same explicit scale.
    """
    if kind not in ('cell_average', 'cell_integral'):
        raise ValueError('Explicit cell_average or cell_integral semantics required')
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive finite physical normalization scale required')
    exchange = ConservativeMap(faces, reference_faces)
    a = vector(values, len(exchange.source_widths))
    b = vector(reference_values, len(exchange.target_widths))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if kind == 'cell_integral':
                a = a / exchange.source_widths
                b = b / exchange.target_widths
            support = exchange.overlap > 0
            # Subtract first so identical large values cannot erase a small
            # error elsewhere. Overflow on disjoint cell pairs is irrelevant.
            with np.errstate(over='ignore'):
                difference = a[None, :] - b[:, None]
            if not np.isfinite(difference[support]).all():
                raise DomainError('Profile difference exceeds finite range')
            difference = np.where(support, difference, 0.)
            amplitude = float(np.max(abs(difference))) or 1.
            delta = difference / amplitude
            length = float(exchange.source_faces[-1] - exchange.source_faces[0])
            weights = exchange.overlap / length
            l1 = float(np.sum(weights * abs(delta)) * amplitude)
            l2 = float(np.sqrt(np.sum(weights * delta**2)) * amplitude)
            linf = amplitude if np.any(difference) else 0.
            integral_difference = float(np.sum(weights * delta) * amplitude * length)
            result = dict(l1_mean=l1, l2_rms=l2, linf=linf,
                          normalized_l1=l1/scale, normalized_l2=l2/scale,
                          normalized_linf=linf/scale,
                          integral_difference=integral_difference)
    except FloatingPointError as exc:
        raise DomainError('Profile comparison exceeds finite range') from exc
    if not all(np.isfinite(v) for v in result.values()):
        raise DomainError('Nonfinite profile comparison')
    return dict(result, kind=kind, scale=float(scale), length_m=length,
                interpretation='Exact discrepancy of piecewise constant reconstructions; no continuum-error estimate')
