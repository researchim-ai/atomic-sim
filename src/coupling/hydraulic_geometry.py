"""Complete one-dimensional dual cells and compatible lumped boundary links.

Reference-density kinetic storage is the existing reduced-model closure,
not a variable-density mechanical-energy or rod-bundle geometry model.
"""
import math
import numpy as np

from .conservative import faces


def _positive(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('Positive finite hydraulic geometry scalar required')
    return float(value)


def dual_geometry(cell_faces, area_m2, loss_coefficient=0.):
    a=faces(cell_faces);area=_positive(area_m2)
    if type(loss_coefficient) not in (int,float) or not math.isfinite(loss_coefficient) or not 0<=loss_coefficient<=100:
        raise ValueError('Channel loss coefficient in0..100 required')
    centres=a[:-1]+np.diff(a)/2
    lengths=np.diff(np.r_[a[0],centres,a[-1]])
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:
            inertance=lengths/area
            loss=loss_coefficient*lengths/(a[-1]-a[0])
        except FloatingPointError as exc:raise ValueError('Hydraulic geometry exceeds finite range') from exc
    if not np.isfinite(inertance).all() or np.any(inertance<=0):raise ValueError('Resolved positive inertance required')
    return dict(lengths_m=lengths.tolist(),inertance_m_inv=inertance.tolist(),loss_coefficients=loss.tolist(),
                length_m=float(a[-1]-a[0]),area_m2=area,loss_coefficient=float(loss_coefficient))


def combine_boundary(*,area_m2,length_m,loss_coefficient,reference_density,
                     channel_area_m2,channel_length_m,channel_loss,channel_reference_density):
    """Represent legacy link + half cell with one momentum/energy owner.

    I_eff/rho_eff = I_old/rho_old + I_half/rho_channel.
    Loss is converted to the old link's velocity-area convention.
    """
    area,length,rho,channel_area,half,rhoc=map(_positive,
        (area_m2,length_m,reference_density,channel_area_m2,channel_length_m,channel_reference_density))
    for k in (loss_coefficient,channel_loss):
        if type(k) not in (int,float) or not math.isfinite(k) or k<0:raise ValueError('Finite nonnegative hydraulic loss required')
    try:
        old=length/area;extra=half/channel_area;total=old+extra
        effective_length=area*total
        effective_density=total/(old/rho+extra/rhoc)
        effective_loss=loss_coefficient+channel_loss*(area/channel_area)**2
        legacy_loss=loss_coefficient/area/area
    except (OverflowError,ZeroDivisionError) as exc:raise ValueError('Boundary geometry exceeds finite range') from exc
    if not all(math.isfinite(v) and v>0 for v in (total,effective_length,effective_density)) or not all(math.isfinite(v) for v in (effective_loss,legacy_loss)):
        raise ValueError('Finite combined boundary geometry required')
    return dict(length_m=effective_length,reference_density_kg_m3=effective_density,
                loss_coefficient=effective_loss,legacy_inertance_m_inv=old,
                channel_inertance_m_inv=extra,total_inertance_m_inv=total,
                kinetic_coefficient_m2_kg=old/rho+extra/rhoc,
                legacy_kinetic_coefficient_m2_kg=old/rho,
                legacy_loss_per_m4=legacy_loss)
