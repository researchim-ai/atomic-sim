"""Construct an optional thermal mesh that preserves source/fluid interfaces.

No interfaces are merged by tolerance, no states are remapped, and the
existing FuelGrid state budget is retained.
"""
import numpy as np

from .conservative import ConservativeMap
from .pwr_fuel import FuelGrid


def aligned_fuel_grid(neutron_faces, fluid_faces, *, subdivisions=1):
    """Include every neutron and coolant face, then subdivide each interval.

    Intended for the local constant-per-length fuel/cladding closure. This
    does not infer a physical rod-bundle geometry or refine hydraulic inertia.
    """
    if type(subdivisions) is not int or not 1 <= subdivisions <= 16:
        raise ValueError('Integer thermal subdivisions in 1..16 required')
    exchange = ConservativeMap(neutron_faces, fluid_faces)
    interfaces = np.unique(np.r_[exchange.source_faces, exchange.target_faces])
    if (len(interfaces)-1)*subdivisions > 16:
        raise ValueError('Aligned thermal mesh exceeds the sixteen-cell state budget')
    points = [float(x) for a,b in zip(interfaces[:-1], interfaces[1:])
              for x in np.linspace(a,b,subdivisions+1)[:-1]]
    return FuelGrid(tuple([*points, float(interfaces[-1])]))
