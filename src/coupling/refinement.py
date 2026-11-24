"""Explicit nested PWR meshes with aligned heat contact and full dual momentum.

This constructs configurations, never remaps a running physical state.
"""
import numpy as np

from atomic_model_station import CoreConfig
from .pwr_channel import FluidGrid
from .thermal_grid import aligned_fuel_grid


def coupled_pwr_mesh(cells, *, length_m=4., loss_coefficient=1.):
    """Return session keyword arguments for 2/4/8 neutron and fluid cells.

    Each source/fluid interval has two thermal children: 4/8/16 cells.
    Changing length or loss changes the physical problem, not just resolution.
    """
    if type(cells) is not int or cells not in (2,4,8):
        raise ValueError('Nested coupled resolution must be2,4,8')
    core=CoreConfig(kind='diffusion',cells=cells,length_m=length_m)
    faces=tuple(float(x) for x in np.linspace(0.,core.length_m,cells+1))
    fluid=FluidGrid(faces,loss_coefficient,'complete_dual')
    return dict(core=core,fluid_grid=fluid,
                fuel_grid=aligned_fuel_grid(faces,faces,subdivisions=2))
