"""Finite reproducible thermal cases; input bounds do not imply qualification."""
from dataclasses import dataclass, asdict
import re

from atomic_model_station_regular.model import _finite_real


@dataclass(frozen=True)
class ThermalCase:
    name: str = 'baseline'
    seed: int = 75
    viscosity: float = .05
    diffusivity: float = .03
    buoyancy: float = .2
    velocity_amplitude: float = .01
    temperature_mean: float = .1
    temperature_modal: float = .02
    phase: float = .3
    radial_power: int = 1
    mean_axial: float = 0.
    source_mean: float = 0.
    alpha: float = .628319

    def __post_init__(self):
        if type(self.name) is not str or re.fullmatch(r'[a-z][a-z0-9_]{0,47}',self.name) is None:
            raise ValueError('Short lowercase case name required')
        if type(self.seed) is not int or not 0<=self.seed<2**32:
            raise ValueError('Integer random seed required')
        if type(self.radial_power) is not int or not 1<=self.radial_power<=3:
            raise ValueError('Thermal radial power1..3 required')
        bounds=dict(viscosity=(.01,.2),diffusivity=(.01,.2),buoyancy=(-1.,1.),
            velocity_amplitude=(0.,.02),temperature_mean=(-.2,.2),temperature_modal=(-.05,.05),
            phase=(-2.,2.),mean_axial=(-.05,.05),source_mean=(-.02,.02),alpha=(.05,2.))
        for name,(low,high) in bounds.items():_finite_real(getattr(self,name),name,low,high)

    def specification(self):
        return asdict(self)

    @classmethod
    def from_specification(cls,specification):
        expected=set(cls.__dataclass_fields__)
        if type(specification) is not dict or set(specification)!=expected:
            raise ValueError('Complete thermal case specification required')
        return cls(**specification)


def thermal_catalog():
    """Exactly these rows, not the Cartesian box between their parameters."""
    return (
        ThermalCase(),
        ThermalCase(name='negative_b',seed=81,viscosity=.04,diffusivity=.025,buoyancy=-.3,
                    velocity_amplitude=.008,temperature_mean=.08,temperature_modal=.015,phase=-.2,mean_axial=.02),
        ThermalCase(name='equal_diffusivities',seed=83,viscosity=.06,diffusivity=.06,buoyancy=.15,
                    velocity_amplitude=.012,temperature_mean=.12,temperature_modal=.025,phase=.5,mean_axial=.015),
        ThermalCase(name='cooling_source',seed=85,viscosity=.07,diffusivity=.04,buoyancy=.25,
                    temperature_mean=-.08,temperature_modal=-.016,phase=-.4,mean_axial=-.01,source_mean=-.004),
        ThermalCase(name='flat_passive',seed=87,viscosity=.05,diffusivity=.05,buoyancy=0.,
                    velocity_amplitude=.006,temperature_mean=.1,temperature_modal=.015,radial_power=3),
    )
