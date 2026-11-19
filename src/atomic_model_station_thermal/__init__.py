"""Coupled scalar transport and axial Boussinesq research dynamics."""
from .scalar import ScalarMode, scalar_mode
from .system import ThermalSystem, ThermalResult, evolve_thermal

__all__ = ['ScalarMode', 'scalar_mode', 'ThermalSystem', 'ThermalResult', 'evolve_thermal']
from .audit import audit_thermal
__all__ += ['audit_thermal']
