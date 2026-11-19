"""Separately versioned high-resolution thermal reference dynamics."""
from .velocity import resolved_velocity_mode
from .scalar import resolved_scalar_mode
from .system import ResolvedThermalSystem, resolve_thermal
from .audit import audit_resolved

__all__ = ['resolved_velocity_mode','resolved_scalar_mode','ResolvedThermalSystem','resolve_thermal','audit_resolved']
from .quality import ThermalQualityError, accept_resolved, initial_wall_compatibility
from .comparison import compare_thermal
__all__ += ['ThermalQualityError','accept_resolved','initial_wall_compatibility','compare_thermal']
