"""Spatial checkpoint and statistical-window audits; no turbulence closure."""
from .modes import layout,field_energy,divergence
from .io import audit_state,compare_states,read_state
from .statistics import WindowPolicy,analyse_window,correlation_time
__all__=['layout','field_energy','divergence','audit_state','compare_states','read_state','WindowPolicy','analyse_window','correlation_time']
