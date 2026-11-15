"""Causal research faults; scenario truth is privileged."""
from .spec import Fault, CATALOG

__all__ = ['Fault', 'CATALOG', 'FaultSession', 'Monitor', 'MonitorConfig']


def __getattr__(name):
    if name == 'FaultSession':
        from .model import FaultSession
        return FaultSession
    if name in ('Monitor', 'MonitorConfig'):
        from . import monitor
        return getattr(monitor, name)
    raise AttributeError(name)
