"""Verified synthetic-data 1D multigroup diffusion, separate from plant runtime."""

from .model import Model
from .solvers import eigenvalue, fixed_source
from .transient import checkpoint, restore, start, step

__all__ = ["Model", "eigenvalue", "fixed_source", "start", "step", "checkpoint", "restore"]
