"""Explicit thermal constraints and independently checked laminar mean profiles."""
from .model import HeatedCase,solve_heated,flux_to_delta
from .reference import continuum
__all__=['HeatedCase','solve_heated','flux_to_delta','continuum']
