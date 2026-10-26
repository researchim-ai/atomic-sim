"""Вычислительное ядро: контракты состояния, сетка, порты и интегратор."""

from core.diagnostics import energy_residual, mass_residual
from core.solvers import integrate, observe_cfl, substep_count
from core.status import Applicability, DomainError, ModelPassport, ModelStatus, SolverError

__all__ = [
    "Applicability",
    "DomainError",
    "ModelPassport",
    "ModelStatus",
    "SolverError",
    "energy_residual",
    "integrate",
    "mass_residual",
    "observe_cfl",
    "substep_count",
]
