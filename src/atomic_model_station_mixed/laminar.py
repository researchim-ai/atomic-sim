"""Steady fully developed axisymmetric Boussinesq UHF tube reference.

x=r/R, v=u/Umean, theta=(T-Twall) k/(q_wall R).
L(v)+A+B theta=0, L(theta)=2v, integral(v*x dx)=1/2.
No slip, theta(1)=0, symmetry at0. B is SIGNED g*beta*q*R^3/(k*nu*U).
Constant properties, no axial conduction, no eddy viscosity or instability
prediction. This is a laminar reference, not a turbulent station substitute.
"""
from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from core.status import DomainError, ModelLimitError, SolverError


@dataclass(frozen=True)
class LaminarCase:
    reynolds: float = 1000.
    prandtl: float = 1.
    buoyancy: float = 0.
    cells: int = 64

    def __post_init__(self):
        for key, low, high in [('reynolds', 1., 2000.), ('prandtl', .6, 200.), ('buoyancy', -1000., 1000.)]:
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Finite laminar parameter: ' + key)
            if not low <= value <= high:
                raise DomainError('Outside selected laminar reference window: ' + key)
        if type(self.cells) is not int or not 8 <= self.cells <= 512:
            raise ValueError('Bounded radial mesh8..512 required')


def solve_laminar(case=None):
    case = case if case is not None else LaminarCase()
    if type(case) is not LaminarCase:
        raise ValueError('LaminarCase required')
    n = case.cells
    h = 1 / n
    faces = np.linspace(0., 1., n + 1)
    x = (faces[1:] + faces[:-1]) / 2
    weights = (faces[1:] ** 2 - faces[:-1] ** 2) / 2
    conductances = faces[1:-1] / h
    # Conservative annular FV matrix; wall is half a cell from the last center.
    diag = np.zeros(n)
    diag[:-1] -= conductances
    diag[1:] -= conductances
    diag[-1] -= 2 / h
    diffusion = diags([conductances, diag, conductances], [-1, 0, 1], format='csc')
    W = diags(weights, format='csc')
    column = csc_matrix(weights[:, None])
    matrix = bmat([[diffusion, case.buoyancy * W, column],
                   [-2 * W, diffusion, None],
                   [column.T, None, csc_matrix((1, 1))]], format='csc')
    rhs = np.zeros(2 * n + 1)
    rhs[-1] = .5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            solution = spsolve(matrix, rhs)
    except (MatrixRankWarning, RuntimeError) as error:
        raise SolverError('Laminar mixed reference matrix failed') from error
    if not np.all(np.isfinite(solution)):
        raise SolverError('Nonfinite laminar reference')
    v = solution[:n]
    theta = solution[n:2*n]
    pressure_wall = float(solution[-1])
    equation_residual = float(np.max(np.abs(matrix @ solution - rhs)))
    if equation_residual > 1e-8 * max(1., float(np.max(np.abs(rhs)))):
        raise SolverError('Laminar mixed reference residual failure')
    if float(np.min(v)) <= 0:
        raise ModelLimitError('Laminar reference contains local flow reversal', criterion='local_forward_velocity',
                              node='vertical_tube', observed=float(np.min(v)), limit=0.)
    mean = float(2 * weights @ v)
    theta_bulk = float(weights @ (v * theta) / (weights @ v))
    wall_heat = float(-2 * theta[-1] / h)
    wall_shear_gradient = float(-2 * v[-1] / h)
    gradient_integral = float(-theta @ (diffusion @ theta))
    momentum_residual = abs(wall_shear_gradient + pressure_wall / 2 + case.buoyancy * float(weights @ theta))
    if theta_bulk >= 0 or gradient_integral <= 0:
        raise DomainError('Laminar heating reference lost its thermal domain')
    report = dict(radius_fraction=x.tolist(), velocity_ratio=v.tolist(), wall_relative_temperature=theta.tolist(),
        nusselt=2 / -theta_bulk, darcy_friction_factor=-16 * wall_shear_gradient / case.reynolds,
        pressure_wall_reference=pressure_wall, pressure_bulk_reference=pressure_wall + case.buoyancy * theta_bulk,
        bulk_wall_relative_temperature=theta_bulk, thermal_gradient_integral=gradient_integral,
        checks=dict(equation_residual=equation_residual,mean_velocity_residual=abs(mean-1),
                    wall_flux_residual=abs(wall_heat-1),momentum_residual=momentum_residual,
                    thermal_work_residual=abs(gradient_integral+theta_bulk)),
        reynolds=case.reynolds,prandtl=case.prandtl,buoyancy=case.buoyancy,cells=n,
        model_id='fully-developed-laminar-boussinesq-tube-uhf',station_usable=False,validated=False,
        assumptions=['Steady laminar axisymmetry imposed, not a stability prediction',
                     'Constant properties, uniform wall heat flux, negligible axial conduction',
                     'Fully developed profiles; entrance and turbulent transition not modeled'])
    if max(report['checks'].values()) > 1e-8:
        raise SolverError('Laminar mixed reference balance failure')
    return report
