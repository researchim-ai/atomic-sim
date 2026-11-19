"""Independent collocation reference for the laminar FV model.

Different discretization: solve_bvp with explicit singular-axis matrix, integral
mass coordinate and unknown pressure parameter. Temperature bulk mean is
integrated by adaptive quadrature, not FV cell weights. Never used in the FV RHS.
"""
import numpy as np
from scipy.integrate import quad, solve_bvp
from core.status import SolverError
from .laminar import LaminarCase


def collocation(case=None):
    case = case if case is not None else LaminarCase()
    if type(case) is not LaminarCase:
        raise ValueError('LaminarCase required')
    x = np.linspace(0., 1., 49)
    guess = np.array([2*(1-x*x), -4*x, x*x-x**4/4-.75, 2*x-x**3, x*x-x**4/2])
    def fun(x,y,p):
        return np.array([y[1],-p[0]-case.buoyancy*y[2],y[3],2*y[0],x*y[0]])
    def boundary(a,b,p):
        return np.array([a[1],a[3],a[4],b[0],b[2],b[4]-.5])
    singular = np.diag([0.,-1.,0.,-1.,0.])
    result = solve_bvp(fun,boundary,x,guess,p=np.array([8.]),S=singular,tol=2e-10,max_nodes=4096)
    if not result.success:
        raise SolverError('Independent laminar collocation failed')
    mean = quad(lambda r:float(result.sol(r)[0])*r,0.,1.,epsabs=1e-11,epsrel=1e-11)[0]
    mixed = quad(lambda r:float(result.sol(r)[0]*result.sol(r)[2])*r,0.,1.,epsabs=1e-11,epsrel=1e-11)[0]/mean
    sample = np.linspace(0.,1.,1025)
    return dict(nusselt=2/-mixed,wall_shear_gradient=float(result.sol(1.)[1]),
                pressure_wall_reference=float(result.p[0]),mean_velocity=2*mean,
                maximum_collocation_residual=float(np.max(result.rms_residuals)),
                sample_radius_fraction=sample.tolist(),velocity_ratio=result.sol(sample)[0].tolist(),
                wall_relative_temperature=result.sol(sample)[2].tolist(),
                station_usable=False,validated=False)
