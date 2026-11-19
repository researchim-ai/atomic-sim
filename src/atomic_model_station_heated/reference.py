"""Continuum singular-axis collocation, independent of the FV operators."""
import math
import numpy as np
from scipy.integrate import solve_bvp,quad
from .model import HeatedCase
from core.status import SolverError,ModelLimitError


def continuum(case=None):
    case=HeatedCase() if case is None else case
    if type(case) is not HeatedCase:raise ValueError('HeatedCase required')
    x=np.linspace(0,1,121);a=6. if case.boundary=='delta_temperature' else 8.
    u=1-x*x;t=1-a*(3-4*x*x+x**4)/16
    y=np.array([u,-2*x,t,a*(x/2-x**3/4),x*x/2-x**4/4,
                x*x/2-a*(3*x*x/2-x**4+x**6/6)/16])
    s=np.zeros((6,6));s[1,1]=s[3,3]=-1;params=np.array([0.,a]);solution=None
    for c in np.linspace(0,case.buoyancy_c,max(1,math.ceil(case.buoyancy_c/.5))+1):
        def rhs(r,y,p):return np.array([y[1],-4*(1+p[0]+c*y[2]),y[3],p[1]*y[0],r*y[0],r*y[2]])
        def bc(left,right,p):return np.array([left[1],left[3],left[4],left[5],right[0],right[2]-1,right[4]-.25,
                    right[5]-.25 if case.boundary=='delta_temperature' else right[3]-2])
        solution=solve_bvp(rhs,bc,x,y,p=params,S=s,tol=2e-9,max_nodes=8000)
        if not solution.success:raise SolverError('Heated continuum collocation: '+solution.message)
        x=solution.x;y=solution.y;params=solution.p
    if np.min(solution.sol(np.linspace(0,1-1e-7,501))[0])<=0:
        raise ModelLimitError('Continuum heated branch reverses',criterion='heated_forward_flow',node='heated_pipe',observed=-1.,limit=0.)
    area=float(2*solution.sol(1)[5]);q=float(solution.sol(1)[3])
    mix=quad(lambda r:float(r*solution.sol(r)[0]*solution.sol(r)[2]),0,1,epsabs=1e-11)[0]/.25
    return dict(solution=solution,area_mean_temperature=area,mixing_cup_temperature=mix,wall_gradient=q,
        pressure_beta=float(params[0]),axial_gradient_scaled=float(params[1]),
        nusselt_area=2*q/(1-area),nusselt_mixing=2*q/(1-mix))
