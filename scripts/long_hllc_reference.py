"""Window-budgeted independent HLLC time integration, original RHS/tolerances.

The original 20k RHS limit is enforced per requested output window; total work
is bounded separately. Counters are not thermodynamic state and never enter RHS.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_hem_contact.reference import Assembly
from scripts.graph_history import validate_times

TOTAL_EVALUATIONS=200000


def solve(request,times,method,*,max_step,on_window=None,rtol=2e-11,atol=1e-10):
    times=validate_times(times)
    if method not in ('DOP853','Radau','BDF') or any(not math.isfinite(v) or v<=0 for v in (max_step,rtol,atol)):
        raise ValueError('Explicit reference method and finite positive max step required')
    assembly=Assembly(request,scheme='hllc-local');y=assembly.y0.copy()
    observations=[assembly.observation(0.,y)];windows=[];total=0
    for a,b in zip(times,times[1:]):
        assembly.evaluations=0
        def rhs(time,state):
            if total+assembly.evaluations>=TOTAL_EVALUATIONS:raise ValueError('Long reference total RHS budget exceeded')
            return assembly.rhs(time,state)
        try:
            result=solve_ivp(rhs,(a,b),y,method=method,rtol=rtol,atol=atol,max_step=max_step,first_step=min(1e-6,max_step,b-a))
        except ValueError as error:
            raise ValueError(f'{method} window {a}..{b}, RHS={assembly.evaluations}, completed={total}: {error}') from error
        total+=assembly.evaluations
        if total>TOTAL_EVALUATIONS:raise ValueError('Long reference total RHS budget exceeded')
        if not result.success or not np.isfinite(result.y).all():raise ValueError('Long reference integration refused')
        y=result.y[:,-1];observations.append(assembly.observation(b,y))
        windows.append(dict(start_s=a,end_s=b,evaluations=assembly.evaluations))
        if on_window is not None:on_window(dict(method=method,windows=windows,total_evaluations=total))
    return dict(method=method,observations=observations,evaluations=total,windows=windows,rtol=rtol,atol=atol)
