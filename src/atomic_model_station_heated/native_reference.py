"""Regular ODE power series with the external code's discrete mean constraints.

Not a fit to external velocity/temperature data: only native mesh weights are
inputs. Predicts quadrature-induced bias separately from continuum accuracy.
"""
from importlib.resources import files
from importlib import import_module
import json
import numpy as np
from scipy.optimize import root
from atomic_model_multiphysics.provenance import unseal,sha
from core.status import SolverError


def load_quadrature():
    p=unseal(json.loads(files(__package__).joinpath('data/native_quadrature.json').read_text()))
    if p['format']!='openpipeflow-native-quadrature/v1' or set(p['records'])!={'32','64'}:
        raise ValueError('Native quadrature identity differs')
    if p['harness_sha256']!=sha(import_module('scripts.audit_station_heated_quadrature').__file__):
        raise ValueError('Native quadrature audit harness differs')
    return p


def coefficients(parameters,c):
    u0,t0,beta,a=parameters;u=np.zeros(40);t=np.zeros(40);u[0]=u0;t[0]=t0
    for n in range(39):
        u[n+1]=-((1+beta if n==0 else 0)+c*t[n])/(n+1)**2
        t[n+1]=a*u[n]/(4*(n+1)**2)
    return u,t


def native_prediction(boundary,c,cells):
    if boundary not in ('delta_temperature','heat_flux') or c not in (0,2,5) or cells not in (32,64):raise ValueError('Declared native case required')
    row=load_quadrature()['records'][str(cells)];r=np.array(row['radius']);w=np.array(row['weights_rdr']);rr=r*r
    a=6. if boundary=='delta_temperature' else 8.;parameters=np.array([1.,1-3*a/16,0.,a])
    def evaluate(parameters,c):
        u,t=coefficients(parameters,c)
        return np.polynomial.polynomial.polyval(rr,u),np.polynomial.polynomial.polyval(rr,t),u,t
    for at in np.linspace(0,c,1+max(1,2*c)):
        def residual(parameters):
            uv,tv,u,t=evaluate(parameters,at)
            return np.array([np.sum(u),np.sum(t)-1,w@(uv-(1-rr)),
                w@(tv-rr) if boundary=='delta_temperature' else 2*np.arange(40)@t-2])
        result=root(residual,parameters,tol=1e-11,options={'maxfev':300})
        if not np.all(np.isfinite(result.x)) or np.max(np.abs(residual(result.x)))>1e-10:
            raise SolverError('Native-constraint power-series residual')
        parameters=result.x
    uv,tv,u,t=evaluate(parameters,c)
    return dict(radius=r.tolist(),velocity=uv.tolist(),temperature=tv.tolist(),
        pressure_beta=float(parameters[2]),axial_gradient_scaled=float(parameters[3]),
        residual=float(np.max(np.abs(residual(parameters)))),scope='Native discrete mean constraints, not continuum validation')
