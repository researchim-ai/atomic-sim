"""Optional atomic acceptance gate on sampled strong Stokes momentum."""
from pathlib import Path
import numpy as np
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_station_regular.model import _finite_real
from atomic_model_station_stokes_pressure import PressureStokesSession
from . import operators


class WallMomentumError(ValueError):
    def __init__(self,audit):
        super().__init__('Sampled strong momentum rejected; candidate not committed')
        self.audit=audit


def _policy():return dict(session=sha(Path(__file__)),operators=sha(operators.__file__))


def _radii(radius):
    r=np.array(radius,dtype=float,copy=True)
    if r.ndim!=1 or not 2<=len(r)<=4096 or not np.all(np.isfinite(r)) or r[0]<0 or r[-1]!=1. or np.any(np.diff(r)<=0):
        raise ValueError('Finite increasing audit radii ending at wall required')
    r.setflags(write=False);return r


class WallCheckedSession:
    def __init__(self,model,q,*,radius,viscosity=.1,tolerance=1e-8):
        self.radius=_radii(radius);_finite_real(tolerance,'tolerance',1e-14,1e-4)
        self.tolerance=float(tolerance);self._viscosity=float(viscosity)
        self._core=PressureStokesSession(model,q,viscosity=viscosity)

    @property
    def model(self):return self._core.model

    @property
    def q(self):return self._core.q

    @property
    def time(self):return self._core.time

    def advance(self,dt,force=None):
        """force(r,t_mid) must provide physical vector values, never weak loads."""
        _finite_real(dt,'step',1e-10,1e3)
        if force is not None and not callable(force):raise ValueError('Physical force callback(r,time) required')
        mid=self.time+dt/2
        def evaluate(r):
            f=np.zeros((3,len(r)),complex) if force is None else np.asarray(force(r.copy(),mid),dtype=complex)
            if f.shape!=(3,len(r)) or not np.all(np.isfinite(f)):raise ValueError('Complete finite physical force values required')
            return f
        load=self.model.load(evaluate(self.model.radius));boundary_force=evaluate(self.radius)
        candidate=PressureStokesSession.restore(self._core.checkpoint())
        receipt,pressure=candidate.advance(dt,load)
        audit=operators.audit_momentum(self.model,self.q,candidate.q,pressure,dt=dt,viscosity=self._viscosity,
            radius=self.radius,force=boundary_force,tolerance=self.tolerance)
        if not audit['sampled_momentum_passed'] or not audit['wall_momentum_passed']:raise WallMomentumError(audit)
        self._core=candidate
        return dict(receipt,strong_momentum=audit),pressure

    def checkpoint(self):
        return seal(dict(format='wall-checked-stokes/v1',policy=_policy(),radius=self.radius.tolist(),tolerance=self.tolerance,core=self._core.checkpoint()))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        if type(p) is not dict or set(p)!={'format','policy','radius','tolerance','core'} or p['format']!='wall-checked-stokes/v1':raise ValueError('Complete wall-checked checkpoint required')
        if p['policy']!=_policy():raise ValueError('Strong momentum policy identity differs')
        obj=cls.__new__(cls);obj.radius=_radii(p['radius']);_finite_real(p['tolerance'],'tolerance',1e-14,1e-4)
        obj.tolerance=float(p['tolerance']);obj._core=PressureStokesSession.restore(p['core'])
        obj._viscosity=unseal(unseal(p['core'])['core'])['viscosity']
        return obj
