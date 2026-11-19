"""Atomic wrapper: commit a Stokes step only after physical pressure recovery."""
from pathlib import Path
import numpy as np
from atomic_model_multiphysics.provenance import sha,seal,unseal
from atomic_model_station_regular import StokesSession
from . import field as pressure_module
from .field import recover_step_pressure


def policy_identity():
    return dict(wrapper=sha(Path(__file__)),pressure=sha(pressure_module.__file__))


class PressureStokesSession:
    def __init__(self,model,q,*,viscosity=.1):
        self._core=StokesSession(model,q,viscosity=viscosity)

    @property
    def model(self):return self._core.model

    @property
    def q(self):return self._core.q.copy()

    @property
    def time(self):return self._core.time

    def advance(self,dt,force=None):
        candidate=StokesSession.restore(self._core.checkpoint())
        f=np.zeros(3*self.model.order,complex) if force is None else np.asarray(force,dtype=complex)
        receipt=candidate.advance(dt,f)
        pressure,proof=recover_step_pressure(self.model,self._core.q,candidate.q,dt,f,self._core.viscosity)
        # Pressure is a midpoint algebraic field, not an independently integrated state.
        receipt=dict(receipt,physical_pressure_reconstructed=True,pressure=proof,
            pressure_time=self.time+dt/2)
        self._core=candidate
        return receipt,pressure

    def checkpoint(self):
        return seal(dict(format='regular-stokes-physical-pressure/v1',policy=policy_identity(),core=self._core.checkpoint()))

    @classmethod
    def restore(cls,checkpoint):
        p=unseal(checkpoint)
        if type(p) is not dict or set(p)!={'format','policy','core'} or p['format']!='regular-stokes-physical-pressure/v1':
            raise ValueError('Complete pressure Stokes checkpoint required')
        if p['policy']!=policy_identity():raise ValueError('Pressure policy identity differs')
        obj=cls.__new__(cls);obj._core=StokesSession.restore(p['core']);return obj
