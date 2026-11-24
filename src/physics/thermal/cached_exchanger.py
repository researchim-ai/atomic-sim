"""Opt-in bounded exact-key transition reuse for the constant-property exchanger.

Caches a linear response to the current derivative, never a physical state or
rounded operating point. The base exchanger remains the default implementation.
"""
from collections import OrderedDict

import numpy as np
from scipy.linalg import expm

from core.network import NetworkError,number
from .two_stream_exchanger import TwoStreamExchanger


class CachedTwoStreamExchanger(TwoStreamExchanger):
    def __init__(self,configuration,boundary_controls,initialization,*,cache_entries=8):
        if type(cache_entries) is not int or not 1<=cache_entries<=32:
            raise NetworkError('Cache requires 1..32 entries')
        self._cache_entries=cache_entries;self._transitions=OrderedDict();self._hits=0;self._misses=0
        super().__init__(configuration,boundary_controls,initialization)

    def cache_info(self):
        return dict(entries=len(self._transitions),capacity=self._cache_entries,hits=self._hits,misses=self._misses,
            bytes=sum(len(k[1])+len(k[2])+value.nbytes for k,value in self._transitions.items()),
            key_policy='Exact dt, matrix and capacity bytes; no rounding',
            byte_scope='Numeric response and key payload only; excludes Python container overhead',physical_states_cached=False)

    def clear_cache(self):
        self._transitions.clear();self._hits=self._misses=0

    def _response(self,dt):
        size=3*self.n
        key=(dt,self._matrix.tobytes(),self._capacity.tobytes())
        if key in self._transitions:
            self._hits+=1;self._transitions.move_to_end(key);return self._transitions[key]
        self._misses+=1
        # z=(deltaT, independent cell-energy integrals, constant current dT/dt).
        # Only the seed derivative changes with state or inlet temperatures.
        augmented=np.zeros((3*size,3*size))
        augmented[:size,:size]=self._matrix
        augmented[:size,2*size:]=np.eye(size)
        augmented[size:2*size,:size]=self._capacity[:,None]*self._matrix
        augmented[size:2*size,2*size:]=np.diag(self._capacity)
        response=np.array(expm(augmented*dt)[:2*size,2*size:],copy=True)
        if not np.isfinite(response).all():raise NetworkError('Nonfinite exchanger transition')
        response.setflags(write=False)
        self._transitions[key]=response
        if len(self._transitions)>self._cache_entries:self._transitions.popitem(last=False)
        return response

    def advance(self,dt_s):
        dt=number(dt_s,'dt_s',positive=True);end=self.time_s+dt
        if not np.isfinite(end) or end<=self.time_s:raise NetworkError('Exchanger clock cannot advance')
        derivative=self._matrix@self._temperature+self._drive
        result=self._response(dt)@derivative;size=3*self.n
        candidate=self._temperature+result[:size]
        integrals=self._integrals+result[size:]
        observed=self._observe(candidate,integrals,end)
        self._temperature,self._integrals,self.time_s=candidate,integrals,end
        return observed
