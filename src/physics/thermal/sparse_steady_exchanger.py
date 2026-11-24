"""Optional sparse steady initialization; native exchanger dynamics unchanged."""
import numpy as np
from core.network import NetworkError
from core.sparse_linear import SparseFactorCache
from .two_stream_exchanger import TwoStreamExchanger


class SparseSteadyExchanger(TwoStreamExchanger):
    def __init__(self,configuration,boundary_controls,initialization,*,factor_cache=None):
        if factor_cache is not None and type(factor_cache) is not SparseFactorCache:
            raise NetworkError('SparseFactorCache required')
        self.factor_cache=SparseFactorCache() if factor_cache is None else factor_cache
        super().__init__(configuration,boundary_controls,initialization)

    def _initialize(self,initialization):
        if initialization!={'mode':'steady'}:return super()._initialize(initialization)
        state=self.factor_cache.solve(self._matrix,-self._drive)
        if not np.isfinite(state).all() or np.min(state)<=0:raise NetworkError('Invalid sparse initialized temperatures')
        residual=self._capacity*(self._matrix@state+self._drive)
        if max(abs(residual))>1e-6:raise NetworkError('Sparse steady residual exceeds 1e-6 W per cell')
        return state
