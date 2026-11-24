"""Opt-in CSC Jacobian/SuperLU path for the anchored local-moderator PWR.

All finite-difference coefficients are retained; no sparsity pattern, dropped
small values, frozen Jacobian or new physical approximation is introduced.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
from atomic_model_multiphysics.provenance import seal,unseal
from .pwr_moderator import LocalModeratorPWRSession,identity as moderator_identity
from .pwr_stationary import StationaryConfig,identity as stationary_identity
from .pwr_channel import ChannelPWRSession


def identity():
    return dict(moderator=moderator_identity(),numerical_policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


class SparseLocalModeratorPWRSession(LocalModeratorPWRSession):
    def __init__(self,*args,**kwargs):
        self.sparse_policy_ready=False;self.last_sparse_jacobian=None
        super().__init__(*args,**kwargs)
        self.sparse_policy_ready=True

    def _solver_jacobian(self,matrix):
        if not isinstance(matrix,np.ndarray) or matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError('Complete finite square physical Jacobian required')
        sparse=csc_matrix(matrix)
        self.last_sparse_jacobian=dict(rows=matrix.shape[0],nnz=int(sparse.nnz),total_entries=int(matrix.size),
            all_nonzero_coefficients_retained=True,drop_tolerance=None)
        return sparse

    def snapshot(self):
        result=super().snapshot()
        if self.sparse_policy_ready:
            result.update(model_id='pwr-sparse-local-moderator/v1',sparse_solver_source=identity(),
                sparse_jacobian=deepcopy(self.last_sparse_jacobian),linear_solver_policy=dict(
                    jacobian='Full admitted finite differences converted to CSC without threshold',
                    integrator='SciPy Radau',factorization='SuperLU through Radau; native refresh/reuse policy',
                    physical_approximation_added=False))
        return result

    def checkpoint(self):
        if not self.sparse_policy_ready:return super().checkpoint()
        return seal(dict(format='pwr-sparse-local-moderator/v1',source=identity(),initial_stationary=deepcopy(self.initial_stationary),
            history=list(deepcopy(self.history)),snapshot=self.snapshot()))

    @classmethod
    def restore(cls,saved):
        p=unseal(deepcopy(saved))
        if type(p) is not dict or set(p)!={'format','source','initial_stationary','history','snapshot'} or p['format']!='pwr-sparse-local-moderator/v1' or p['source']!=identity():
            raise ValueError('Sparse PWR checkpoint source/format')
        if type(p['history']) is not list or len(p['history'])>256:raise ValueError('Sparse PWR history budget')
        initial=unseal(p['initial_stationary'])
        if (type(initial) is not dict or set(initial)!={'format','source','stationary','initial_channel','history','snapshot'}
                or initial['format']!='pwr-stationary-channel/v1' or initial['source']!=stationary_identity() or initial['history']!=[]):
            raise ValueError('Unadvanced sparse PWR reference required')
        if type(initial['stationary']) is not dict or set(initial['stationary'])!=set(StationaryConfig.__dataclass_fields__):
            raise ValueError('Sparse PWR stationary configuration')
        seed=ChannelPWRSession.restore(initial['initial_channel'])
        if seed.time!=0 or seed.history:raise ValueError('Unadvanced sparse PWR seed required')
        obj=cls(seed.config,cooling=seed.cooling_config,feed=seed.feed_config,backup=seed.backup_config,core=seed.core_config,
            fuel_grid=seed.fuel_grid,fluid_grid=seed.fluid_grid,max_step_s=seed.max_step_s,rtol=seed.rtol,
            stationary=StationaryConfig(**initial['stationary']))
        for row in p['history']:
            if type(row) is not dict or set(row)!={'duration_s','patch'}:raise ValueError('Sparse PWR history fields')
            obj.advance(**row)
        if obj.checkpoint()!=saved:raise ValueError('Sparse PWR replay differs')
        return obj
