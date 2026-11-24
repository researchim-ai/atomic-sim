"""Bounded exact-key sparse LU reuse for small real, full-rank systems.

A dense rank guard is evaluated once per matrix miss (dimension <=256).
Every right-hand side still receives a backward-error check. This is not
an approximate Jacobian cache or a nonlinear modified-Newton method.
"""
from collections import OrderedDict
import warnings

import numpy as np
from scipy.sparse import csc_matrix,issparse
from scipy.sparse.linalg import splu

from .network import NetworkError


def _payload(matrix):
    return matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes


class SparseFactorCache:
    def __init__(self,*,entries=8,max_bytes=8*1024*1024):
        if type(entries) is not int or not 1<=entries<=32:raise NetworkError('Factor cache requires 1..32 entries')
        if type(max_bytes) is not int or not 1024<=max_bytes<=64*1024*1024:raise NetworkError('Factor cache byte budget requires 1024..67108864')
        self._limit=entries;self._max_bytes=max_bytes;self._cache=OrderedDict()
        self._hits=self._misses=self._factorizations=self._bytes=self._oversized=0
        self._last_error=None

    def cache_info(self):
        return dict(entries=len(self._cache),capacity=self._limit,bytes=self._bytes,max_bytes=self._max_bytes,
            hits=self._hits,misses=self._misses,factorizations=self._factorizations,uncached_oversized=self._oversized,
            last_backward_error=self._last_error,byte_scope='CSC/LU arrays and byte keys; excludes Python/SuperLU internal overhead',
            rank_guard='Dense numerical rank on matrix misses only',key_policy='Canonical float64 CSC, exact values and indices; no rounding')

    def clear_cache(self):
        self._cache.clear();self._hits=self._misses=self._factorizations=self._bytes=self._oversized=0
        self._last_error=None

    def _canonical(self,matrix):
        if not (isinstance(matrix,np.ndarray) or issparse(matrix)):
            raise NetworkError('Real numpy or scipy sparse matrix required')
        shape=matrix.shape
        if len(shape)!=2 or shape[0]!=shape[1] or not 1<=shape[0]<=256:
            raise NetworkError('Square factor matrix requires 1..256 rows')
        if matrix.dtype.kind not in 'fiu':raise NetworkError('Real nonboolean factor matrix required')
        if issparse(matrix) and matrix.nnz>65536:raise NetworkError('Factor input sparsity budget exceeded')
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            try:
                m=csc_matrix(matrix,dtype=np.float64,copy=True);m.sum_duplicates();m.eliminate_zeros();m.sort_indices()
            except (ValueError,OverflowError,RuntimeWarning) as exc:raise NetworkError('Invalid canonical factor matrix') from exc
        if not np.isfinite(m.data).all():raise NetworkError('Finite factor coefficients required')
        return m

    def _discard(self,key):
        if key in self._cache:
            _,_,size=self._cache.pop(key);self._bytes-=size

    def solve(self,matrix,rhs):
        self._last_error=None
        m=self._canonical(matrix);n=m.shape[0]
        if isinstance(rhs,np.ndarray):
            if rhs.shape!=(n,):raise NetworkError('Complete real right-hand side required')
        elif isinstance(rhs,(list,tuple)):
            if len(rhs)!=n or any(not np.isscalar(v) for v in rhs):raise NetworkError('Complete real right-hand side required')
        else:raise NetworkError('Complete real right-hand side required')
        raw=np.asarray(rhs)
        if raw.shape!=(n,) or raw.dtype.kind not in 'fiu':raise NetworkError('Complete real right-hand side required')
        b=np.array(raw,dtype=np.float64,copy=True)
        if not np.isfinite(b).all():raise NetworkError('Finite right-hand side required')
        key=(n,m.indptr.astype(np.int64).tobytes(),m.indices.astype(np.int64).tobytes(),m.data.tobytes())
        if key in self._cache:
            self._hits+=1;self._cache.move_to_end(key);factor=self._cache[key][1]
        else:
            self._misses+=1
            try:rank=np.linalg.matrix_rank(m.toarray())
            except np.linalg.LinAlgError as exc:raise NetworkError('Numerical rank guard failed') from exc
            if rank!=n:raise NetworkError('Factor matrix is unanchored or numerically singular')
            try:factor=splu(m)
            except RuntimeError as exc:raise NetworkError('Sparse factorization failed') from exc
            self._factorizations+=1
            size=sum(len(x) for x in key[1:])+_payload(m)+_payload(factor.L)+_payload(factor.U)+factor.perm_r.nbytes+factor.perm_c.nbytes
            if size>self._max_bytes:self._oversized+=1
            else:
                while self._cache and (len(self._cache)>=self._limit or self._bytes+size>self._max_bytes):
                    _,(_,_,old_bytes)=self._cache.popitem(last=False);self._bytes-=old_bytes
                self._cache[key]=(m,factor,size);self._bytes+=size
        try:x=factor.solve(b)
        except (RuntimeError,ValueError) as exc:
            self._discard(key);raise NetworkError('Sparse solve failed') from exc
        with np.errstate(over='ignore',invalid='ignore'):
            residual=m@x-b
        if not np.isfinite(x).all() or not np.isfinite(residual).all():
            self._discard(key);raise NetworkError('Nonfinite sparse solution or residual')
        # Long-double norm products prevent overflow from masquerading as zero error.
        norm_a=np.max(np.asarray(abs(m).astype(np.longdouble).sum(axis=1)))
        denominator=norm_a*np.longdouble(np.max(abs(x)))+np.longdouble(np.max(abs(b)))
        numerator=np.longdouble(np.max(abs(residual)))
        if not np.isfinite(norm_a) or not np.isfinite(denominator):
            self._discard(key);raise NetworkError('Sparse solve backward-error norm is nonfinite')
        error=float(numerator/denominator) if denominator else (0. if numerator==0 else float('inf'))
        self._last_error=error
        if not np.isfinite(error) or error>1e-12:
            self._discard(key);raise NetworkError('Sparse solve backward-error guard failed')
        return x
