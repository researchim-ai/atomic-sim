from copy import deepcopy
import numpy as np
import pytest
from scipy.sparse import csc_matrix,csr_matrix,coo_matrix,diags
from core.network import NetworkError
from core import sparse_linear as linear


def test_multiple_rhs_reuse_exact_factor_and_match_dense():
    cache=linear.SparseFactorCache();a=np.array([[4.,-1.],[-2.,3.]])
    for b in ([1.,2.],[4.,5.],[-3.,0.]):
        x=cache.solve(a,b);np.testing.assert_allclose(x,np.linalg.solve(a,b),rtol=1e-14,atol=1e-14)
    assert cache.cache_info()['hits']==2 and cache.cache_info()['factorizations']==1
    assert cache.cache_info()['last_backward_error']<=1e-12


def test_csr_csc_duplicate_entries_and_explicit_zeros_have_same_key():
    cache=linear.SparseFactorCache();a=np.array([[4.,-1.],[-2.,3.]])
    cache.solve(a,[1.,2.]);cache.solve(csr_matrix(a),[2.,3.])
    duplicated=coo_matrix(([2.,2.,-1.,-2.,3.,0.],([0,0,0,1,1,1],[0,0,1,0,1,0])),shape=(2,2))
    np.testing.assert_allclose(cache.solve(duplicated,[3.,4.]),np.linalg.solve(a,[3.,4.]))
    assert cache.cache_info()['hits']==2


def test_coefficient_change_is_not_rounded_and_does_not_reuse_factor():
    c=linear.SparseFactorCache();a=np.eye(2);c.solve(a,[1.,1.]);a[0,0]=np.nextafter(1.,2.);c.solve(a,[1.,1.])
    assert c.cache_info()['factorizations']==2


def test_input_and_output_aliases_do_not_mutate_cached_coefficients():
    c=linear.SparseFactorCache();a=np.array([[2.,0.],[0.,3.]]);b=np.array([4.,6.]);x=c.solve(a,b)
    a[0,0]=20.;b[0]=0.;x[0]=100.
    np.testing.assert_array_equal(c.solve(np.diag([2.,3.]),[4.,6.]),[2.,2.])
    assert c.cache_info()['hits']==1


def test_lru_byte_budget_and_explicit_clear():
    c=linear.SparseFactorCache(entries=2,max_bytes=1024)
    for k in (1.,2.,1.,3.,2.):c.solve(np.eye(2)*k,[1.,2.])
    assert c.cache_info()['entries']<=2 and c.cache_info()['bytes']<=1024 and c.cache_info()['hits']==1
    c.clear_cache();assert c.cache_info()['factorizations']==0 and c.cache_info()['bytes']==0


def test_oversized_factor_can_solve_without_retention():
    c=linear.SparseFactorCache(max_bytes=1024);a=diags([2.]*40,format='csc')
    for _ in range(2):np.testing.assert_array_equal(c.solve(a,np.ones(40)),np.full(40,.5))
    assert c.cache_info()['entries']==0 and c.cache_info()['uncached_oversized']==2


@pytest.mark.parametrize('a',[np.zeros((2,2)),np.diag([1.,1e-18]),np.ones((2,2))])
def test_singular_and_numerically_unanchored_matrices_are_not_cached(a):
    c=linear.SparseFactorCache()
    with pytest.raises(NetworkError,match='singular'):c.solve(a,[1.,1.])
    assert c.cache_info()['entries']==0 and c.cache_info()['factorizations']==0


@pytest.mark.parametrize('a',[np.ones((2,3)),np.ones((257,257)),np.eye(2,dtype=complex),np.eye(2,dtype=bool),np.array([[float('nan')]])])
def test_invalid_matrix_inputs(a):
    with pytest.raises(NetworkError):linear.SparseFactorCache().solve(a,np.ones(a.shape[0]))


@pytest.mark.parametrize('b',[[1.],[1.,float('inf')],[True,False],[[1.],[2.]]])
def test_invalid_rhs_inputs(b):
    with pytest.raises(NetworkError):linear.SparseFactorCache().solve(np.eye(2),b)


def test_corrupted_factor_solution_fails_original_matrix_residual(monkeypatch):
    original=linear.splu
    class Corrupt:
        def __init__(self,m):self.factor=original(m)
        def __getattr__(self,k):return getattr(self.factor,k)
        def solve(self,b):return self.factor.solve(b)+1.
    monkeypatch.setattr(linear,'splu',Corrupt)
    c=linear.SparseFactorCache()
    with pytest.raises(NetworkError,match='backward-error'):c.solve(np.eye(2),[1.,2.])
    assert c.cache_info()['entries']==0


@pytest.mark.parametrize('kwargs',[dict(entries=0),dict(entries=True),dict(entries=33),dict(max_bytes=1),dict(max_bytes=2**30)])
def test_bad_cache_budgets(kwargs):
    with pytest.raises(NetworkError):linear.SparseFactorCache(**kwargs)


def test_large_coefficient_row_norm_cannot_overflow_into_false_acceptance(monkeypatch):
    original=linear.splu
    class Corrupt:
        def __init__(self,m):self.factor=original(m)
        def __getattr__(self,k):return getattr(self.factor,k)
        def solve(self,b):return self.factor.solve(b)+.001
    monkeypatch.setattr(linear,'splu',Corrupt)
    a=np.array([[1e308,1e308],[0.,1e308]])
    with pytest.raises(NetworkError,match='backward-error'):linear.SparseFactorCache().solve(a,[1e308,1e308])


def test_invalid_rhs_is_refused_before_array_conversion(monkeypatch):
    c=linear.SparseFactorCache();a=np.eye(2)
    class BadList(list):
        def __array__(self,*a,**k):raise AssertionError('must refuse before allocation')
    with pytest.raises(NetworkError,match='right-hand'):c.solve(a,BadList([1.,2.,3.]))


def test_platform_without_extended_float_range_refuses_overflowed_norm(monkeypatch):
    monkeypatch.setattr(linear.np,'longdouble',np.float64)
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(NetworkError,match='norm is nonfinite'):
            linear.SparseFactorCache().solve(np.array([[1e308,1e308],[0.,1e308]]),[1e308,1e308])
