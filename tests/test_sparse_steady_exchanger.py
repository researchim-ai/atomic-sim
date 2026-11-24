import json
from copy import deepcopy
import numpy as np
import pytest
from core.network import NetworkError
from core.sparse_linear import SparseFactorCache
from physics.thermal.sparse_steady_exchanger import SparseSteadyExchanger
from physics.thermal.two_stream_exchanger import TwoStreamExchanger


def inputs(n=4):
    cfg=dict(cells=n,a_cp_j_kg_k=4180.,b_cp_j_kg_k=3900.,a_cell_mass_kg=[10./n]*n,b_cell_mass_kg=[12./n]*n,
        wall_capacity_j_k=[1e5/n]*n,a_wall_ua_w_k=[4000./n]*n,b_wall_ua_w_k=[4000./n]*n,
        ambient_ua_w_k=[0.]*n,ambient_temperature_k=300.,wall_power_w=[0.]*n,energy_tolerance_j=1e-5)
    boundary=dict(a_mass_flow_kg_s=1.,b_mass_flow_kg_s=-1.,a_inlet_temperature_k=560.,b_inlet_temperature_k=350.)
    return cfg,boundary,dict(mode='steady')


@pytest.mark.parametrize('n',[1,4,16,32])
def test_steady_family_matches_dense_with_one_factorization(n):
    cfg,boundary,initial=inputs(n);cache=SparseFactorCache()
    for temperature in (540.,550.,560.):
        controls=dict(boundary,a_inlet_temperature_k=temperature)
        a=SparseSteadyExchanger(cfg,controls,initial,factor_cache=cache);b=TwoStreamExchanger(cfg,controls,initial)
        np.testing.assert_allclose(a.snapshot()['temperature_k'],b.snapshot()['temperature_k'],rtol=0,atol=1e-9)
        assert np.max(abs(np.asarray(a.initialization_report()['energy_rate_residual_w'])))<1e-6
        np.testing.assert_allclose(a.advance(.2)['temperature_k']['wall'],b.advance(.2)['temperature_k']['wall'],rtol=0,atol=1e-9)
    assert cache.cache_info()['factorizations']==1 and cache.cache_info()['hits']==2


def test_flow_change_creates_new_factor_and_given_mode_avoids_it():
    args=inputs();cache=SparseFactorCache();SparseSteadyExchanger(*args,factor_cache=cache)
    controls=dict(args[1],a_mass_flow_kg_s=-1.)
    SparseSteadyExchanger(args[0],controls,args[2],factor_cache=cache)
    SparseSteadyExchanger(args[0],controls,dict(mode='given',temperature_k=dict(a=[400.]*4,b=[350.]*4,wall=[375.]*4)),factor_cache=cache)
    assert cache.cache_info()['factorizations']==2


def test_sparse_snapshot_restarts_with_fresh_cache_and_exact_suffix():
    a=SparseSteadyExchanger(*inputs());a.set_controls(dict(a.boundary_controls,a_inlet_temperature_k=520.));a.advance(.2)
    saved=json.loads(json.dumps(a.snapshot()));b=SparseSteadyExchanger.from_snapshot(saved)
    assert b.snapshot()==saved and b.factor_cache.cache_info()['factorizations']==1
    assert b.advance(.2)==a.advance(.2)


def test_legacy_snapshot_can_be_restored_with_numerically_close_initial_anchor():
    # Exact initial anchors can differ by roundoff: legacy acceptance remains
    # on the legacy reader; the new reader checks its own physical ledgers.
    a=TwoStreamExchanger(*inputs());a.advance(.2)
    b=SparseSteadyExchanger.from_snapshot(json.loads(json.dumps(a.snapshot())))
    np.testing.assert_allclose(a.snapshot()['temperature_k'],b.snapshot()['temperature_k'],atol=1e-9,rtol=0)


def test_unanchored_zero_flow_rejected():
    cfg,boundary,initial=inputs();boundary.update(a_mass_flow_kg_s=0.,b_mass_flow_kg_s=0.)
    with pytest.raises(NetworkError,match='singular'):SparseSteadyExchanger(cfg,boundary,initial)


def test_failed_temperature_or_energy_guard_does_not_accept_bad_state():
    cfg,boundary,initial=inputs()
    cfg['wall_power_w']=[1e308]*4
    with pytest.raises((NetworkError,ValueError)):SparseSteadyExchanger(cfg,boundary,initial)
