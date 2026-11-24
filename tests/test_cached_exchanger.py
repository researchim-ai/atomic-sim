"""Cache correctness, independent ledgers, invalidation and bounded memory."""
from copy import deepcopy
import json

import numpy as np
import pytest
from core.network import NetworkError
from physics.thermal import cached_exchanger as cached
from physics.thermal.two_stream_exchanger import TwoStreamExchanger


def inputs(n=3):
    cfg=dict(cells=n,a_cp_j_kg_k=4180.,b_cp_j_kg_k=3900.,a_cell_mass_kg=[10./n]*n,
        b_cell_mass_kg=[12./n]*n,wall_capacity_j_k=[1e5/n]*n,a_wall_ua_w_k=[4000./n]*n,
        b_wall_ua_w_k=[4000./n]*n,ambient_ua_w_k=[0.]*n,ambient_temperature_k=300.,
        wall_power_w=[0.]*n,energy_tolerance_j=1e-5)
    controls=dict(a_mass_flow_kg_s=1.,b_mass_flow_kg_s=-1.,a_inlet_temperature_k=560.,b_inlet_temperature_k=350.)
    initial=dict(mode='given',temperature_k=dict(a=[500.]*n,b=[360.]*n,wall=[430.]*n))
    return cfg,controls,initial


def compare(a,b):
    for name in ('a','b','wall'):
        np.testing.assert_allclose(a['temperature_k'][name],b['temperature_k'][name],rtol=0,atol=1e-9)
    np.testing.assert_allclose(a['integrated_net_cell_energy_j'],b['integrated_net_cell_energy_j'],rtol=1e-10,atol=1e-6)
    assert abs(a['energy_residual_j'])<1e-5 and abs(a['port_power_residual_w'])<1e-6


@pytest.mark.parametrize('n',[1,3,8,16,32])
def test_cached_response_agrees_with_default_and_survives_multiple_states(n):
    args=inputs(n);a=cached.CachedTwoStreamExchanger(*args);b=TwoStreamExchanger(*args)
    for _ in range(6):compare(a.advance(.1),b.advance(.1))
    assert a.cache_info()['hits']==5 and a.cache_info()['misses']==1


def test_inlet_temperature_change_reuses_response_but_recomputes_drive():
    args=inputs();a=cached.CachedTwoStreamExchanger(*args);b=TwoStreamExchanger(*args)
    compare(a.advance(.2),b.advance(.2))
    boundary=dict(args[1],a_inlet_temperature_k=520.,b_inlet_temperature_k=370.)
    a.set_controls(boundary);b.set_controls(boundary)
    compare(a.advance(.2),b.advance(.2))
    assert a.cache_info()['misses']==1 and a.cache_info()['hits']==1


def test_flow_reversal_changes_key_and_return_to_old_flow_can_hit():
    args=inputs();a=cached.CachedTwoStreamExchanger(*args);b=TwoStreamExchanger(*args)
    compare(a.advance(.1),b.advance(.1))
    for flow in (-2.,1.):
        controls=dict(args[1],a_mass_flow_kg_s=flow)
        a.set_controls(controls);b.set_controls(controls);compare(a.advance(.1),b.advance(.1))
    assert a.cache_info()['misses']==2 and a.cache_info()['hits']==1


def test_nearby_time_steps_are_not_rounded_into_one_key():
    a=cached.CachedTwoStreamExchanger(*inputs())
    a.advance(.1);a.advance(np.nextafter(.1,1.))
    assert a.cache_info()['misses']==2


def test_lru_budget_and_cache_clearing_do_not_change_physical_snapshot():
    a=cached.CachedTwoStreamExchanger(*inputs(),cache_entries=2)
    for dt in (.1,.2,.1,.3,.2):a.advance(dt)
    info=a.cache_info();assert info['entries']==2 and info['hits']==1 and info['misses']==4
    assert 0<info['bytes']<20000
    saved=a.snapshot();a.clear_cache();assert a.snapshot()==saved and a.cache_info()['entries']==0


def test_restart_has_no_serialized_or_trusted_cache_and_reproduces_suffix_exactly():
    a=cached.CachedTwoStreamExchanger(*inputs());a.advance(.1)
    saved=json.loads(json.dumps(a.snapshot(),allow_nan=False));b=cached.CachedTwoStreamExchanger.from_snapshot(saved)
    assert b.cache_info()['entries']==0 and b.snapshot()==saved
    assert b.advance(.1)==a.advance(.1) and b.snapshot()==a.snapshot()


def test_legacy_snapshot_can_be_used_by_cached_implementation():
    a=TwoStreamExchanger(*inputs());a.advance(.1)
    b=cached.CachedTwoStreamExchanger.from_snapshot(json.loads(json.dumps(a.snapshot())))
    compare(b.advance(.2),a.advance(.2))


def test_independent_energy_integral_block_is_not_replaced_by_state_difference(monkeypatch):
    a=cached.CachedTwoStreamExchanger(*inputs());before=a.snapshot();original=cached.expm
    def corrupt(matrix):
        result=original(matrix);size=3*a.n;result[size:2*size,2*size:]*=1.01;return result
    monkeypatch.setattr(cached,'expm',corrupt)
    with pytest.raises(NetworkError,match='ledger'):a.advance(.1)
    assert a.snapshot()==before


def test_failed_propagator_is_not_cached_or_committed(monkeypatch):
    a=cached.CachedTwoStreamExchanger(*inputs());before=a.snapshot()
    monkeypatch.setattr(cached,'expm',lambda m:np.full_like(m,np.nan))
    with pytest.raises(NetworkError,match='Nonfinite'):a.advance(.1)
    assert a.snapshot()==before and a.cache_info()['entries']==0


@pytest.mark.parametrize('budget',[0,33,True,1.5])
def test_invalid_cache_budget_rejected(budget):
    with pytest.raises(NetworkError,match='1..32'):cached.CachedTwoStreamExchanger(*inputs(),cache_entries=budget)


def test_response_cannot_be_mutated_through_returned_cache_info():
    a=cached.CachedTwoStreamExchanger(*inputs());a.advance(.1)
    info=a.cache_info();info['entries']=1000
    assert a.cache_info()['entries']==1
    response=next(iter(a._transitions.values()))
    with pytest.raises(ValueError):response[0,0]=0.


def test_closed_one_cell_mode_matches_analytic_decay_and_conserves_energy():
    cfg,controls,initial=inputs(1)
    cfg.update(a_cp_j_kg_k=1000.,b_cp_j_kg_k=1000.,a_cell_mass_kg=[1.],b_cell_mass_kg=[1.],
        wall_capacity_j_k=[1000.],a_wall_ua_w_k=[100.],b_wall_ua_w_k=[100.])
    controls.update(a_mass_flow_kg_s=0.,b_mass_flow_kg_s=0.)
    initial['temperature_k']=dict(a=[330.],b=[300.],wall=[315.])
    a=cached.CachedTwoStreamExchanger(cfg,controls,initial)
    for _ in range(10):row=a.advance(.7)
    expected=315.+np.array([15.,-15.,0.])*np.exp(-.1*7.)
    np.testing.assert_allclose(np.array(list(row['temperature_k'].values())).ravel(),expected,rtol=0,atol=1e-10)
    assert abs(row['stored_energy_change_j'])<1e-8


def test_different_material_capacities_cannot_share_operators_between_instances():
    args=inputs();other=deepcopy(args);other[0]['a_cp_j_kg_k']*=1.2
    a=cached.CachedTwoStreamExchanger(*args);b=cached.CachedTwoStreamExchanger(*other)
    a.advance(.1);compare(b.advance(.1),TwoStreamExchanger(*other).advance(.1))
    assert a.cache_info()['misses']==b.cache_info()['misses']==1
