import numpy as np
import pytest
from atomic_model_plant_extension import water


@pytest.fixture(autouse=True)
def clear_raw_cache():
    water._raw_values.cache_clear()
    yield
    water._raw_values.cache_clear()


@pytest.mark.parametrize('p,t,region',[(1e5,300.,1),(15.7e6,563.15,1),
    (16.5e6,620.,1),(1e5,700.,2),(6.4e6,650.,2),(16.5e6,900.,2)])
def test_cached_values_equal_direct_formulation_exactly(p,t,region):
    direct=water._raw_uncached(p,t,region)
    assert water._raw(p,t,region)==direct
    assert water._raw(p,t,region)==direct
    assert water._raw_values.cache_info().hits==1


def test_one_ulp_pressure_temperature_and_region_are_distinct_keys():
    p,t=6.4e6,550.
    for args in [(p,t,1),(np.nextafter(p,np.inf),t,1),(p,np.nextafter(t,np.inf),1),(p,t,2)]:
        assert water._raw(*args)==water._raw_uncached(*args)
    assert water._raw_values.cache_info().misses==4
    assert water._raw_values.cache_info().hits==0


def test_mutation_of_returned_dictionary_cannot_poison_future_state():
    first=water._raw(6.4e6,550.,1);expected=dict(first)
    first['u']=0.;first.clear()
    assert water._raw(6.4e6,550.,1)==expected


def test_capacity_is_bounded_and_eviction_keeps_values_exact():
    expected=water._raw(1e6,400.,1)
    for i in range(2050):water._raw(2e6+i,400.,1)
    info=water._raw_values.cache_info()
    assert info.currsize==info.maxsize==2048
    assert water._raw(1e6,400.,1)==expected
    assert water._raw_values.cache_info().misses==info.misses+1


def test_exceptions_are_not_cached_or_replaced(monkeypatch):
    calls=[]
    def refused(*args):
        calls.append(args);raise ValueError('outside property chart')
    monkeypatch.setattr(water,'_raw_uncached',refused)
    for _ in range(2):
        with pytest.raises(ValueError,match='outside property chart'):water._raw(1e6,400.,1)
    assert len(calls)==2 and water._raw_values.cache_info().currsize==0


@pytest.mark.parametrize('p,t',[(3e6,400.),(15.7e6,563.15),(6.4e6,650.)])
def test_inverse_and_analytic_derivatives_match_with_cache_disabled(monkeypatch,p,t):
    state=water.pt(p,t)
    cached=water.rho_u(state.density_kg_m3,state.internal_energy_j_kg,p*.999,t+.01)
    derivative=water.derivatives(state)
    monkeypatch.setattr(water,'_raw',water._raw_uncached)
    direct=water.rho_u(state.density_kg_m3,state.internal_energy_j_kg,p*.999,t+.01)
    assert direct==cached
    assert np.array_equal(water.derivatives(state),derivative)


def test_qualification_switch_restores_cached_function_after_refusal():
    from scripts.qualify_station_water_cache import cache_mode
    original=water._raw
    with pytest.raises(RuntimeError):
        with cache_mode(False):
            assert water._raw is water._raw_uncached
            raise RuntimeError('solver refused')
    assert water._raw is original and water._raw_values.cache_info().currsize==0


def test_qualification_retains_failure_and_executes_later_cases(tmp_path,monkeypatch):
    import json
    from scripts import qualify_station_water_cache as driver
    calls=[]
    def execute(case,enabled):
        calls.append(case)
        if case=='hold':raise ValueError('native refused')
        return dict(scientific={'same':True},cache=dict(hits=int(enabled),currsize=int(enabled)),
            timing=dict(advance_s=1.))
    monkeypatch.setattr(driver,'execute',execute)
    with pytest.raises(ValueError,match='attempts retained'):driver.run(tmp_path/'study',repeats=1)
    report=json.loads((tmp_path/'study/report.json').read_text())['payload']
    assert report['attempts']==3 and report['accepted']==2 and report['failed']==1
    assert calls==['hold','feed-cycle','feed-cycle','pressure-cycle','pressure-cycle']


def test_qualification_does_not_accept_faster_different_result(tmp_path,monkeypatch):
    import json
    from scripts import qualify_station_water_cache as driver
    monkeypatch.setattr(driver,'execute',lambda case,enabled:dict(
        scientific={'changed':enabled},cache=dict(hits=int(enabled),currsize=int(enabled)),
        timing=dict(advance_s=.001 if enabled else 1.)))
    with pytest.raises(ValueError,match='attempts retained'):driver.run(tmp_path/'study',repeats=1)
    report=json.loads((tmp_path/'study/report.json').read_text())['payload']
    assert report['accepted']==0 and report['failed']==3
