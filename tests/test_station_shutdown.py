from copy import deepcopy
import json
import pytest
from core.status import DomainError
from atomic_model_station_shutdown import ShutdownCase,assess
from scripts import qualify_station_shutdown as study
from scripts.qualify_station_inventory import read


@pytest.mark.parametrize('kwargs',[dict(family='unknown'),dict(family=True),dict(tick_s=True),dict(duration_s=601),
 dict(duration_s=599),dict(duration_s=0),dict(dhr_requested=1),dict(tick_s=.001)])
def test_shutdown_contract_refuses_ambiguous_or_outside_horizons(kwargs):
    with pytest.raises(ValueError):ShutdownCase(**kwargs)


def test_required_horizons_and_coastdown_command_are_explicit():
    assert ShutdownCase().duration_s==600 and ShutdownCase('coastdown').duration_s==1800
    assert ShutdownCase('coastdown').command()['primary_pump_on'] is False
    assert ShutdownCase().command()['dhr_requested'] and ShutdownCase().full_horizon
    assert not ShutdownCase(duration_s=10).full_horizon


@pytest.fixture(scope='module')
def run_data(tmp_path_factory):
    p=tmp_path_factory.mktemp('shutdown-short');c=ShutdownCase(tick_s=.01,duration_s=.04)
    normal=study.run(p/'normal',c,max_step_s=.1)
    native=study.run(p/'native',c,max_step_s=.1,restart=True,reference=p/'normal')
    assert normal['passed'] and native['passed'] and native['native_prefix_exact'] and native['native_whole_path_exact']
    return p,c,read(p/'normal/initial.json')['record'],read(p/'normal/trajectory.json')['records']


def test_real_short_shutdown_preserves_latches_dhr_and_native_replay(run_data):
    p,c,initial,rows=run_data;r=read(p/'normal/report.json')
    assert r['passed'] and not r['assessment']['full_horizon'] and not r['npp_11_03_accepted']
    assert r['assessment']['dhr_transferred_j']>0 and r['closed_tasks']==[]
    assert not r['assessment']['shutdown_response_reached']


@pytest.mark.parametrize('kind',['trip','reserve','dhr','entropy','budget','clock'])
def test_midpath_violations_are_not_hidden_by_final_state(run_data,kind):
    _,c,initial,rows=run_data;bad=deepcopy(rows);p=bad[1]['physical']
    if kind=='trip':p['actuator_status']['scram_latched']=False
    if kind=='reserve':p['feed_train']['makeup_flow_kg_s']=1.
    if kind=='dhr':p['electrical']['dhr_active']=False
    if kind=='entropy':p['electrical']['dhr_entropy_generation_w_k']=-100.
    if kind=='budget':p['mass_relative']=1e-3
    if kind=='clock':p['time_s']=99.
    assert not assess(initial,bad,c)['passed']
    assert not assess(initial,rows[:-1],c)['passed']


def test_domain_failure_is_recorded_as_partial_and_exactly_rolled_back(tmp_path,monkeypatch):
    original=study.FaultSession.step
    def refuses(self,*a,**k):
        if len(self.history)==1:raise DomainError('deliberate unsupported closure')
        return original(self,*a,**k)
    monkeypatch.setattr(study.FaultSession,'step',refuses)
    r=study.run(tmp_path/'refused',ShutdownCase(tick_s=.01,duration_s=.04),max_step_s=.1)
    assert not r['passed'] and not r['completed'] and r['completed_time_s']==.01 and r['rollback_exact']
    assert r['failure']['category']=='domain' and r['closed_tasks']==[]


def test_no_requested_dhr_has_no_free_heat(tmp_path):
    c=ShutdownCase(tick_s=.01,duration_s=.02,dhr_requested=False)
    r=study.run(tmp_path/'without',c,max_step_s=.1)
    assert r['passed'] and r['assessment']['dhr_transferred_j']==0


@pytest.mark.parametrize('field',['heat','fission','momentum'])
def test_nonfinite_derived_physics_never_passes(run_data,field):
    _,c,initial,rows=run_data;bad=deepcopy(rows);p=bad[-1]['physical']
    if field=='heat':p['powers']['nuclear_heat_w']=float('nan')
    if field=='fission':p['core']['fission_total_w']=float('nan')
    if field=='momentum':p['primary']['momentum_residual_pa']=float('nan')
    report=assess(initial,bad,c);assert not report['passed'];json.dumps(report,allow_nan=False)


def test_requested_dhr_cools_donor_and_heats_the_same_finite_receiver(tmp_path):
    for name,active in [('with',True),('without',False)]:
        r=study.run(tmp_path/name,ShutdownCase(tick_s=.01,duration_s=.02,dhr_requested=active),max_step_s=.1)
        assert r['passed']
    a=read(tmp_path/'with/trajectory.json')['records'][-1]['physical']['electrical']
    b=read(tmp_path/'without/trajectory.json')['records'][-1]['physical']['electrical']
    assert a['dhr_donor_temperature_k']<b['dhr_donor_temperature_k']
    assert a['dhr_receiving_temperature_k']>b['dhr_receiving_temperature_k']
