import json
from copy import deepcopy
from pathlib import Path

import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station.journal import Journal,history_budget
from atomic_model_station_faults import Fault,FaultSession
from atomic_model_station_campaign import Episode,Limits,run_campaign
from atomic_model_station_campaign.engine import run_episode
from atomic_model_station_campaign.matrix import catalog,DURATIONS


def test_transaction_journal_does_not_share_mutable_rows_or_future_appends():
    row={'operator':{'x':1}}
    first=Journal([row]);row['operator']['x']=2
    trial=deepcopy(first);trial.append({'operator':{'x':3}})
    returned=list(trial);returned[0]['operator']['x']=9
    assert list(first)==[{'operator':{'x':1}}]
    assert list(trial)==[{'operator':{'x':1}},{'operator':{'x':3}}]
    first.append({'operator':{'x':4}})
    assert list(first)[-1]['operator']['x']==4 and list(trial)[-1]['operator']['x']==3


def test_large_journal_clones_and_serializes_without_recursive_copy_or_row_aliases():
    journal=Journal({'i':i} for i in range(8192));clone=deepcopy(journal)
    clone.append({'i':8192})
    assert len(journal)==8192 and len(clone)==8193
    assert json.loads(json.dumps(list(clone)))[-1]=={'i':8192}


@pytest.mark.parametrize('value',[True,255,8193,256.,-1])
def test_history_budget_refuses_unsafe_or_ambiguous_configuration(value):
    with pytest.raises(ValueError):history_budget(value)


@pytest.mark.parametrize('kw',[dict(workers=0),dict(workers=5),dict(wall_s=True),dict(cpu_s=0),dict(memory_mb=1),dict(output_mb=0)])
def test_resource_contract_is_explicit_and_bounded(kw):
    with pytest.raises(ValueError):Limits(**kw)


@pytest.mark.parametrize('kw',[dict(ticks=True),dict(ticks=8193),dict(ticks=2000,tick_s=5.),dict(tick_s=.001),
    dict(record_every=0),dict(checkpoint_ticks=(1,1)),dict(checkpoint_ticks=(-1,)),
    dict(commands=({'tick':1},{'tick':1})),dict(commands=({'tick':0,'operator':{'x':float('nan')}},)),
    dict(tracking_tau_s=.1),dict(automatic=1)])
def test_episode_rejects_invalid_horizon_schedule_or_recorder(kw):
    params=dict(name='case',ticks=2);params.update(kw)
    with pytest.raises(ValueError):Episode(**params)


def test_manifest_roundtrip_preserves_declared_faults_and_command_order():
    e=Episode('case',4,faults=(Fault('f','bias','pressure',2,value=150000.),),
        commands=({'tick':3,'operator':{'load_fraction':1.}}, {'tick':1,'operator':{'load_fraction':.99}}),checkpoint_ticks=(3,0))
    rebuilt=Episode.from_payload(json.loads(json.dumps(e.payload(),sort_keys=True)))
    assert rebuilt.payload()==e.payload() and rebuilt.commands[0]['tick']==1


def test_frozen_r1_catalog_contains_full_horizons_and_three_refinements():
    p=unseal(catalog());assert len(p['families'])==12 and not p['full_r1_accepted']
    for family in p['families']:
        assert family['duration_s']==DURATIONS[family['name']]
        episodes=[Episode.from_payload(e) for e in family['variants']]
        assert len(episodes)==3 and {e.max_step_s for e in episodes}=={1.,.5,.25}
        assert all(e.ticks*e.tick_s==family['duration_s'] for e in episodes)
        assert all(family['restart_tick'] in e.checkpoint_ticks for e in episodes)


def test_short_episode_anchor_replay_reconstructs_whole_state_and_trajectory():
    e=Episode('recover',3,tick_s=.01,automatic=False,checkpoint_ticks=(1,2))
    normal=run_episode(e);anchor=unseal(normal)['checkpoints']['1']
    replay=run_episode(e,json.loads(json.dumps(anchor,sort_keys=True)))
    assert normal==replay
    bad=unseal(anchor);bad['trajectory_sha256']='0'*64
    with pytest.raises(ValueError,match='trajectory'):run_episode(e,seal(bad))


def test_rejected_command_is_audited_and_never_hidden_as_full_horizon():
    e=Episode('bad_input',3,tick_s=.01,automatic=False,commands=({'tick':1,'operator':{'unknown_valve':.5}},),expected_rejection=True)
    r=unseal(run_episode(e))
    assert r['accepted'] and r['status']=='rejected' and r['rollback_exact']
    assert r['completed_ticks']==1 and r['completed_time_s']<.03
    ordinary=Episode.from_payload(dict(e.payload(),expected_rejection=False))
    assert not unseal(run_episode(ordinary))['accepted']


def test_extended_budget_is_opt_in_and_propagates_to_controller_and_physics():
    a=FaultSession();b=FaultSession(history_budget_steps=8192)
    assert a.control.plant._maximum_history_steps()==256
    assert b.control.plant._maximum_history_steps()==8192
    assert b.step()['receipt']['sequence']==0
    saved=b.checkpoint();assert FaultSession.restore(saved).checkpoint()==saved


def test_parallel_workers_have_separate_processes_and_exact_scientific_results(tmp_path):
    episodes=[Episode('a',1,tick_s=.01,automatic=False),Episode('b',1,tick_s=.01,automatic=False,seed=7)]
    one=run_campaign(tmp_path/'one',episodes,limits=Limits(workers=1,wall_s=60,cpu_s=60))
    two=run_campaign(tmp_path/'two',episodes,limits=Limits(workers=2,wall_s=60,cpu_s=60))
    assert one['accepted'] and two['accepted']
    processes=[]
    for episode in episodes:
        a=unseal(json.loads((tmp_path/'one'/(episode.name+'.json')).read_text()))
        b=unseal(json.loads((tmp_path/'two'/(episode.name+'.json')).read_text()))
        assert a['scientific']==b['scientific']
        assert b['execution']['thread_environment']==dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        processes.append(b['execution']['pid'])
    assert len(set(processes))==2


def test_invalid_campaign_has_no_output_side_effect(tmp_path):
    output=tmp_path/'none'
    with pytest.raises(ValueError):run_campaign(output,[Episode('same',1),Episode('same',1)])
    assert not output.exists()
