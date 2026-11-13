from copy import deepcopy
import numpy as np
import pytest
from atomic_model_thermal_agent import ThermalConfig
from scripts import train_thermal_recovery as recovery


def rows():
    return [dict(candidate=k,case=c,scenario=list(s),seed=recovery.TRAIN_SEED,split='train',completed=True,score=.5)
        for k in recovery.CANDIDATES for c in range(2) for s in recovery.TRAIN_SCENARIOS]


def test_selection_uses_worst_group_not_best_average():
    data=rows()
    for r in data:
        if r['candidate']=='continue':r['score']=.8 if r['scenario'][0]=='bias' else .01
        elif r['candidate']=='ten':r['score']=.4
        else:r['score']=.6
    selection=recovery.select(data)
    assert selection['selected']=='ten' and selection['test_used'] is False
    assert recovery.select(list(reversed(data)))==selection

@pytest.mark.parametrize('damage',['missing','duplicate','test_leak','unknown_fault','nan','boolean'])
def test_selection_refuses_bad_provenance_or_scores(damage):
    data=rows()
    if damage=='missing':data.pop()
    elif damage=='duplicate':data.append(deepcopy(data[0]))
    elif damage=='test_leak':data[0]['split']='test'
    elif damage=='unknown_fault':data[0]['scenario']=['bias',-.22,3]
    elif damage=='nan':data[0]['score']=float('nan')
    else:data[0]['score']=True
    with pytest.raises(ValueError):recovery.select(data)


def test_incomplete_candidates_never_win_and_all_failed_refused():
    data=rows()
    for r in data:r['completed']=False
    with pytest.raises(ValueError):recovery.select(data)
    for r in data:
        if r['candidate']=='continue':r['completed']=True;r['score']=100.
    assert recovery.select(data)['selected']=='continue'

@pytest.mark.parametrize('candidate,expected',[('continue',.7),('zero',0.),('four',.04),('ten',.1),('twenty',.2)])
def test_recovery_action_has_no_fault_identity_input(candidate,expected):
    assert recovery.recovery_action(candidate,.7,False)==.7
    assert recovery.recovery_action(candidate,.7,True)==expected


def test_new_fault_sizes_and_onsets_do_not_overlap_training():
    train={(s[0],s[1],s[2]) for s in recovery.TRAIN_SCENARIOS if s[0] not in ('healthy','model_mismatch')}
    test={(s[0],s[1],s[2]) for s in recovery.TEST_SCENARIOS if s[0] not in ('healthy','model_mismatch')}
    assert train.isdisjoint(test)
    assert recovery.TRAIN_SEED not in recovery.TEST_SEEDS
    assert {s[2] for s in recovery.TRAIN_SCENARIOS if s[0]=='stuck'}!={s[2] for s in recovery.TEST_SCENARIOS if s[0]=='stuck'}


def test_model_mismatch_changes_monitor_assumption_not_plant_or_sensor():
    cfg=ThermalConfig(steps=4,action_dt_s=1e-5)
    a=recovery.RecoveryEnv(cfg,('healthy',0.,2));b=recovery.RecoveryEnv(cfg,('model_mismatch',0.,2))
    x,_=a.reset(seed=1);y,_=b.reset(seed=1);np.testing.assert_array_equal(x,y)
    for _ in range(3):
        x=a.step([.1]);y=b.step([.1]);np.testing.assert_array_equal(x[0],y[0]);assert x[4]==y[4]
    assert recovery.monitor_config(cfg,('model_mismatch',0.,2),'train').capacity_min==100.
    assert recovery.monitor_config(cfg,('model_mismatch',0.,2),'test').capacity_min==500.

@pytest.mark.parametrize('scenario',[('bias',.6,2),('bias',.1,True),('unknown',0.,2)])
def test_bad_recovery_scenario_refused(scenario):
    with pytest.raises(ValueError):recovery.RecoveryEnv(ThermalConfig(steps=4),scenario)


def test_real_valued_training_aggregation_is_exactly_order_independent():
    data=rows();rng=np.random.default_rng(71)
    for r in data:r['score']=float(rng.uniform(.2,1.3))
    result=recovery.select(data)
    assert recovery.select(list(reversed(data)))==result
    rng.shuffle(data)
    assert recovery.select(data)==result
