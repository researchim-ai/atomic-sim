from copy import deepcopy
import pytest
from atomic_model_decision import evaluate,commit
from atomic_model_decision import planning
from atomic_model_hem_heated.model import canonical
from atomic_model_multiphysics.provenance import seal,unseal
from scripts.verify_agent_decisions import setup,execute

@pytest.fixture(scope='module')
def decision():
    model,candidates,target,limits=setup('wall_control')
    return model,evaluate(model,candidates,target=target,limits=limits)


def test_decision_explains_rejections_and_never_changes_live_state(decision):
    model,saved=decision;payload=unseal(saved)
    assert canonical(model.checkpoint())==canonical(payload['base_checkpoint'])
    assert payload['selected_id']=='off'
    rows={r['id']:r for r in payload['results']}
    assert rows['off']['status']=='eligible' and rows['cool']['status']=='eligible'
    assert rows['high']['status']=='constraint_violated'
    assert rows['high']['violations'][0]['unit']=='K'
    assert 'Observation boundaries' in payload['constraint_scope']

@pytest.mark.parametrize('case',['wall_control','thermal_domain'])
@pytest.mark.parametrize('scheme',['rusanov','hllc','hllc-local'])
def test_real_network_branches_and_commit_replay(case,scheme):
    assert execute(case,scheme)['passed']

@pytest.mark.parametrize('damage',['duplicate','unknown_wall','nan_target','bool_limit','nan_control','unknown_control',
 'empty','too_many','zero_duration','unequal_horizon','different_grid','too_many_steps','oversize','wrong_model'])
def test_invalid_planning_contract_refused_without_mutation(damage):
    model,candidates,target,limits=setup('wall_control');before=model.checkpoint()
    if damage=='duplicate':candidates[1]['id']=candidates[0]['id']
    elif damage=='unknown_wall':target['wall']='absent'
    elif damage=='nan_target':target['temperature_k']=float('nan')
    elif damage=='bool_limit':limits['wall_temperature_max_k']=True
    elif damage=='nan_control':candidates[0]['steps'][0]['controls']['walls']['wall']=float('nan')
    elif damage=='unknown_control':candidates[0]['steps'][0]['controls']['fluid']['unknown']=0.
    elif damage=='empty':candidates=[]
    elif damage=='too_many':candidates*=6
    elif damage=='zero_duration':candidates[0]['steps'][0]['duration_s']=0.
    elif damage=='unequal_horizon':candidates[0]['steps'][0]['duration_s']=1e-5
    elif damage=='different_grid':
        candidates[0]['steps'][0]['duration_s']=2.5e-5;candidates[0]['steps'][1]['duration_s']=7.5e-5
    elif damage=='too_many_steps':candidates[0]['steps']*=5
    elif damage=='oversize':candidates[0]['id']='x'*49
    else:
        with pytest.raises(ValueError):evaluate(object(),candidates,target=target,limits=limits)
        return
    with pytest.raises(ValueError):evaluate(model,candidates,target=target,limits=limits)
    assert canonical(model.checkpoint())==canonical(before)

@pytest.mark.parametrize('field',['metric','selected','checkpoint','observations'])
def test_resealed_tampering_refused_by_reexecution(decision,field):
    model,saved=decision;before=model.checkpoint();payload=unseal(deepcopy(saved))
    if field=='metric':payload['results'][0]['metrics']['tracking_error_k_s']=0.
    elif field=='selected':payload['selected_id']='cool'
    elif field=='checkpoint':payload['results'][0]['checkpoint']['payload']['coordinates'][0]+=1e-9
    else:payload['results'][0]['observations'][-1]['walls'][0]['temperature_k']+=1e-5
    with pytest.raises(ValueError,match='replay differs'):commit(model,seal(payload))
    assert canonical(model.checkpoint())==canonical(before)


def test_no_eligible_action_is_an_explicit_result():
    model,candidates,target,limits=setup('wall_control');limits['wall_temperature_max_k']=450.
    saved=evaluate(model,candidates,target=target,limits=limits)
    assert unseal(saved)['selected_id'] is None
    before=model.checkpoint()
    with pytest.raises(ValueError,match='No eligible'):commit(model,saved)
    assert canonical(model.checkpoint())==canonical(before)


def test_live_change_during_planning_refuses_commit_without_undoing_other_change(monkeypatch):
    model,candidates,target,limits=setup('wall_control');saved=evaluate(model,candidates,target=target,limits=limits)
    real=planning.evaluate;advanced=[]
    def concurrent(*args,**kwargs):
        result=real(*args,**kwargs);model.advance(1e-5);advanced.append(model.checkpoint());return result
    monkeypatch.setattr(planning,'evaluate',concurrent)
    with pytest.raises(ValueError,match='advanced during planning'):commit(model,saved)
    assert canonical(model.checkpoint())==canonical(advanced[0])


def test_broken_digest_refused(decision):
    model,saved=decision;bad=deepcopy(saved);bad['sha256']='0'*64;before=model.checkpoint()
    with pytest.raises(ValueError):commit(model,bad)
    assert canonical(model.checkpoint())==canonical(before)
