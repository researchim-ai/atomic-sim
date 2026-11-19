from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_refined_heat import step_refined_heat
from atomic_model_station_refined_heat import model


def make_env():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.01,automatic=False));env.reset();return env


def test_dhr_refinement_preserves_native_step_and_checkpoint():
    env=make_env();native=deepcopy(env);action={'operator':{'dhr_requested':True}}
    expected=native.step(action);actual,proof=step_refined_heat(env,action)
    assert actual==expected and env.checkpoint()==native.checkpoint()
    assert proof['passed'] and all(proof['checks'].values())
    assert not proof['coarse_audit']['passed'] and proof['audit']['passed']
    errors=[r['audit']['heat_closure_relative'] for r in proof['replays']]
    assert errors[2]<errors[1]<errors[0]
    assert not proof['controller_substepped']


@pytest.mark.parametrize('counts',[(1,2,4),(4,8,32),(4,8),(True,2,4),(32,64,128)])
def test_invalid_refinement_budget_is_atomic(counts):
    env=make_env();before=env.checkpoint()
    with pytest.raises(ValueError,match='subdivision'):step_refined_heat(env,{},subdivisions=counts)
    assert env.checkpoint()==before


def test_insufficient_refinement_is_not_accepted():
    env=make_env();before=env.checkpoint()
    with pytest.raises(HeatEventError) as exc:
        step_refined_heat(env,{'operator':{'dhr_requested':True}},subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='refinement_failed'
    assert not exc.value.receipt['candidate']['native_state_committed']
    assert env.checkpoint()==before


def test_endpoint_disagreement_cannot_be_hidden_by_good_heat_balance(monkeypatch):
    env=make_env();before=env.checkpoint();real=model._replay
    def wrong(*args):
        row=real(*args);row['endpoint_state_relative']=.01;return row
    monkeypatch.setattr(model,'_replay',wrong)
    with pytest.raises(HeatEventError) as exc:step_refined_heat(env,{})
    assert not exc.value.receipt['candidate']['checks']['endpoint_agreement']
    assert env.checkpoint()==before


@pytest.mark.parametrize('epoch',[.005,.01,float('nan')])
def test_unresolved_events_are_refused_before_replay(monkeypatch,epoch):
    env=make_env();before=env.checkpoint();real=StationEnv.step
    def inject(self,*args,**kwargs):
        result=real(self,*args,**kwargs)
        self._session.control.plant.events.append(dict(time_s=epoch,kind='test'))
        return result
    monkeypatch.setattr(StationEnv,'step',inject)
    with pytest.raises(HeatEventError,match='refused'):step_refined_heat(env,{})
    assert env.checkpoint()==before


def test_source_change_rolls_back(monkeypatch):
    env=make_env();before=env.checkpoint();versions=iter([1,2])
    monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:step_refined_heat(env,{})
    assert exc.value.receipt['reason']=='source_changed'
    assert env.checkpoint()==before
