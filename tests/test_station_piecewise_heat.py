from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_piecewise_heat import step_piecewise_heat
from atomic_model_station_piecewise_heat import model


def make_env():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.01,automatic=False,rtol=1e-10,max_step_s=.01));env.reset();return env


@pytest.mark.parametrize('offset',[.000037,-.000037,.0001])
def test_rod_boundary_is_sampled_without_changing_native_transition(offset):
    env=make_env();native=deepcopy(env)
    action={'operator':{'rod_target':float(env._session.control.plant.state[12])+offset}}
    expected=native.step(action);actual,proof=step_piecewise_heat(env,action,subdivisions=(8,16,32) if offset==.0001 else (4,8,16))
    assert actual==expected and env.checkpoint()==native.checkpoint()
    arrival=proof['motion']['arrival_s']
    assert proof['passed'] and all(proof['checks'].values())
    for replay in proof['replays']:
        assert replay['event_boundaries_s']==[arrival]
        assert any(s['energy']['time_s']==arrival for s in replay['samples'])
        assert max(s['analytic_error'] for s in replay['rod_samples'])<=1e-12
        assert replay['samples'][-1]['energy']['time_s']==.01


def test_missing_native_event_is_not_silently_reconstructed(monkeypatch):
    env=make_env();before=env.checkpoint();real=StationEnv.step
    def remove(self,*args,**kwargs):
        result=real(self,*args,**kwargs)
        self._session.control.plant.events[:]=[e for e in self._session.control.plant.events if e['kind']!='rod_arrived']
        return result
    monkeypatch.setattr(StationEnv,'step',remove)
    with pytest.raises(HeatEventError) as exc:
        step_piecewise_heat(env,{'operator':{'rod_target':float(env._session.control.plant.state[12])+.000037}})
    assert exc.value.receipt['reason']=='rod_event_count' and env.checkpoint()==before


@pytest.mark.parametrize('mutation',['time','position','kind','duplicate'])
def test_corrupt_event_refuses_atomic_step(monkeypatch,mutation):
    env=make_env();before=env.checkpoint();real=StationEnv.step
    def corrupt(self,*args,**kwargs):
        result=real(self,*args,**kwargs);events=self._session.control.plant.events
        event=next(e for e in events if e['kind']=='rod_arrived')
        if mutation=='time':event['time_s']+=.001
        elif mutation=='position':event['position']+=.01
        elif mutation=='kind':event['kind']='backup_exhausted'
        else:events.append(deepcopy(event))
        return result
    monkeypatch.setattr(StationEnv,'step',corrupt)
    with pytest.raises(HeatEventError):
        step_piecewise_heat(env,{'operator':{'rod_target':float(env._session.control.plant.state[12])+.000037}})
    assert env.checkpoint()==before


def test_bad_endpoint_match_refuses_commit(monkeypatch):
    env=make_env();before=env.checkpoint();real=model._replay
    def corrupt(*args):
        row=real(*args);row['endpoint_state_relative']=.1;return row
    monkeypatch.setattr(model,'_replay',corrupt)
    with pytest.raises(HeatEventError) as exc:step_piecewise_heat(env,{})
    assert not exc.value.receipt['candidate']['checks']['endpoint_agreement']
    assert env.checkpoint()==before


def test_source_change_refuses_commit(monkeypatch):
    env=make_env();before=env.checkpoint();versions=iter([1,2])
    monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:step_piecewise_heat(env,{})
    assert exc.value.receipt['reason']=='source_changed' and env.checkpoint()==before


def test_default_solver_tolerance_can_refuse_endpoint_agreement():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.01,automatic=False));env.reset()
    before=env.checkpoint()
    with pytest.raises(HeatEventError) as exc:
        step_piecewise_heat(env,{'operator':{'rod_target':float(env._session.control.plant.state[12])+.000037}})
    assert not exc.value.receipt['candidate']['checks']['endpoint_agreement']
    assert env.checkpoint()==before
