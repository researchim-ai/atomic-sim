from copy import deepcopy
import numpy as np
import pytest
from atomic_model_station import PWRPowerSession
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_hybrid_heat import record_hybrid_window,step_hybrid_heat
from atomic_model_station_hybrid_heat import model


def make_env():
    env=StationEnv(EnvConfig(max_steps=4,tick_s=.02,automatic=False,rtol=1e-10));env.reset();return env


@pytest.mark.parametrize('pump',['feed_pump_on','condensate_pump_on'])
def test_instrumented_solver_matches_native_with_and_without_root(pump):
    session=PWRPowerSession(rtol=1e-10);native=deepcopy(session);recorded=model._copy_recorder(session)
    expected=native.advance(.02,{pump:False});actual=recorded.advance(.02,{pump:False})
    assert actual==expected and np.array_equal(recorded.state,native.state)
    assert bool(recorded.heat_events)==(pump=='feed_pump_on')
    assert all(abs(e['before']['stored_j']-e['after']['stored_j'])<=1e-3 for e in recorded.heat_events)


def test_environment_valve_close_preserves_transition_and_checkpoint():
    env=make_env();native=deepcopy(env);action={'operator':{'feed_pump_on':False}}
    expected=native.step(action);actual,proof=step_hybrid_heat(env,action,subdivisions=(4,8,16))
    assert actual==expected and env.checkpoint()==native.checkpoint()
    assert proof['passed'] and proof['native_state_committed']
    assert all([e['kind'] for e in r['events']]==['feed_check_closed'] for r in proof['replays'])


def test_endpoint_mismatch_refuses_environment_commit(monkeypatch):
    env=make_env();before=env.checkpoint();real=model._record
    def bad(*args):
        row=real(*args);row['checks']['endpoint_state']=False;row['passed']=False;return row
    monkeypatch.setattr(model,'_record',bad)
    with pytest.raises(HeatEventError) as exc:step_hybrid_heat(env,{},subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='hybrid_heat_refused'
    assert not exc.value.receipt['candidate']['native_state_committed']
    assert env.checkpoint()==before


def test_source_change_refuses_environment_commit(monkeypatch):
    env=make_env();before=env.checkpoint();versions=iter([1,2])
    monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:step_hybrid_heat(env,{},subdivisions=(2,4,8))
    assert exc.value.receipt['reason']=='source_changed' and env.checkpoint()==before


@pytest.mark.parametrize('counts',[(1,2,4),(4,8,32),(True,2,4),(128,256,512)])
def test_invalid_budget_does_not_advance(counts):
    env=make_env();before=env.checkpoint()
    with pytest.raises(ValueError):step_hybrid_heat(env,{},subdivisions=counts)
    assert env.checkpoint()==before


def test_unknown_native_event_cannot_hide_in_energy_balance(monkeypatch):
    env=make_env();before=env.checkpoint();real=StationEnv.step
    def injected(self,*args,**kwargs):
        result=real(self,*args,**kwargs)
        self._session.control.plant.events.append(dict(time_s=.01,kind='unqualified_relief'))
        return result
    monkeypatch.setattr(StationEnv,'step',injected)
    with pytest.raises(ValueError,match='Unsupported native event'):
        step_hybrid_heat(env,{},subdivisions=(2,4,8))
    assert env.checkpoint()==before


def test_direct_recorder_is_readonly():
    session=PWRPowerSession(rtol=1e-10);before=session.checkpoint()
    report=record_hybrid_window(session,.02,{'feed_pump_on':False},subdivisions=16)
    assert report['passed'] and session.checkpoint()==before


def test_root_residual_guard_is_not_bypassed():
    copy=model._copy_recorder(PWRPowerSession(rtol=1e-10))
    endpoint=copy.state.copy()
    endpoint[copy.plant.feed_index]=1.
    with pytest.raises(ValueError,match='zero-root residual'):
        copy._custom_root('feed_closed',endpoint,0.)


def test_raw_root_meter_matches_public_meter_on_admitted_state():
    env=make_env();session=env._session.control.plant;before=session.checkpoint()
    assert model._root_meter(session)==model._meter(session)
    assert session.checkpoint()==before
