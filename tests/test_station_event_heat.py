from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig,ActionError
from atomic_model_station_faults import Fault
from atomic_model_station_event_heat import HeatEventError,step_heat,run_event_window
from atomic_model_station_event_heat import model


def env():
    e=StationEnv(EnvConfig(max_steps=4,tick_s=.01,automatic=False));e.reset();return e


def test_load_switch_matches_native_step_exactly_and_does_not_smear_flux():
    wrapped=env();native=deepcopy(wrapped);action={'operator':{'load_fraction':.95}}
    expected=native.step(action);actual,proof=step_heat(wrapped,action)
    assert actual==expected and wrapped.checkpoint()==native.checkpoint()
    assert proof['audit']['passed'] and not proof['naive_unsplit_audit']['passed']
    assert proof['left']['energy']['counters']==proof['prior']['energy']['counters']
    assert proof['left']['energy']['powers']['net_export_w']!=proof['prior']['energy']['powers']['net_export_w']


def test_dhr_unresolved_quadrature_is_refused_without_losing_diagnostics():
    e=env();before=e.checkpoint()
    with pytest.raises(HeatEventError) as exc:
        step_heat(e,{'operator':{'dhr_requested':True}})
    assert exc.value.receipt['reason']=='segment_heat_budget'
    p=exc.value.receipt['candidate']
    assert 'sampled_heat_closure' in p['audit']['reasons']
    assert e.checkpoint()==before
    assert p['left']['stored_j']==p['prior']['stored_j']
    assert p['left']['energy']['powers']['own_needs_w']-p['prior']['energy']['powers']['own_needs_w']==pytest.approx(1e5)


def test_invalid_action_does_not_commit():
    e=env();before=e.checkpoint()
    with pytest.raises(ActionError):step_heat(e,{'operator':{'load_fraction':99}})
    assert e.checkpoint()==before


def test_failed_audit_rolls_back_actual_candidate_step(monkeypatch):
    e=env();before=e.checkpoint();real=model.assess_heat
    def refuse(samples):
        result=real(samples);result['passed']=False;return result
    monkeypatch.setattr(model,'assess_heat',refuse)
    with pytest.raises(HeatEventError) as exc:step_heat(e,{})
    assert exc.value.receipt['reason']=='segment_heat_budget'
    assert exc.value.receipt['candidate']['right']['energy']['time_s']==.01
    assert e.checkpoint()==before


def test_source_change_does_not_commit_candidate(monkeypatch):
    e=env();before=e.checkpoint();versions=iter([{'v':1},{'v':2}])
    monkeypatch.setattr(model,'identity',lambda:next(versions))
    with pytest.raises(HeatEventError) as exc:step_heat(e,{})
    assert exc.value.receipt['reason']=='source_changed' and e.checkpoint()==before


@pytest.mark.parametrize("event_time", [.005, .01])
def test_unknown_internal_event_is_refused_atomically(monkeypatch,event_time):
    e=env();before=e.checkpoint();original=StationEnv.step
    def injected(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        self._session.control.plant.events.append(dict(time_s=event_time,kind='unresolved_test_event'))
        return result
    monkeypatch.setattr(StationEnv,'step',injected)
    with pytest.raises(HeatEventError) as exc:step_heat(e,{})
    assert exc.value.receipt['reason']=='unresolved_internal_event' and e.checkpoint()==before


def test_scheduled_fault_scope_is_explicit():
    e=StationEnv(EnvConfig(max_steps=2,tick_s=.01,automatic=False,faults=(Fault('x','bias','pressure',1,value=1.),)))
    e.reset()
    with pytest.raises(ValueError,match='Scheduled faults'):step_heat(e,{})


def test_window_retains_prefix_when_later_command_is_invalid():
    result=run_event_window(EnvConfig(max_steps=3,tick_s=.01,automatic=False),{1:{'operator':{'load_fraction':99}}})
    assert not result['passed'] and not result['complete'] and len(result['records'])==1
    assert result['failure']['code']=='invalid_action'


def test_checkpoint_continuation_with_switches_is_exact():
    e=env();step_heat(e,{'operator':{'load_fraction':.95}})
    restored=StationEnv.restore(e.checkpoint())
    a,p=step_heat(e,{'operator':{'load_fraction':1.0}})
    b,q=step_heat(restored,{'operator':{'load_fraction':1.0}})
    assert a==b and p==q and e.checkpoint()==restored.checkpoint()


def test_initialization_failure_is_a_failed_window(monkeypatch):
    def fail(self):raise RuntimeError('initialization failed')
    monkeypatch.setattr(StationEnv,'reset',fail)
    result=run_event_window(EnvConfig(max_steps=2,tick_s=.01,automatic=False),{})
    assert not result['passed'] and not result['complete']
    assert result['records']==[] and result['failure']['type']=='RuntimeError'
