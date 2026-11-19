from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import json
import pytest
from atomic_model_station_env import EnvConfig
from scripts import qualify_station_precision as study


def test_probe_records_returned_iterations_and_actual_linear_calls_without_changing_result(monkeypatch):
    result=(True,3,object(),None)
    def collocation(fun,t,y,h,Z0,scale,tol,LU_real,LU_complex,solve_lu):
        solve_lu(LU_real,None);solve_lu(LU_complex,None);return result
    solver_result=SimpleNamespace(nfev=7,njev=1,nlu=2,t=[0.,1.])
    original=lambda *a,**k:solver_result
    monkeypatch.setattr(study.radau,'solve_collocation_system',collocation)
    monkeypatch.setattr(study.pwr_train,'solve_ivp',original)
    with study.SolverProbe() as probe:
        assert study.radau.solve_collocation_system(*([None]*9),lambda *a:None) is result
        assert study.pwr_train.solve_ivp() is solver_result
    assert probe.counts['newton_iterations']==3 and probe.counts['linear_solve_calls']==2
    assert probe.counts['nlu']==2 and probe.counts['segments']==1 and probe.counts['iteration_count_complete']
    assert study.radau.solve_collocation_system is collocation and study.pwr_train.solve_ivp is original


def test_hook_restoration_on_error_and_nested_probe_refusal():
    a,b=study.radau.solve_collocation_system,study.pwr_train.solve_ivp
    with pytest.raises(RuntimeError):
        with study.SolverProbe():
            with pytest.raises(ValueError,match='Nested'):study.SolverProbe().__enter__()
            raise RuntimeError('measurement failed')
    assert study.radau.solve_collocation_system is a and study.pwr_train.solve_ivp is b and not study._ACTIVE


def test_unsupported_private_scipy_signature_is_refused_before_patching(monkeypatch):
    def changed(x):return x
    original=study.pwr_train.solve_ivp;monkeypatch.setattr(study.radau,'solve_collocation_system',changed)
    with pytest.raises(ValueError,match='schema'):study.SolverProbe().__enter__()
    assert study.pwr_train.solve_ivp is original and not study._ACTIVE


def test_unreturned_iteration_count_is_marked_incomplete(monkeypatch):
    def collocation(fun,t,y,h,Z0,scale,tol,LU_real,LU_complex,solve_lu):raise ValueError('RHS failed')
    monkeypatch.setattr(study.radau,'solve_collocation_system',collocation)
    with study.SolverProbe() as probe:
        with pytest.raises(ValueError,match='RHS failed'):study.radau.solve_collocation_system(*([None]*10))
    assert not probe.counts['iteration_count_complete'] and probe.counts['collocation_exceptions']==1


def test_real_instrumented_and_native_trajectories_and_checkpoint_match_exactly():
    original=study.execute('hold','fast_candidate');measured=study.execute('hold','fast_candidate',instrumented=True)
    assert original['scientific']==measured['scientific']
    counts=measured['phases']['advance']['solver_counts']
    assert counts['newton_iterations']>0 and counts['linear_solve_calls']>0 and counts['nlu']>0
    assert counts['iteration_count_complete']
    reference=study.execute('hold','reference')
    assert study.compare(original['scientific'],reference['scientific'])['passed']


def sample(case='hold',mode='fast_candidate'):
    cfg=asdict(EnvConfig(max_steps=1,tick_s=.05,automatic=False,**study.SETTINGS[mode]))
    receipt=dict(trip_latched=False,trip_reasons=[],pressure_interlock=False,modes={'rod':'manual'},targets={'rod':1.})
    point=lambda t:dict(time_s=t,state=[1.,2.],audits=dict(energy_relative=0.,mass_relative=0.,boron_relative=0.),events=[])
    science=dict(case=case,setting=mode,config=cfg,completed=True,failure=None,restore_exact=True,
        initial=point(0.),records=[dict(transition=({},0.,False,True,dict(receipt=receipt)),physical=point(.05))])
    if case=='invalid-command':
        science.update(completed=False,records=[],failure=dict(phase='advance',type='ActionError',code='invalid_action',message='invalid'))
    return science


@pytest.mark.parametrize('change',['state','event','trip','budget'])
def test_accuracy_or_discrete_difference_fails_preset(change):
    a=sample();b=sample(mode='reference')
    if change=='state':a['records'][0]['physical']['state'][0]+=.1
    if change=='event':a['records'][0]['physical']['events']=[dict(kind='new-root',time_s=.01)]
    if change=='trip':a['records'][0]['transition'][4]['receipt']['trip_latched']=True
    if change=='budget':a['records'][0]['physical']['audits']['energy_relative']=.1
    assert not study.compare(a,b)['passed']


def test_partial_result_has_no_accuracy_metric_and_mismatched_physics_is_refused():
    a=sample();b=sample(mode='reference');a['completed']=False
    result=study.compare(a,b)
    assert not result['comparable'] and result['state_relative'] is None
    a=sample();a['config']['seed']=1
    with pytest.raises(ValueError,match='Same physical'):study.compare(a,b)


def fake_execute(case,mode,*,instrumented=False):
    metrics=dict(wall_s=.01,profile=None,solver_counts=dict(iteration_count_complete=True) if instrumented else None)
    return dict(scientific=sample(case,mode),phases={p:deepcopy(metrics) for p in ('initialize','advance','checkpoint','serialize','restore')})


def test_fixed_matrix_keeps_input_refusals_and_does_not_rank_their_speed(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'CASES',{'hold':[{}],'invalid-command':[{}]})
    monkeypatch.setattr(study,'execute',fake_execute)
    report=study.run(tmp_path/'out')
    assert report['passed'] and report['summary']==dict(attempted=18,completed=9,failed=9)
    assert report['timing']['invalid-command']['detailed_to_fast_ratio'] is None
    assert report['timing']['invalid-command']['fast_candidate']['advance_wall_per_simulated_s'] is None


def test_unexpected_setup_failure_does_not_disappear_or_stop_later_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'CASES',{'hold':[{}],'invalid-command':[{}]});seen=[]
    def fail(case,mode,**kwargs):
        seen.append((case,mode))
        if case=='hold' and mode=='fast_candidate':raise RuntimeError('setup unavailable')
        return fake_execute(case,mode,**kwargs)
    monkeypatch.setattr(study,'execute',fail)
    with pytest.raises(ValueError,match='all attempts retained'):study.run(tmp_path/'out')
    report=json.loads((tmp_path/'out/report.json').read_text())['payload']
    assert len(seen)==report['summary']['attempted']==18
    assert report['summary']['failed']==12 and report['attempts'][-1]['case']=='invalid-command'
    assert report['timing']['hold']['detailed_to_fast_ratio'] is None


def test_missing_iteration_counts_cannot_pass_as_complete_measurements(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'CASES',{'hold':[{}]})
    def incomplete(*args,**kwargs):
        r=fake_execute(*args,**kwargs)
        if kwargs.get('instrumented'):r['phases']['advance']['solver_counts']['iteration_count_complete']=False
        return r
    monkeypatch.setattr(study,'execute',incomplete)
    with pytest.raises(ValueError):study.run(tmp_path/'out')
    report=json.loads((tmp_path/'out/report.json').read_text())['payload']
    assert not report['passed']


@pytest.mark.parametrize('repeat',[True,1,6])
def test_invalid_compute_budget_creates_no_output(tmp_path,repeat):
    with pytest.raises(ValueError):study.run(tmp_path/'out',repeats=repeat)
    assert not (tmp_path/'out').exists()


def test_source_change_refuses_report(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'CASES',{'hold':[{}]});monkeypatch.setattr(study,'execute',fake_execute)
    identities=iter([1,2]);monkeypatch.setattr(study,'identity',lambda:next(identities))
    with pytest.raises(ValueError):study.run(tmp_path/'out')
    assert not json.loads((tmp_path/'out/report.json').read_text())['payload']['checks']['source-unchanged']
