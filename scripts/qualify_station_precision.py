"""Paired numerical presets, real Radau work counts and exclusive-time profiles.

Instrumentation is process-global and must run in an exclusive benchmark
process. Only uninstrumented trials contribute to timing comparisons.
"""
import argparse
import cProfile
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import pstats
import statistics
import time
import numbers

import numpy as np
from scipy.integrate._ivp import radau
from atomic_model_station import pwr_train
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_env.model import signature
from atomic_model_plant_extension import water
from physics.materials import iapws
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.profile_station_matrix import machine,summarize
from scripts import profile_station_matrix as cost_model

SETTINGS={'fast_candidate':dict(rtol=1e-7,max_step_s=.1),
          'detailed':dict(rtol=2e-8,max_step_s=.05),
          'reference':dict(rtol=5e-9,max_step_s=.025)}
CASES={
    'hold':[{},{}],
    'load':[{'operator':{'load_fraction':.99}},{'operator':{'load_fraction':1.}}],
    'feed-cycle':[{'operator':{'feed_pump_on':False}},{'operator':{'feed_pump_on':True}},{}],
    'condensate-cycle':[{'operator':{'condensate_pump_on':False}},{'operator':{'condensate_pump_on':True}},{}],
    'dhr':[{'operator':{'dhr_requested':True}},{'operator':{'dhr_requested':False}},{}],
    'rod-motion':[{'operator':{'rod_target':.251}},{'operator':{'rod_target':.25}},{}],
    'scram':[{'operator':{'scram':True}},{}],
    'invalid-command':[{'operator':{'load_fraction':99.}}],
}
CRITERIA=dict(state_relative=1e-3,event_time_s=1e-3,energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10)
_ACTIVE=False


def identity():
    return dict(station=signature(),driver_sha256=sha(__file__),
        cost_helper_sha256=sha(cost_model.__file__),
        radau_sha256=sha(radau.__file__))


class SolverProbe(AbstractContextManager):
    def __init__(self):
        self.counts=dict(segments=0,nfev=0,njev=0,nlu=0,accepted_nodes=0,
            newton_calls=0,newton_iterations=0,newton_nonconverged=0,
            linear_solve_calls=0,collocation_exceptions=0,iteration_count_complete=True)

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE:raise ValueError('Nested/concurrent solver instrumentation unsupported')
        self.original_collocation=radau.solve_collocation_system;self.original_ivp=pwr_train.solve_ivp
        self.signature=inspect.signature(self.original_collocation)
        if set(self.signature.parameters)!={'fun','t','y','h','Z0','scale','tol','LU_real','LU_complex','solve_lu'}:
            raise ValueError('Unsupported installed SciPy Radau instrumentation schema')
        def collocation(*args,**kwargs):
            bound=self.signature.bind(*args,**kwargs);original=bound.arguments['solve_lu']
            def linear(*a,**k):
                self.counts['linear_solve_calls']+=1
                return original(*a,**k)
            bound.arguments['solve_lu']=linear;self.counts['newton_calls']+=1
            try:
                result=self.original_collocation(*bound.args,**bound.kwargs)
                if len(result)!=4 or isinstance(result[1],bool) or not isinstance(result[1],numbers.Integral) or not 1<=result[1]<=100:
                    raise ValueError('Unsupported installed SciPy Newton result schema')
                self.counts['newton_iterations']+=int(result[1])
                self.counts['newton_nonconverged']+=int(not result[0])
                return result
            except Exception:
                self.counts['collocation_exceptions']+=1;self.counts['iteration_count_complete']=False
                raise
        def ivp(*args,**kwargs):
            result=self.original_ivp(*args,**kwargs);self.counts['segments']+=1
            for key in ('nfev','njev','nlu'):self.counts[key]+=int(getattr(result,key))
            self.counts['accepted_nodes']+=len(result.t)
            return result
        _ACTIVE=True;radau.solve_collocation_system=collocation;pwr_train.solve_ivp=ivp
        return self

    def __exit__(self,*args):
        global _ACTIVE
        radau.solve_collocation_system=self.original_collocation;pwr_train.solve_ivp=self.original_ivp;_ACTIVE=False


def measure(call,instrumented):
    profiler=cProfile.Profile() if instrumented else None;probe=SolverProbe();error=None;value=None
    start=time.perf_counter()
    try:
        if instrumented:
            with probe:
                profiler.enable();value=call()
        else:value=call()
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc),code=getattr(exc,'code',None))
    finally:
        if profiler:profiler.disable()
    elapsed=time.perf_counter()-start
    profile=summarize(pstats.Stats(profiler).stats) if profiler and profiler.getstats() else None
    return value,dict(wall_s=elapsed,solver_counts=probe.counts if instrumented else None,profile=profile),error


def physical(env):
    owner=deepcopy(env._session.control.plant);snapshot=owner.snapshot()
    return dict(time_s=owner.time,state=owner.state.tolist(),
        audits={key:snapshot[key] for key in ('energy_relative','mass_relative','boron_relative')},
        events=deepcopy(snapshot['events']))


def execute(case,setting,*,instrumented=False):
    if case not in CASES or setting not in SETTINGS:raise ValueError('Declared case and numerical setting required')
    water._raw_values.cache_clear();water.saturation.cache_clear();iapws.saturation_properties.cache_clear()
    config=EnvConfig(max_steps=len(CASES[case]),tick_s=.05,automatic=False,seed=0,**SETTINGS[setting])
    env=StationEnv(config);phases={};records=[];failure=None;checkpoint=None;restore_exact=False;initial=None
    _,phases['initialize'],error=measure(env.reset,instrumented)
    if error:failure=dict(phase='initialize',**error)
    else:
        initial=physical(env)
        def advance():
            for action in CASES[case]:
                transition=env.step(deepcopy(action));records.append(dict(transition=transition,physical=physical(env)))
        _,phases['advance'],error=measure(advance,instrumented)
        if error:failure=dict(phase='advance',**error)
        checkpoint,phases['checkpoint'],checkpoint_error=measure(env.checkpoint,instrumented)
        if checkpoint_error:failure=dict(phase='checkpoint',**checkpoint_error)
        else:
            serialized,phases['serialize'],serialize_error=measure(lambda:json.dumps(checkpoint,sort_keys=True,allow_nan=False),instrumented)
            if serialize_error:failure=dict(phase='serialize',**serialize_error)
            else:
                restored,phases['restore'],restore_error=measure(lambda:StationEnv.restore(json.loads(serialized)),instrumented)
                if restore_error:failure=dict(phase='restore',**restore_error)
                else:restore_exact=restored.checkpoint()==checkpoint
    science=dict(case=case,setting=setting,config=asdict(config),initial=initial,records=records,
        final_checkpoint=checkpoint,failure=failure,restore_exact=restore_exact,
        completed=failure is None and len(records)==len(CASES[case]))
    return dict(scientific=science,phases=phases)


def compare(candidate,reference):
    if not candidate['completed'] or not reference['completed']:return dict(comparable=False,passed=False,state_relative=None,event_time_error_s=None)
    a=deepcopy(candidate['config']);b=deepcopy(reference['config'])
    for config in (a,b):
        for key in ('rtol','max_step_s'):config.pop(key)
    if a!=b or candidate['case']!=reference['case']:raise ValueError('Same physical configuration and commands required')
    if len(candidate['records'])!=len(reference['records']):raise ValueError('Same complete physical sampling grid required')
    maximum=0.;budget=True;control=True
    for a,b in zip(candidate['records'],reference['records']):
        x=a['transition'];y=b['transition']
        control=control and x[2:4]==y[2:4]
        for key in ('trip_latched','trip_reasons','pressure_interlock','modes','targets'):
            control=control and x[4]['receipt'][key]==y[4]['receipt'][key]
    for a,b in zip([candidate['initial'],*[r['physical'] for r in candidate['records']]],
                   [reference['initial'],*[r['physical'] for r in reference['records']]]):
        if a['time_s']!=b['time_s'] or len(a['state'])!=len(b['state']):raise ValueError('Physical trace clock/schema mismatch')
        x=np.asarray(a['state']);y=np.asarray(b['state'])
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite complete state required')
        maximum=max(maximum,float(np.max(abs(x-y)/np.maximum(abs(y),1.))))
        budget=budget and all(0<=a['audits'][key]<=CRITERIA[key] and 0<=b['audits'][key]<=CRITERIA[key]
            for key in ('energy_relative','mass_relative','boron_relative'))
    a=candidate['records'][-1]['physical']['events'];b=reference['records'][-1]['physical']['events']
    kinds=len(a)==len(b) and all(x['kind']==y['kind'] and x.get('index')==y.get('index') and x.get('target')==y.get('target') for x,y in zip(a,b))
    event_error=max((abs(x['time_s']-y['time_s']) for x,y in zip(a,b)),default=0.) if kinds else None
    passed=maximum<=CRITERIA['state_relative'] and budget and control and kinds and event_error<=CRITERIA['event_time_s']
    return dict(comparable=True,passed=bool(passed),state_relative=maximum,
        budgets_passed=bool(budget),discrete_control_match=bool(control),event_kinds_match=kinds,event_time_error_s=event_error)


def attempt(case,mode,*,instrumented=False):
    started=time.perf_counter()
    try:return execute(case,mode,instrumented=instrumented)
    except Exception as error:
        return dict(scientific=dict(case=case,setting=mode,config=None,initial=None,records=[],
            final_checkpoint=None,restore_exact=False,completed=False,
            failure=dict(phase='execution_setup',type=type(error).__name__,message=str(error),code=getattr(error,'code',None))),
            phases=dict(execution_setup=dict(wall_s=time.perf_counter()-started,profile=None,solver_counts=None)))


def run(output,*,repeats=2):
    if type(repeats) is not int or not 2<=repeats<=5:raise ValueError('Two to five uninstrumented repeats required')
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();out.mkdir(parents=True);trials={};profiles={};science={};comparisons={};counts={};attempts=[];checks={}
    atomic_json(out/'protocol.json',seal(dict(source=source,cases=CASES,settings=SETTINGS,criteria=CRITERIA,
        tick_s=.05,repeats=repeats,cache_reset='raw water and both saturation caches before every execution',
        order='case order fixed; preset order alternates by repeat; no warmup discarded',
        wall_limit_scope='existing model work guards; no new physical tolerance relaxation')))
    for case in CASES:
        trials[case]={key:[] for key in SETTINGS};science[case]={};profiles[case]={};counts[case]={}
        for repeat in range(repeats):
            order=list(SETTINGS) if repeat%2==0 else list(reversed(SETTINGS))
            for mode in order:
                execution=attempt(case,mode);payload=execution['scientific'];trials[case][mode].append(execution['phases'])
                attempts.append(dict(case=case,mode=mode,repeat=repeat,instrumented=False,completed=payload['completed'],failure=payload['failure']))
                if mode in science[case]:checks[f'{case}/{mode}/repeat-{repeat}']=payload==science[case][mode]
                else:science[case][mode]=payload
        for mode in SETTINGS:
            execution=attempt(case,mode,instrumented=True);payload=execution['scientific'];profiles[case][mode]=execution['phases']
            attempts.append(dict(case=case,mode=mode,repeat=None,instrumented=True,completed=payload['completed'],failure=payload['failure']))
            checks[f'{case}/{mode}/instrumentation-exact']=payload==science[case][mode]
            counts[case][mode]={phase:row['solver_counts'] for phase,row in execution['phases'].items()}
            checks[f'{case}/{mode}/restart']=payload['restore_exact']
            checks[f'{case}/{mode}/iteration-count-complete']=all(
                row['solver_counts'] is not None and row['solver_counts']['iteration_count_complete'] for row in execution['phases'].values())
        if case=='invalid-command':
            checks['invalid-command/retained']=all(not p['completed'] and p['failure'] is not None and
                p['failure']['phase']=='advance' and p['failure']['code']=='invalid_action' for p in science[case].values())
        else:
            comparisons[case]={mode:compare(science[case][mode],science[case]['reference']) for mode in ('fast_candidate','detailed')}
            checks[case+'/accuracy']=all(r['passed'] for r in comparisons[case].values())
        atomic_json(out/(case+'-science.json'),seal(dict(runs=science[case],comparisons=comparisons.get(case),
            instrumentation_exact={mode:checks[f'{case}/{mode}/instrumentation-exact'] for mode in SETTINGS},solver_counts=counts[case])))
        atomic_json(out/(case+'-cost.json'),seal(dict(trials=trials[case],instrumented=profiles[case])))
    checks['source-unchanged']=source==identity()
    timing={}
    for case in CASES:
        timing[case]={}
        for mode in SETTINGS:
            values=[r['advance']['wall_s'] for r in trials[case][mode] if 'advance' in r]
            count=sum(r['completed'] for r in attempts if r['case']==case and r['mode']==mode and not r['instrumented'])
            elapsed=len(science[case][mode]['records'])*.05
            median=statistics.median(values) if values else None
            timing[case][mode]=dict(advance_median_wall_s=median,advance_trials_measured=len(values),
                completed_timing_trials=count,
                total_median_wall_s=statistics.median(sum(p['wall_s'] for p in r.values()) for r in trials[case][mode]),
                advance_wall_per_simulated_s=median/elapsed if median is not None and elapsed>0 and count==repeats else None)
        x=timing[case]['detailed'];y=timing[case]['fast_candidate']
        timing[case]['detailed_to_fast_ratio']=x['advance_median_wall_s']/max(y['advance_median_wall_s'],1e-12) if x['completed_timing_trials']==y['completed_timing_trials']==repeats else None
    summary=dict(attempted=len(attempts),completed=sum(r['completed'] for r in attempts),failed=sum(not r['completed'] for r in attempts))
    report=dict(source=source,checks=checks,passed=all(checks.values()),comparisons=comparisons,
        timing=timing,summary=summary,attempts=attempts,machine=machine(),
        exact_instrumentation_required=True,timing_uses_uninstrumented_only=True,
        nonlinear_iterations_are_actual=True,private_scipy_hook_version_bound=True,
        nonlinear_iteration_scope='Radau collocation including rejected attempts; not EOS inverse-iteration counts',
        advance_timing_scope='Native agent steps plus identical privileged physical trace capture',
        failed_attempt_costs_retained=True,deliberate_input_refusal_controls=True,
        full_r1_accepted=False,industrial_qualification=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Station precision/cost matrix refused; all attempts retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    result=run(**vars(parser.parse_args()));print(dict(passed=result['passed'],summary=result['summary']))
