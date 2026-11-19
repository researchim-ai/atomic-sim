"""Executable public API tutorials; completion is not physical qualification."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_env.model import signature as station_identity
from atomic_model_station_faults import Fault
from atomic_model_station_control.policy import valid_reading
from atomic_model_station_tasks.model import tracking_policy
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json

NAMES=('first','load','fault','agent','replay')


def identity():
    return dict(station=station_identity(),files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def run_example(name,output):
    if type(name) is not str or name not in NAMES:raise ValueError('Unknown station example')
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();out.mkdir(parents=True);checks={};failure=None;transitions=[];actions=[];alarms=[]
    faults=(Fault('feed_drive_loss','pump_off','feed',1),) if name=='fault' else ()
    config=EnvConfig(max_steps=1 if name=='first' else 3,tick_s=.02,
        automatic=False,rtol=1e-10,faults=faults)
    try:
        env=StationEnv(config);observation,_=env.reset()
        initial=deepcopy(observation);schema=env.available_actions()
        target=valid_reading(observation,'power','fission_power_w')
        if name=='agent' and target is None:raise ValueError('Agent example requires a valid initial power measurement')
        midpoint=None
        for tick in range(config.max_steps):
            if name=='load':action={'operator':{'load_fraction':(1.,.99,1.)[tick]}}
            elif name=='fault':action={'operator':{'feed_pump_on':True}}
            elif name=='agent':
                choice=tracking_policy(deepcopy(observation),deepcopy(schema),
                    dict(target_w=target,diagnostic_threshold_pa=5e4))
                action=choice['action'];alarms.append(choice['alarm'])
            else:action={}
            actions.append(deepcopy(action));transition=env.step(action)
            transitions.append(transition);observation=transition[0]
            if tick==0 and name=='replay':
                midpoint=env.checkpoint();atomic_json(out/'midpoint.json',midpoint)
        final=env.checkpoint();atomic_json(out/'checkpoint.json',final)
        checks['declared_steps']=len(transitions)==config.max_steps
        checks['horizon_reached']=transitions[-1][3] is True
        if name=='load':
            checks['load_commands_selected']=[r[4]['receipt']['decisions']['load_fraction']['selected'] for r in transitions]==[1.,.99,1.]
        if name=='fault':
            # Privileged evaluator evidence is read only after all policy steps.
            p=unseal(final);p=unseal(p['session']);p=unseal(p['control'])
            snapshot=unseal(p['state']['plant'])['snapshot']
            truth=snapshot['equipment_fault_truth']
            checks['fault_activated']=truth['active']==['feed_drive_loss']
            checks['requested_on_effective_off']=snapshot['controls']['feed_pump_on'] is True and truth['effective_controls']['feed_pump_on'] is False
            checks['fault_epoch']=truth['events']==[dict(name='feed_drive_loss',time_s=.02,transition='started')]
        if name=='replay':
            restored=StationEnv.restore(midpoint)
            tail=[restored.step(a) for a in actions[1:]]
            checks['replay_transitions_exact']=tail==transitions[1:]
            checks['replay_checkpoint_exact']=restored.checkpoint()==final
        atomic_json(out/'trajectory.json',seal(dict(initial_observation=initial,actions=actions,
            transitions=transitions,agent_alarms=alarms,
            observation_visibility='measured_only',checkpoint_visibility='privileged_research_data')))
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
    checks['source_unchanged']=source==identity()
    report=dict(source=source,example=name,config=asdict(config),checks=checks,failure=failure,
        passed=failure is None and len(checks)>=3 and all(checks.values()),
        simulated_s=len(transitions)*config.tick_s,example_completed=len(transitions)==config.max_steps,
        physical_scenario_qualified=False,industrial_qualification=False,task_success_evaluated=False,
        closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Station example failed; evidence retained')
    return report
