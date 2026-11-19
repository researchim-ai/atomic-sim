"""Native station delivery checks, with explicit reuse of source-bound hour evidence."""
import argparse
from copy import deepcopy
from pathlib import Path

from atomic_model_station_env import StationEnv, EnvConfig
from atomic_model_station_env.model import signature
from atomic_model_station_view.__main__ import demo
from atomic_model_station_view import model as viewer
from atomic_model_multiphysics.provenance import sha, seal, unseal
from atomic_model_training.spec import atomic_json, read_json
from scripts import qualify_station_precision as precision
from scripts import qualify_station_equilibrium as nominal
from scripts import requalify_station_nominal as nominal_delivery


def identity():
    return dict(station=signature(), precision_driver_sha256=sha(precision.__file__),
        viewer_sha256=sha(viewer.__file__), driver_sha256=sha(__file__))


def accept_nominal(value):
    """A previous full-hour receipt is usable only for the exact current recipe."""
    p=unseal(value)
    try:
        r=p['report'];s=r['source_report'];n=p['installed_native']
        if p['format']!='station-nominal-requalification/v105':raise ValueError('Unknown nominal receipt')
        if r['source']!=nominal.identity() or r['driver_sha256']!=sha(nominal_delivery.__file__):
            raise ValueError('Nominal recipe source changed; requalification required')
        required={'source_inputs_unchanged','wheel_unchanged','all_mathematical_json_exact',
            'native_replay','installed_fine','source_unchanged'}
        if (r['passed'] is not True or r['failure'] is not None or set(r['checks'])!=required or
            any(v is not True for v in r['checks'].values()) or r['mathematical_files_compared']!=18):
            raise ValueError('Complete source/wheel nominal acceptance required')
        if s['passed'] is not True or s['source_unchanged'] is not True or s['duration_s']!=3600.:
            raise ValueError('Full nominal source hour required')
        for role in ('coarse','fine'):
            x=s[role]
            if (x['duration_s']!=3600. or x['window_count']!=601 or any(x[k] is not True for k in
                ('completed','passed','source_unchanged','all_600s_windows_ready','physical_audits_passed'))):
                raise ValueError('Both full source settings required')
        if n['duration_s']!=3600. or any(n[k] is not True for k in
            ('completed','prefix_exact','full_suffix_and_final_exact','source_unchanged')):
            raise ValueError('Full installed native replay required')
    except (KeyError,TypeError) as error:
        raise ValueError('Incomplete nominal receipt') from error
    return dict(receipt_sha256=value['sha256'],recipe_source=r['source'],duration_s=3600.,
        coarse_and_fine_windows=601,mathematical_files_exact=18,
        reused_existing_execution=True,new_hour_executed=False,
        scope='Previously accepted nominal only; source-bound reuse, not a new wheel hour')


def midpoint_replay():
    env=StationEnv(EnvConfig(max_steps=3,tick_s=.05,automatic=False));env.reset()
    env.step({'operator':{'feed_pump_on':False}});checkpoint=env.checkpoint()
    restored=StationEnv.restore(deepcopy(checkpoint));actions=[{'operator':{'feed_pump_on':True}},{}]
    a=[env.step(deepcopy(action)) for action in actions]
    b=[restored.step(deepcopy(action)) for action in actions]
    return dict(midpoint=checkpoint,transitions=a,final_checkpoint=env.checkpoint(),
        transitions_exact=a==b,checkpoint_exact=env.checkpoint()==restored.checkpoint())


def run(output, *, nominal_receipt=None, require_installed=False):
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('Fresh ordinary output file required')
    if require_installed:
        from atomic_model_station_env import model
        if any('site-packages' not in Path(p).parts for p in (__file__,model.__file__,viewer.__file__)):
            raise ValueError('Station checks must import the installed wheel')
    accepted=accept_nominal(read_json(nominal_receipt)) if nominal_receipt is not None else None
    source=identity();cases={};checks={};failures=[]
    for case in precision.CASES:
        scientific=precision.attempt(case,'fast_candidate')['scientific'];cases[case]=scientific
        checks[case+'/restart']=scientific['restore_exact'] is True
        if case=='invalid-command':
            checks[case+'/expected-refusal']=(not scientific['completed'] and scientific['failure'] is not None and
                scientific['failure']['phase']=='advance' and scientific['failure']['code']=='invalid_action')
        else:checks[case+'/complete']=scientific['completed'] is True and scientific['failure'] is None
    replay=recording=None
    try:
        replay=midpoint_replay();checks['midpoint-transitions']=replay['transitions_exact']
        checks['midpoint-checkpoint']=replay['checkpoint_exact']
    except Exception as error:
        checks['midpoint-transitions']=checks['midpoint-checkpoint']=False
        failures.append(dict(phase='midpoint-replay',type=type(error).__name__,message=str(error)))
    try:
        recording=demo();viewer.load(recording);checks['view-document']=True
    except Exception as error:
        checks['view-document']=False
        failures.append(dict(phase='view-document',type=type(error).__name__,message=str(error)))
    checks['source-unchanged']=source==identity()
    body=dict(source=source,cases=cases,midpoint_replay=replay,view_document=recording,
        nominal=accepted,checks=checks,failures=failures,passed=all(checks.values()),full_r1_accepted=False,
        physical_validation=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,seal(dict(scientific=body,execution=dict(installed_imports_required=require_installed))))
    if not body['passed']:raise ValueError('Native station installation checks failed; results retained')
    return body


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nominal-receipt',type=Path)
    parser.add_argument('--require-installed',action='store_true')
    result=run(**vars(parser.parse_args()));print(dict(passed=result['passed'],checks=len(result['checks'])))
