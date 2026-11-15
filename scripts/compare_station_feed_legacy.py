"""Compare opt-in feed extension with an explicitly supplied previous wheel."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv

import numpy as np
from atomic_model_station import PWRBypassSession
from atomic_model_station.bypass import bypass_signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json

WORKER='''
import json,sys
from pathlib import Path
import atomic_model_station as station
from atomic_model_station import PWRBypassSession
from atomic_model_station.bypass import bypass_signature
assert 'site-packages' in Path(station.__file__).parts
s=PWRBypassSession(max_step_s=1.,rtol=1e-7)
rows=[dict(time_s=s.time,state=s.state.tolist())]
for dt in (1.,4.,15.):
 s.advance(dt);rows.append(dict(time_s=s.time,state=s.state.tolist()))
Path(sys.argv[1]).write_text(json.dumps(dict(source=bypass_signature(),trace=rows)))
'''


def run(output,old_wheel,baseline):
    output=Path(output).absolute();old_wheel=Path(old_wheel).resolve();baseline=Path(baseline).resolve()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output refused')
    accepted=unseal(json.loads(baseline.read_text()))
    if not accepted['study_report']['passed'] or old_wheel.suffix!='.whl':raise ValueError('Accepted baseline wheel required')
    if sha(old_wheel)!=accepted['installed_replay']['wheel_sha256']:raise ValueError('Wheel differs from accepted baseline')
    output.mkdir(parents=True,exist_ok=False)
    driver=sha(__file__);source=bypass_signature();wheel_hash=sha(old_wheel);baseline_hash=sha(baseline)
    with tempfile.TemporaryDirectory(prefix='atomic-feed-legacy-') as temp:
        temp=Path(temp).resolve();checkout=Path(__file__).resolve().parents[1]
        if temp==checkout or checkout in temp.parents:raise ValueError('Old wheel must execute outside checkout')
        venv.EnvBuilder(with_pip=True,system_site_packages=True).create(temp/'venv')
        python=temp/'venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for name,command in [('install',[str(python),'-I','-m','pip','install','--no-index','--no-deps',str(old_wheel)]),
            ('old-trace',[str(python),'-I','-c',WORKER,str(output/'old.json')])]:
            with (output/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=temp,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    old=json.loads((output/'old.json').read_text())
    baseline_source=accepted['protocol']['source']
    for _ in range(3):
        if 'station' in baseline_source:break
        if 'control' in baseline_source:baseline_source=baseline_source['control']
        elif 'faults' in baseline_source:baseline_source=baseline_source['faults']
        else:raise ValueError('Unsupported baseline source hierarchy')
    if old['source']!=baseline_source.get('station'):raise ValueError('Baseline station identity mismatch')
    s=PWRBypassSession(max_step_s=1.,rtol=1e-7);rows=[dict(time_s=s.time,state=s.state.tolist())]
    for dt in (1.,4.,15.):
        s.advance(dt);rows.append(dict(time_s=s.time,state=s.state.tolist()))
    if len(rows)!=len(old['trace']):raise ValueError('Legacy trace shape')
    errors=[];exact=True
    for a,b in zip(rows,old['trace']):
        if a['time_s']!=b['time_s']:raise ValueError('Legacy clock')
        x=np.asarray(a['state']);y=np.asarray(b['state'])
        if x.shape!=y.shape:raise ValueError('Legacy state dimension changed')
        errors.append(float(max(abs(x-y)/np.maximum(abs(y),1.))));exact=exact and np.array_equal(x,y)
    unchanged=(source==bypass_signature() and driver==sha(__file__) and wheel_hash==sha(old_wheel) and baseline_hash==sha(baseline))
    passed=bool(unchanged and max(errors)<=1e-10)
    report=dict(passed=passed,source_unchanged=unchanged,new_source=source,old_source=old['source'],
        old_wheel_sha256=wheel_hash,baseline_sha256=baseline_hash,driver_sha256=driver,
        old_trace_sha256=sha(output/'old.json'),duration_s=20.,relative_errors=errors,all_states_exact=bool(exact),
        criteria=dict(relative_error=1e-10),external_flow_control=False,external_inventory_control=False,
        current_trace=rows,closed_tasks=[],full_duration_regression=False,validated=False)
    atomic_json(output/'report.json',seal(report))
    if not passed:raise ValueError('Legacy feed extension changed baseline physics')
    print(json.dumps(dict(passed=True,relative_error=max(errors),exact=bool(exact))),flush=True)
    return report


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-wheel',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.old_wheel,args.baseline)
