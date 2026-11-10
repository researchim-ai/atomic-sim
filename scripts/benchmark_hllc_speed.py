"""Paired exact-result CPU measurements against an explicit prior source snapshot.

No accuracy/CFL changes. Timings are observations, never a pass/fail speed gate.
Baseline code must match the supplied sealed pre-change protocol before import.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import benchmark_modelica_hllc as cases

MATRIX=(('thermal_step',4,'hllc-local'),('pressure_thermal_step',4,'hllc-local'),
        ('wide_pressure_step',2,'hllc-local'),('pressure_thermal_step',4,'hllc'))
TIMES=(0.,.00025,.0005,.001)
TRIALS=5


def baseline_class(path,protocol):
    path=Path(path).absolute();protocol=Path(protocol).absolute()
    previous=unseal(json.loads(protocol.read_text()))['source'];current=signature()
    files={p.name:sha(p) for p in path.glob('*.py')}
    if files!=previous['contact'] or previous['foundation']!=current['foundation'] or previous==current:
        raise ValueError('Baseline snapshot/foundation mismatch or no source change')
    name='_atomic_hllc_speed_baseline_'+sha(path/'flux.py')[:16]
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if sys.modules[name+'.model'].signature()!=previous:raise ValueError('Baseline signature changed during import')
    return module.ContactHEMNetwork,previous,files


def request_for(case,cells):
    request=cases.request_for(case,cells,cases.CAPS[0]);request['model']['max_step_s']=1e-5
    return request


def capture(model_class,request,scheme):
    started_wall=time.perf_counter();started_cpu=time.process_time()
    model=model_class(request,scheme=scheme);observations=[model.snapshot()]
    for a,b in zip(TIMES,TIMES[1:]):observations.append(model.advance(b-a))
    checkpoint=model.checkpoint()
    cpu=time.process_time()-started_cpu;wall=time.perf_counter()-started_wall
    return dict(observations=observations,checkpoint=checkpoint,cpu_s=cpu,wall_s=wall)


def equivalent(before,after):
    a=dict(before['checkpoint']['payload']);b=dict(after['checkpoint']['payload'])
    a.pop('source');b.pop('source')
    def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    return dict(observations_exact=canonical(before['observations'])==canonical(after['observations']),numerical_checkpoint_exact=canonical(a)==canonical(b))


def run(output,*,baseline,baseline_protocol):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    baseline=Path(baseline).absolute();baseline_protocol=Path(baseline_protocol).absolute()
    old,previous,oldfiles=baseline_class(baseline,baseline_protocol)
    current=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,cases.__file__)}
    protocol_sha=sha(baseline_protocol)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=current,baseline_source=previous,baseline_files=oldfiles,
        baseline_protocol_sha256=protocol_sha,drivers=drivers,matrix=MATRIX,trials=TRIALS,times_s=TIMES,
        scope='Same-machine CPU/wall; constructor, advances, checkpoint; five paired alternating trials, one warmup',
        dedicated_hardware=False,speed_acceptance_gate=False,validated=False,plant_referenced=False)))
    rows=[]
    for case,cells,scheme in MATRIX:
        request=request_for(case,cells)
        warm_before=capture(old,request,scheme);warm_after=capture(ContactHEMNetwork,request,scheme)
        checks=[equivalent(warm_before,warm_after)];trials=[]
        for trial in range(TRIALS):
            order=('before','after') if trial%2==0 else ('after','before');values={}
            for key in order:values[key]=capture(old if key=='before' else ContactHEMNetwork,request,scheme)
            check=equivalent(values['before'],values['after']);checks.append(check)
            trials.append(dict(trial=trial,order=order,checks=check,**{k:{f:values[k][f] for f in ('cpu_s','wall_s')} for k in order}))
        medians={key:{field:statistics.median(t[key][field] for t in trials) for field in ('cpu_s','wall_s')} for key in ('before','after')}
        row=dict(case=case,cells=cells,scheme=scheme,trials=trials,median=medians,
            cpu_speedup=medians['before']['cpu_s']/medians['after']['cpu_s'],wall_speedup=medians['before']['wall_s']/medians['after']['wall_s'],
            checks=dict(all_observations_exact=all(c['observations_exact'] for c in checks),all_numerical_checkpoints_exact=all(c['numerical_checkpoint_exact'] for c in checks)))
        row['passed']=all(row['checks'].values());rows.append(row)
        atomic_json(output/f'{case}-{cells}-{scheme}.json',seal(dict(record=row,last_before=values['before'],last_after=values['after'])))
        print(case,cells,scheme,row['passed'],row['cpu_speedup'],flush=True)
    unchanged=current==signature() and all(sha(p)==h for p,h in drivers.items()) and oldfiles=={p.name:sha(p) for p in baseline.glob('*.py')} and protocol_sha==sha(baseline_protocol)
    report=dict(completed=True,passed=unchanged and all(r['passed'] for r in rows),source_unchanged=unchanged,rows=rows,
        speed_acceptance_gate=False,checkpoint_sources_differ=True,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','baseline','baseline_protocol'):p.add_argument('--'+key.replace('_','-'),required=True,type=Path)
    args=p.parse_args();return 0 if run(args.output,baseline=args.baseline,baseline_protocol=args.baseline_protocol)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
