"""Qualify bounded counterfactual decisions and exact commit on six graph paths."""
import argparse
from copy import deepcopy
from pathlib import Path
from atomic_model_decision import evaluate,commit,signature
from atomic_model_hem_heated import HeatedHEMNetwork
from atomic_model_hem_heated.model import canonical
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import verify_heated_hllc as heated

CASES=('wall_control','thermal_domain')
SCHEMES=('rusanov','hllc','hllc-local')


def setup(case,scheme='hllc-local'):
    if case not in CASES or scheme not in SCHEMES:raise ValueError('Declared decision matrix required')
    request=heated.request_for('wall_relaxation',heated.CAPS[-1])
    if case=='thermal_domain':
        request['model']['walls'][0].update(initial_temperature_k=899.999,conductance_w_k=0.,ambient_ua_w_k=0.)
    model=HeatedHEMNetwork(request,scheme=scheme);model.advance(1e-5)
    candidates=[]
    for name,power in [('off',0.),('high',1e6),('cool',-1e5)]:
        controls=deepcopy(model.snapshot()['controls']);controls['walls']['wall']=power
        candidates.append(dict(id=name,steps=[dict(duration_s=5e-5,controls=deepcopy(controls)) for _ in range(2)]))
    target=dict(wall='wall',temperature_k=459.8 if case=='wall_control' else 899.999)
    limits=dict(wall_temperature_max_k=460.05 if case=='wall_control' else 900.,fluid_temperature_max_k=500.,pressure_max_pa=8e6)
    return model,candidates,target,limits


def execute(case,scheme):
    model,candidates,target,limits=setup(case,scheme);before=model.checkpoint()
    decision=evaluate(model,candidates,target=target,limits=limits);payload=unseal(decision)
    unchanged=canonical(before)==canonical(model.checkpoint())
    repeat=evaluate(model,list(reversed(candidates)),target=target,limits=limits)
    order_invariant=unseal(repeat)['selected_id']==payload['selected_id']
    branches={r['id']:r for r in payload['results']}
    expected_high='constraint_violated' if case=='wall_control' else 'model_rejected'
    clean=HeatedHEMNetwork(model.request,scheme=scheme);clean.restore(before)
    chosen=next(c for c in candidates if c['id']==payload['selected_id'])
    for index,step in enumerate(chosen['steps']):
        clean.step(step['controls'],step['duration_s'],expected_revision=clean.snapshot()['revision'],
            controller_state={'decision_candidate':chosen['id'],'decision_step':index})
    committed=commit(model,decision)
    exact=canonical(clean.checkpoint())==canonical(model.checkpoint())
    after=model.checkpoint()
    try:commit(model,decision)
    except ValueError:stale=canonical(after)==canonical(model.checkpoint())
    else:stale=False
    checks=dict(read_only=unchanged,order_invariant=order_invariant,off_selected=payload['selected_id']=='off',
        high_classified=branches['high']['status']==expected_high,commit_exact=exact,stale_refused=stale)
    return dict(case=case,scheme=scheme,checks=checks,passed=all(checks.values()),decision=decision,committed=committed)


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False);source=signature();driver=sha(__file__)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver=driver,cases=CASES,schemes=SCHEMES,
        scope='Model-based sampled counterfactual control; no certified continuous constraint or plant validation')))
    records=[]
    for case in CASES:
        for scheme in SCHEMES:
            result=execute(case,scheme);atomic_json(output/f'{case}-{scheme}.json',seal(result))
            records.append({k:v for k,v in result.items() if k not in ('decision','committed')})
            print(case,scheme,result['passed'],flush=True)
    unchanged=source==signature() and sha(__file__)==driver
    report=dict(passed=unchanged and all(r['passed'] for r in records),source_unchanged=unchanged,records=records,
        paths=6,candidates_per_evaluation=3,evaluations_including_replay=18,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    return 0 if run(p.parse_args().output)['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
