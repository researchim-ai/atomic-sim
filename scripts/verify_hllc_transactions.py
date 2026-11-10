"""Transactional control/restart qualification around natural branch reversal.

Fault injection tests software atomicity; it is not an operating accident model.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_workspace.common import digest
from scripts import verify_hllc_branching as branch

SCHEMES=('rusanov','hllc','hllc-local')
MESHES=(2,4)
TIMES=(0.,.0001,.00025,.0005)
FAULTS=('fluid_heat','wall_heat','stale_revision','boolean_revision','nan_controller',
        'zero_duration','nan_duration','coordinates','observation','call_history','scheme','source','stage_failure')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def reject_unchanged(model,fault):
    """Return evidence only for an expected ValueError and byte-exact rollback."""
    if fault not in FAULTS:raise ValueError('Declared transaction fault required')
    before=model.checkpoint();observation=model.snapshot();controls=deepcopy(observation['controls'])
    revision=observation['revision'];controller={'tick':revision};duration=1e-5
    target=deepcopy(before)
    def operation():
        if fault in ('coordinates','observation','call_history','scheme','source'):
            payload=target['payload']
            if fault=='coordinates':payload['coordinates'][0]*=1.001
            elif fault=='observation':payload['observation']['time_s']+=1e-6
            elif fault=='call_history':payload['calls'][0]['duration_s']*=.9
            elif fault=='scheme':payload['scheme']='invalid'
            else:payload['source']={}
            target['sha256']=digest(payload)
            return model.restore(target)
        return model.step(controls,duration,expected_revision=revision,controller_state=controller)
    if fault=='fluid_heat':controls['fluid']['a']=1.
    elif fault=='wall_heat':controls['walls']['wall']=1.
    elif fault=='stale_revision':revision-=1
    elif fault=='boolean_revision':revision=True
    elif fault=='nan_controller':controller={'tick':float('nan')}
    elif fault=='zero_duration':duration=0.
    elif fault=='nan_duration':duration=float('nan')
    calls=0
    original=model.evaluate
    if fault=='stage_failure':
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            # SSP-RK stage one has already been evaluated, but no commit is allowed.
            raise ValueError('Injected post-first-stage failure')
        model.evaluate=interrupted
    try:
        try:operation()
        except ValueError as error:reason=str(error)
        else:raise AssertionError(f'Fault accepted: {fault}')
    finally:
        if fault=='stage_failure':del model.evaluate
    unchanged=canonical(model.checkpoint())==canonical(before) and canonical(model.snapshot())==canonical(observation)
    if not unchanged:raise AssertionError(f'Fault mutated state: {fault}')
    if fault=='stage_failure' and calls!=1:raise AssertionError('Injected failure not reached')
    return dict(fault=fault,rejected=True,unchanged=True,reason=reason,stage_injections=calls)


def trajectory(scheme,cells,*,times=TIMES):
    if scheme not in SCHEMES or cells not in MESHES:raise ValueError('Declared transaction matrix required')
    times=branch.histories.history.validate_times(times)
    if len(times)<3:raise ValueError('At least two advance windows required')
    request=branch.request_for('branch_reversal',cells,branch.CAPS[-1])
    model=ContactHEMNetwork(request,scheme=scheme)
    clean=ContactHEMNetwork(request,scheme=scheme)
    snapshots=[model.snapshot()];restarts=[];faults=[]
    for index,(a,b) in enumerate(zip(times,times[1:]),1):
        controls=deepcopy(model.snapshot()['controls']);state={'tick':index,'policy':'zero-power'}
        expected=clean.step(controls,b-a,expected_revision=index-1,controller_state=state)
        actual=model.step(controls,b-a,expected_revision=index-1,controller_state=state)
        if canonical(expected)!=canonical(actual):raise AssertionError('Continuation differs from clean control')
        snapshots.append(actual)
        for fault in FAULTS:faults.append(dict(time_s=b,**reject_unchanged(model,fault)))
        checkpoint=model.checkpoint()
        restored=ContactHEMNetwork(request,scheme=scheme);restored.restore(checkpoint)
        exact=canonical(restored.checkpoint())==canonical(checkpoint)
        if not exact:raise AssertionError('Restart differs')
        restarts.append(dict(time_s=b,exact=True,revision=index))
        model=restored
    checks=dict(clean_continuation=canonical(model.checkpoint())==canonical(clean.checkpoint()),
                restarts_exact=all(r['exact'] for r in restarts),
                faults_atomic=all(r['unchanged'] and r['rejected'] for r in faults))
    native=dict(request=request,observations=snapshots)
    audit=branch.audit(native);checks['support_balance']=audit['passed']
    checks['mass_balance']=all(abs(s['global_ledger']['mass_residual_kg'])<=1e-9 for s in snapshots)
    checks['energy_balance']=all(abs(s['global_ledger']['energy_residual_j'])<=1e-5 for s in snapshots)
    return dict(scheme=scheme,cells=cells,checks=checks,passed=all(checks.values()),
                audit=audit,restarts=restarts,faults=faults,checkpoint=model.checkpoint())


def run(output):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=signature();drivers={str(Path(p).absolute()):sha(p) for p in (__file__,branch.__file__)}
    protocol=dict(source=source,drivers=drivers,schemes=SCHEMES,meshes=MESHES,times_s=TIMES,faults=FAULTS,
                  scope='Single-phase unforced graph; command/replay atomicity, not physical accident validation')
    atomic_json(output/'protocol.json',seal(protocol));records=[]
    for scheme in SCHEMES:
        for cells in MESHES:
            record=trajectory(scheme,cells);records.append(record)
            atomic_json(output/f'{scheme}-{cells}.json',seal(record))
            print(scheme,cells,record['passed'],flush=True)
    unchanged=source==signature() and all(sha(p)==value for p,value in drivers.items())
    report=dict(passed=unchanged and all(r['passed'] for r in records),source_unchanged=unchanged,
                paths=len(records),restarts=sum(len(r['restarts']) for r in records),
                rejected_faults=sum(len(r['faults']) for r in records),records=records,
                validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    return 0 if run(parser.parse_args().output)['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
