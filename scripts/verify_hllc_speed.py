"""Current-cache qualification: analytic contacts, long equivalence and prior refs.

Previous long evidence is reused only after unchanged foundation/reference-code
checks. This does not silently transfer the entire historical long matrix.
"""
import argparse
import json
from pathlib import Path
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_hem_contact.model import signature
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import verify_local_hllc as local
from scripts import verify_contact_flux as contact
from scripts import verify_heterogeneous_network as previous
from scripts import verify_hllc_long as long


def read(p):return unseal(json.loads(Path(p).read_text()))


def run(output,*,previous_long):
    output,previous_long=map(lambda p:Path(p).absolute(),(output,previous_long))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=signature();old=read(previous_long/'protocol.json')['source']
    if source['foundation']!=old['foundation'] or source['contact']['reference.py']!=old['contact']['reference.py']:
        raise ValueError('Previous independent reference law or foundation changed')
    folder=previous_long/'wide_pressure_step-2';native_path=folder/f'native-{long.CAPS[-1]}.json'
    old_native=read(native_path);references=read(folder/'references.json')['references']
    request=long.request_for('wide_pressure_step',2,long.CAPS[-1])
    if old_native['request']!=request or old_native['checkpoint']['payload']['source']!=old:
        raise ValueError('Previous long packet identity mismatch')
    inherited={str(p):sha(p) for p in (previous_long/'protocol.json',native_path,folder/'references.json')}
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,local.__file__,contact.__file__,previous.__file__,long.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,drivers=drivers,previous_source=old,inherited_evidence_sha256=inherited,
        scope='Current cache:15 physical face cases,5 stationary contacts,one40ms native wide path; two inherited unchanged-RHS references',
        reference_trajectories_recomputed=0,reference_trajectories_reused=2,validated=False,plant_referenced=False)))
    faces=local.face_cases();contacts=[]
    for cells in (2,4,8,16,32):
        result=contact.capture(previous.contact_request(cells),'hllc-local')
        drift=max(abs(a['temperature_k']-b['temperature_k']) for a,b in zip(result['observations'][0]['cells'],result['observations'][-1]['cells'],strict=True))
        row=dict(cells=cells,temperature_drift_k=drift,checks=result['record']['checks'],passed=result['record']['passed'] and drift<=1e-9)
        contacts.append(row);atomic_json(output/f'contact-{cells}.json',seal(dict(record=row,native=result)))
    native=long.capture(request);atomic_json(output/'long-native.json',seal(native))
    a=dict(old_native['checkpoint']['payload']);b=dict(native['checkpoint']['payload']);a.pop('source');b.pop('source')
    checks=dict(observations_exact=native['observations']==old_native['observations'],numerical_checkpoint_exact=a==b,
                current_long_passed=native['record']['passed'])
    model=ContactHEMNetwork(request,scheme='hllc-local');before=model.checkpoint()
    try:model.restore(old_native['checkpoint'])
    except ValueError:checks['old_checkpoint_refused_without_mutation']=model.checkpoint()==before
    else:checks['old_checkpoint_refused_without_mutation']=False
    comparisons=[dict(method=r['method'],**long.history.compare(native,long.peer.as_peer(r['observations']),times=long.TIMES)) for r in references]
    unchanged=source==signature() and all(sha(p)==h for p,h in drivers.items()) and all(sha(p)==h for p,h in inherited.items())
    report=dict(completed=True,passed=unchanged and all(checks.values()) and all(r['passed'] for r in faces+contacts+comparisons),
        source_unchanged=unchanged,faces=faces,contacts=contacts,long_checks=checks,long_comparisons=comparisons,
        inherited_reference_code_unchanged=True,full_historical_long_matrix_replayed=False,validated=False,plant_referenced=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--previous-long',required=True,type=Path)
    args=p.parse_args();return 0 if run(args.output,previous_long=args.previous_long)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
