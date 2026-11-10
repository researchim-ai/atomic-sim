"""Targeted tighter BDF replay; reuse source-bound native evidence explicitly."""
import argparse
import json
import math
from pathlib import Path
import shutil
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import atomic_json
from scripts import verify_hllc_long as bench


def read(path):return unseal(json.loads(Path(path).read_text()))


def run(output,parent):
    output,parent=map(lambda p:Path(p).absolute(),(output,parent))
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    source=bench.signature();old=read(parent/'protocol.json');old_report=read(parent/'report.json')
    if (old['source']!=source or not old_report['source_unchanged'] or old['times_s']!=list(bench.TIMES)
        or old['matrix']!=[list(v) for v in bench.MATRIX] or old['caps_s']!=list(bench.CAPS)):
        raise ValueError('Parent source or long matrix mismatch')
    for path,expected in old['drivers'].items():
        snapshot=parent/'driver.py' if Path(path).name=='verify_hllc_long.py' else parent/'reference-driver.py' if Path(path).name=='long_hllc_reference.py' else Path(path)
        if sha(snapshot)!=expected:raise ValueError('Parent driver evidence changed')
    source_files=[parent/'protocol.json',parent/'report.json',parent/'driver.py',parent/'reference-driver.py']
    for row in old_report['records']:
        if not row['refinement']['passed'] or not all(r['passed'] for r in row['native_runs']):raise ValueError('Parent native evidence not accepted')
        if not row['reference_cross']['passed'] and row['case']!='wide_pressure_step':raise ValueError('Unexpected parent reference failure')
    for case,cells in bench.MATRIX:
        folder=parent/f'{case}-{cells}'
        for name in ['references.json','report.json',*[f'native-{cap}.json' for cap in bench.CAPS]]:
            read(folder/name);source_files.append(folder/name)
    manifest={str(p):sha(p) for p in source_files}
    drivers={str(Path(p).absolute()):sha(p) for p in (__file__,bench.__file__,bench.reference_solver.__file__,bench.history.__file__,bench.cases.__file__,bench.peer.__file__)}
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(old,drivers=drivers,parent_evidence_sha256=manifest,qualification_mode='Targeted wide BDF refinement; source-bound native and seven reference paths reused',
        reference_accuracy_profiles={case:{method:bench.reference_accuracy(case,method) for method in bench.REFERENCE_METHODS} for case,_ in bench.MATRIX},
        native_recomputed=0,native_reused=12,reference_recomputed=1,reference_reused=7)
    atomic_json(output/'protocol.json',seal(protocol));records=[]
    for case,cells in bench.MATRIX:
        folder=output/f'{case}-{cells}';shutil.copytree(parent/f'{case}-{cells}',folder)
        references=read(folder/'references.json')['references']
        for ref in references:ref.update(rtol=2e-11,atol=1e-10)
        request=bench.request_for(case,cells,bench.CAPS[-1])
        if case=='wide_pressure_step':
            replacement=bench.reference_solver.solve(request,bench.TIMES,'BDF',max_step=bench.REFERENCE_CAP,
                **bench.reference_accuracy(case,'BDF'),on_window=lambda w:atomic_json(folder/'BDF-progress.json',seal(w)))
            references=[replacement if r['method']=='BDF' else r for r in references]
        atomic_json(folder/'references.json',seal(dict(references=references)))
        cross=bench.history.compare(bench.reference_packet(request,references[0]),bench.peer.as_peer(references[1]['observations']),times=bench.TIMES,budgets=bench.peer.REFERENCE_BUDGETS)
        comparisons=[];runs=[]
        for cap in bench.CAPS:
            native=read(folder/f'native-{cap}.json')
            if native['request']!=bench.request_for(case,cells,cap) or native['checkpoint']['payload']['source']!=source:raise ValueError('Reused native identity mismatch')
            if [s['time_s'] for s in native['observations']]!=list(bench.TIMES):
                if any(abs(a-b)>1e-12 for a,b in zip([s['time_s'] for s in native['observations']],bench.TIMES,strict=True)):raise ValueError('Reused native time grid mismatch')
            for reference in references:
                comparisons.append(dict(cap_s=cap,method=reference['method'],**bench.history.compare(native,bench.peer.as_peer(reference['observations']),times=bench.TIMES)))
            runs.append(dict(cap_s=cap,**native['record']))
        refined=bench.refinement(comparisons)
        initial_states=[bench.eos.pt(n['initial']['pressure_pa'],n['initial']['temperature_k']) for n in request['model']['nodes']]
        transits=bench.TIMES[-1]*min(bench.eos.characteristic_speeds(s)[0] for s in initial_states)/max(e['length_m'] for e in request['model']['edges'])
        record=dict(case=case,cells=cells,native_runs=runs,comparisons=comparisons,reference_cross=cross,refinement=refined,
            reference_evaluations={r['method']:r['evaluations'] for r in references},acoustic_transits_lower=transits,
            passed=cross['passed'] and refined['passed'] and all(r['passed'] for r in runs))
        records.append(record);atomic_json(folder/'report.json',seal(record));print(case,cells,record['passed'],flush=True)
    unchanged=source==bench.signature() and all(sha(p)==h for p,h in drivers.items()) and all(sha(p)==h for p,h in manifest.items())
    report=dict(completed=True,passed=unchanged and all(r['passed'] for r in records),source_unchanged=unchanged,records=records,
        native_recomputed=0,native_reused=12,reference_recomputed=1,reference_reused=7,
        validated=False,plant_referenced=False,whole_plant_compared=False,experimental_validation=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--parent',required=True,type=Path)
    args=p.parse_args();return 0 if run(args.output,args.parent)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
