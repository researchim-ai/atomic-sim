"""Admit existing native event references and compare fresh external campaigns."""
import argparse
import json
import math
from pathlib import Path

from atomic_model_hem_qualified.model import signature
from atomic_model_multiphysics.provenance import seal,sha,unseal
from atomic_model_training.spec import atomic_json
from scripts import verify_phase_event as event
from scripts import phase_wall_case,verify_phase_cycles


def read_reference(directory):
    directory=Path(directory).absolute()
    protocol=unseal(json.loads((directory/'protocol.json').read_text()))
    report=unseal(json.loads((directory/'report.json').read_text()))
    modules=(event,phase_wall_case,verify_phase_cycles)
    current={Path(m.__file__).name:sha(m.__file__) for m in modules}
    saved={Path(p).name:h for p,h in protocol['drivers'].items()}
    expected={(case,cells,cap) for case in event.CASES for cells in event.MESHES for cap in event.CAPS}
    actual={(r['case'],r['cells'],r['cap_s']) for r in report['native']}
    if (protocol['source']!=signature() or current!=saved or not report['completed'] or
        not report['source_unchanged'] or not report['native_passed'] or actual!=expected or
        len(report['native'])!=len(expected) or len(report['references'])!=4 or
        {r['case'] for r in report['references']}!=set(event.CASES)):
        raise ValueError('Native event reference incomplete, changed or refused')
    refs={row['case']:row['quadrature_time_s'] for row in report['references']}
    for row in report['references']:
        values=[row['quadrature_time_s'],row['quadrature_error_s'],*[r['time_s'] for r in row['ode']]]
        if any(not math.isfinite(v) for v in values) or not 0<row['quadrature_time_s']<.01 or not 0<=row['quadrature_error_s']<=event.REFERENCE_BUDGET_S or not row['passed'] or len(row['ode'])!=2 or {r['method'] for r in row['ode']}!={'DOP853','Radau'} or any(abs(r['time_s']-row['quadrature_time_s'])>event.REFERENCE_BUDGET_S for r in row['ode']):
            raise ValueError('Independent reference refused')
    for row in report['native']:
        lo,hi=row['lower_s'],row['upper_s']
        values=[lo,hi,row['maximum_event_error_s'],*row['signed_energy_bracket_j_kg']]
        error=max(abs(t-refs[row['case']]) for t in (lo,hi))
        if any(not math.isfinite(v) for v in values) or not 0<lo<=hi<.01 or hi-lo>event.ROOT_WIDTH_S or not row['passed'] or row['restart_exact'] is not True or error>event.NATIVE_BUDGET_S or abs(error-row['maximum_event_error_s'])>1e-15:
            raise ValueError('Native root outside current acceptance')
    return report


def run(output,*,reference_from,campaigns):
    if not campaigns:raise ValueError('At least one external campaign required')
    campaigns=[Path(p).absolute() for p in campaigns]
    if len(set(campaigns))!=len(campaigns) or len({p.name for p in campaigns})!=len(campaigns):raise ValueError('Distinct external campaigns required')
    output,reference_from=Path(output).absolute(),Path(reference_from).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    reference=read_reference(reference_from)
    refs={r['case']:r['quadrature_time_s'] for r in reference['references']}
    source=signature();driver=sha(__file__)
    inputs={str(reference_from/name):sha(reference_from/name) for name in ('protocol.json','report.json')}
    plans=[]
    for campaign in campaigns:
        protocol=unseal(json.loads((campaign/'protocol.json').read_text()))
        report=unseal(json.loads((campaign/'report.json').read_text()))
        if protocol['source']!=source or not protocol['include_wall'] or not report['completed'] or not report['source_unchanged']:
            raise ValueError('External campaign not complete with matching physical source')
        inputs.update({str(campaign/name):sha(campaign/name) for name in ('protocol.json','report.json')})
        plans.append((campaign,protocol,report))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'protocol.json',seal(dict(source=source,driver_sha256=driver,inputs=inputs,
        native_reused_from=str(reference_from),external_budget_s=event.EXTERNAL_BUDGET_S,
        scope='Readmission only of exact source-bound roots; external CSV event sides, original budget')))
    records=[]
    for campaign,protocol,report in plans:
        for case in refs:
            for method in ('dassl','ida'):
                folder=campaign/f'p-1e+06-{case}-wall';path=folder/f'{method}.csv'
                try:
                    comparison_path=folder/f'{method}-comparison.json'
                    comparison=unseal(json.loads(comparison_path.read_text()))['record']
                    if sha(path)!=comparison['csv_sha256'] or comparison['case']!=case or comparison['method']!=method:
                        raise ValueError('External CSV identity mismatch')
                    inputs[str(path)]=sha(path);inputs[str(comparison_path)]=sha(comparison_path)
                    root=event.external_events(path);error=abs(root['time_s']-refs[case])
                    record=dict(campaign=campaign.name,case=case,method=method,**root,
                        event_error_s=error,common_grid_passed=comparison['passed'],
                        passed=comparison['passed'] and error<=event.EXTERNAL_BUDGET_S,
                        peer_library_variant=protocol.get('peer_library_variant','as_supplied'))
                except (OSError,ValueError,KeyError) as error:
                    record=dict(campaign=campaign.name,case=case,method=method,passed=False,error=str(error))
                records.append(record)
    unchanged=signature()==source and driver==sha(__file__) and all(sha(p)==h for p,h in inputs.items())
    summary=[dict(campaign=p.name,accepted=sum(r['passed'] for r in records if r['campaign']==p.name),total=8,
                  passed=all(r['passed'] for r in records if r['campaign']==p.name)) for p in campaigns]
    report=dict(completed=True,source_unchanged=unchanged,native_reference_readmitted=True,
        native_recomputed=False,passed=unchanged and len(records)==8*len(campaigns) and all(r['passed'] for r in records),
        campaigns=summary,records=records,input_sha256=inputs,validated=False,plant_referenced=False,baseline_promoted=False)
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reference-from',required=True,type=Path)
    parser.add_argument('--campaigns',required=True,nargs='+',type=Path)
    args=parser.parse_args()
    return 0 if run(args.output,reference_from=args.reference_from,campaigns=args.campaigns)['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
