"""Resume the original radial study by sealed completed cases, without changing its equations."""
import argparse
from pathlib import Path

from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from scripts.qualify_station_inventory import read
from scripts import qualify_station_thermal_radial_paths as study


def validate_case(row,case):
    if row.get('case')!=case.specification() or set(row.get('stages',{}))!=set(study.VARIANTS):
        raise ValueError('Complete matching case/variant evidence required')
    checks=row.get('checks',{})
    expected={'complete','same_initial_fields','weak_pressure','budgets','sampled_wall',*study.VARIANTS[1:]}
    if set(checks)!=expected or any(type(v) is not bool for v in checks.values()):
        raise ValueError('Complete boolean checks required')
    if type(row.get('accepted')) is not bool or row['accepted']!=all(checks.values()):
        raise ValueError('Inconsistent case acceptance')
    for stage in row['stages'].values():
        if type(stage.get('completed')) is not bool or type(stage.get('records')) is not list:
            raise ValueError('Complete stage status required')
        if stage['completed'] and [r['status']['time'] for r in stage['records']]!=list(study.TIMES):
            raise ValueError('Complete ordered epochs required')
    if checks['complete']!=all(s['completed'] for s in row['stages'].values()):
        raise ValueError('Inconsistent stage completeness')
    if row['accepted']:
        if set(row.get('comparisons',{}))!=set(study.VARIANTS[1:]):
            raise ValueError('Complete accepted comparisons required')
        for values in row['comparisons'].values():
            if set(values)!={str(t) for t in study.TIMES} or not all(study.below(v,1e-7) for v in values.values()):
                raise ValueError('Accepted radial comparisons violate protocol')
    return row


def resume(output,*,max_cases=1):
    if type(max_cases) is not int or not 1<=max_cases<=5:
        raise ValueError('Case work budget1..5 required')
    out=Path(output).absolute()
    if not out.is_dir() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Existing ordinary study directory required')
    source=study.identity();protocol=read(out/'protocol.json');cases=study.thermal_catalog()
    if protocol['source']!=source or protocol['cases']!=[c.specification() for c in cases] or protocol['times']!=list(study.TIMES) or protocol['variants']!=list(study.VARIANTS):
        raise ValueError('Protocol/source mismatch; cannot resume')
    allowed={'protocol.json','report.json','recovery.json',*(c.name+'.json' for c in cases)}
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in out.iterdir()):
        raise ValueError('Unexpected study artifact')
    # Validate every saved row before performing new numerical work.
    rows={c.name:validate_case(read(out/(c.name+'.json')),c) for c in cases if (out/(c.name+'.json')).exists()}
    preserved={name:sha(out/(name+'.json')) for name in rows}
    count=0
    for case in cases:
        if case.name in rows or count==max_cases:
            continue
        row=validate_case(study.qualify_case(case),case)
        if source!=study.identity():
            raise ValueError('Scientific source changed during recovery')
        atomic_json(out/(case.name+'.json'),seal(row));rows[case.name]=row;count+=1
    if source!=study.identity() or any(sha(out/(name+'.json'))!=value for name,value in preserved.items()):
        raise ValueError('Existing evidence changed during recovery')
    complete=len(rows)==len(cases)
    recovery=dict(source=source,recovery_driver=sha(__file__),completed=complete,
        newly_computed_cases=count,preserved_case_hashes=preserved,
        missing_cases=[c.name for c in cases if c.name not in rows],
        interrupted_unsaved_case_recomputed=True)
    atomic_json(out/'recovery.json',seal(recovery))
    if not complete:
        return recovery
    ordered=[rows[c.name] for c in cases]
    passed=all(r['accepted'] for r in ordered)
    report=dict(passed=passed,source=source,accepted_cases=[r['case']['name'] for r in ordered if r['accepted']],
        refused_cases=[r['case']['name'] for r in ordered if not r['accepted']],
        completed_paths=sum(s['completed'] for r in ordered for s in r['stages'].values()),
        completed_epochs=sum(len(s['records']) for r in ordered for s in r['stages'].values()),
        sampled_inner_and_shell_radial_comparisons_passed=passed,sample_times=list(study.TIMES),
        continuous_radial_error_bound=False,parameter_box_qualified=False,station_usable=False,
        validated=False,closed_tasks=[],case_evidence={c.name:sha(out/(c.name+'.json')) for c in cases})
    atomic_json(out/'report.json',seal(report))
    return recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cases',type=int,default=1)
    args=parser.parse_args();result=resume(args.output,max_cases=args.max_cases)
    print(result)
    if result['completed'] and not read(args.output/'report.json')['passed']:
        raise SystemExit(1)


if __name__=='__main__':main()
