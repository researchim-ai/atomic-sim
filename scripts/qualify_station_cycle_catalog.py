"""Run explicit pressure-cycle variants, retaining every accepted or refused case."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_hybrid_heat.catalog import CycleCase,run_cycle_catalog
from atomic_model_station_hybrid_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    result=run_cycle_catalog([
        CycleCase('smaller-area',area_m2=.029),
        CycleCase('slower-actuator',stroke_time_s=.101),
    ])
    atomic_json(out/'catalog.json',seal(result))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,
        total_cases=result['total_cases'],passed_cases=result['passed_cases'],
        failed_cases=result['failed_cases'],passed=result['passed'],
        catalog_sha256=sha(out/'catalog.json'),physical_range_qualified=False,
        industrial_qualification=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Cycle catalog refused cases; all evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    print({k:result[k] for k in ('total_cases','passed_cases','failed_cases','passed')})
