"""Local command-boundary heat checks, retaining unsupported DHR evidence."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig
from atomic_model_station_event_heat import run_event_window
from atomic_model_station_event_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    cases={
        'load_down_up':{0:{'operator':{'load_fraction':.95}},2:{'operator':{'load_fraction':1.}}},
        'load_up_down':{0:{'operator':{'load_fraction':1.05}},2:{'operator':{'load_fraction':1.}}},
        'dhr_unresolved':{0:{'operator':{'dhr_requested':True}}},
        'invalid_later':{1:{'operator':{'load_fraction':99}}},
    }
    results={}
    for name,commands in cases.items():
        row=run_event_window(EnvConfig(max_steps=4,tick_s=.01,automatic=False),commands)
        results[name]=row;atomic_json(out/(name+'.json'),seal(row))
    dhr=results['dhr_unresolved'];invalid=results['invalid_later']
    checks=dict(
        load_paths=all(results[k]['passed'] for k in ('load_down_up','load_up_down')),
        jump_negative_controls=all(results[k]['naive_refused_segments']>=2 for k in ('load_down_up','load_up_down')),
        dhr_refused=not dhr['passed'] and not dhr['records'] and dhr['failure'] is not None
            and dhr['failure']['receipt']['reason']=='segment_heat_budget',
        invalid_prefix=not invalid['passed'] and len(invalid['records'])==1
            and invalid['failure']['code']=='invalid_action')
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,passed=all(checks.values()),
        horizon_s=.04,internal_event_subdivision_supported=False,dhr_heat_qualified=False,
        stationary_efficiency_certified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Event qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
