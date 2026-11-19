"""Joint solver/sampling refinement of a one-second manual station heat window."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig
from atomic_model_station_heat_accounting import run_heat_window
from atomic_model_station_heat_accounting.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    protocol=dict(source=source,driver_sha256=driver,horizon_s=1.,ticks_s=[.04,.02,.01],
        refinement_relative_limit=2e-6,trend_noise_floor=1e-8,
        control='Unforced manual control;all commands empty;no scheduled faults',
        scope='Joint change of flux sampling and solver restart cadence;not isolated quadrature error',
        stationary_efficiency_certified=False)
    atomic_json(out/'protocol.json',seal(protocol))
    results={}
    for name,tick in [('coarse',.04),('medium',.02),('fine',.01)]:
        row=run_heat_window(EnvConfig(max_steps=round(1./tick),tick_s=tick,automatic=False))
        results[name]=row;atomic_json(out/(name+'.json'),seal(row))
        print(name,'complete',row['complete'],'passed',row['passed'],flush=True)
    comparisons={}
    complete=all(r['complete'] for r in results.values())
    if complete:
        def quantities(row):
            a=row['accounting']
            return dict(heat=a['sampled_integrals_j']['nuclear_heat_w'],
                export=a['electrical']['energy_delta_j']['net_exported_j'],
                cooling=a['sampled_integrals_j']['external_cooling_heat_out_w'])
        for left,right in [('coarse','medium'),('medium','fine')]:
            a,b=quantities(results[left]),quantities(results[right])
            comparisons[left+'_'+right]={k:abs(a[k]-b[k])/max(abs(b[k]),1.) for k in a}
    checks=dict(complete=complete,all_window_balances=all(r['passed'] for r in results.values()),
        refined_integrals=complete and max(comparisons['medium_fine'].values())<2e-6,
        refinement_trend=complete and all(comparisons['medium_fine'][k]<=max(comparisons['coarse_medium'][k],1e-8) for k in ('heat','export','cooling')),
        grid_connected_no_trip=complete and all(r['accounting']['electrical']['all_sampled_grid_connected'] and not r['accounting']['electrical']['any_sampled_trip'] for r in results.values()))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Heat qualification source changed')
    report=dict(passed=all(checks.values()),source=source,driver_sha256=driver,checks=checks,
        comparisons=comparisons,horizon_s=1.,completed_paths=sum(r['complete'] for r in results.values()),
        sampled_heat_integral_qualified=all(checks.values()),
        stationary_efficiency_certified=False,isolated_quadrature_error_bound=False,
        industrial_thermal_efficiency_claimed=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Sampled heat qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
