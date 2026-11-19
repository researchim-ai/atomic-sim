"""Local two-sided backup-exhaustion accounting with independent subdivisions."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRPowerSession,BackupConfig
from atomic_model_station_jump_heat import record_power_window
from atomic_model_station_jump_heat.model import identity


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    session=PWRPowerSession(backup=BackupConfig(initial_energy_j=1e4),rtol=1e-10)
    # Feed closure during grid loss is deliberately outside this qualified window.
    session.advance(.08,{'grid_available':False});saved=session.checkpoint()
    rows={};checks={};failure=None;comparisons=[]
    try:
        for n in (8,16,32,64):
            row=record_power_window(session,.04,subdivisions=n);rows[n]=row
            atomic_json(out/f'n{n}.json',seal(row))
        restored=PWRPowerSession.restore(saved)
        replay=record_power_window(restored,.04,subdivisions=16)
        checks['restart']=replay==rows[16]
        checks['readonly']=session.checkpoint()==saved
        checks['coarse_refused']=not rows[8]['passed']
        checks['refined_balances']=all(rows[n]['passed'] for n in (16,32,64))
        checks['two_sided_root']=all(len(r['events'])==1 and abs(r['events'][0]['time_s']-.1)<=1e-9 and
            r['events'][0]['before']['energy']['powers']['own_needs_w']==1e5 and
            r['events'][0]['after']['energy']['powers']['own_needs_w']==0. for r in rows.values())
        for a,b in ((16,32),(32,64)):
            x=rows[a]['accounting']['sampled_integrals_j'];y=rows[b]['accounting']['sampled_integrals_j']
            comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
        checks['refined_integrals']=max(comparisons[-1].values())<=2e-6
        checks['trend']=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0])
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,passed=failure is None and bool(checks) and all(checks.values()),
        checks=checks,failure=failure,comparisons=comparisons,window_s=[.08,.12],backup_initial_j=1e4,
        backup_load_w=1e5,rtol=1e-10,refinements=[16,32,64],coarse_negative=8,
        station_env_integration=False,discontinuous_valves_supported=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Jump heat qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
