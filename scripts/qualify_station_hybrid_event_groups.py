"""Condensate-root qualification and regression of grouped-event assembly."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRPowerSession,FeedConfig
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat.model import identity
from scripts.qualify_station_hybrid_heat import run as feed_regression


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    rows={};checks={};failure=None;comparisons=[]
    try:
        cfg=FeedConfig(condensate_inertance_m_inv=1.)
        session=PWRPowerSession(feed=cfg,rtol=1e-10,max_step_s=.01)
        session.advance(.005,{'condensate_pump_on':False});saved=session.checkpoint()
        for count in (8,16,32):
            row=record_hybrid_window(session,.01,subdivisions=count);rows[count]=row
            atomic_json(out/f'n{count}.json',seal(row))
        checks['balances']=all(r['passed'] for r in rows.values())
        checks['condensate_roots']=all([e['kind'] for e in r['events']]==['condensate_check_closed'] for r in rows.values())
        checks['readonly']=session.checkpoint()==saved
        restored=PWRPowerSession.restore(saved)
        checks['restart']=record_hybrid_window(restored,.01,subdivisions=16)==rows[16]
        for a,b in ((8,16),(16,32)):
            x=rows[a]['accounting']['sampled_integrals_j'];y=rows[b]['accounting']['sampled_integrals_j']
            comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
        checks['refinement']=max(comparisons[-1].values())<=2e-6
        checks['trend']=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0])
        unprepared=PWRPowerSession(feed=cfg,rtol=1e-10)
        refused=record_hybrid_window(unprepared,.02,{'condensate_pump_on':False},subdivisions=16)
        atomic_json(out/'unprepared-refusal.json',seal(refused))
        checks['longer_window_refused']=not refused['passed'] and not refused['checks']['root_match']
        regression=feed_regression(out/'feed-regression')
        checks['feed_and_backup_regression']=regression['passed']
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,comparisons=comparisons,
        passed=failure is None and bool(checks) and all(checks.values()),window_s=[.005,.015],
        condensate_inertance_m_inv=1.,rtol=1e-10,subdivisions=[8,16,32],
        simultaneous_assembly_supported=True,simultaneous_physical_scenario_qualified=False,
        internal_opening_qualified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Event-group qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
