"""Qualify pressure-driven internal opening, separate from command-boundary opening."""
import argparse
from copy import deepcopy
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRPowerSession
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat.model import identity
from atomic_model_station_hybrid_heat.scenarios import prepare_condensate_reopening


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    rows={};checks={};failure=None;comparisons=[]
    try:
        session,patch,recipe=prepare_condensate_reopening();saved=session.checkpoint()
        atomic_json(out/'recipe.json',seal(recipe))
        left=deepcopy(session);left._set_controls(patch);left._prepare_interval();left.snapshot()
        checks['closed_after_command']=not left.condensate_open and left.feed_terms['condensate_drive_pa']<0.
        for count in (8,16,32):
            row=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=count);rows[count]=row
            atomic_json(out/f'n{count}.json',seal(row))
        checks['balances']=all(r['passed'] for r in rows.values())
        checks['internal_opening']=all(len(r['events'])==1 and r['events'][0]['kind']=='condensate_check_opened' and
            session.time<r['events'][0]['time_s']<session.time+recipe['duration_s'] for r in rows.values())
        checks['readonly']=session.checkpoint()==saved
        checks['restart']=record_hybrid_window(PWRPowerSession.restore(saved),recipe['duration_s'],patch,subdivisions=16)==rows[16]
        for a,b in ((8,16),(16,32)):
            x=rows[a]['accounting']['sampled_integrals_j'];y=rows[b]['accounting']['sampled_integrals_j']
            comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
        checks['refinement']=max(comparisons[-1].values())<=2e-6
        checks['trend']=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0])
        control=record_hybrid_window(session,recipe['duration_s'],dict(patch,condensate_pump_fraction=0.),subdivisions=8)
        atomic_json(out/'no-head-control.json',seal(control))
        checks['no_head_no_event']=control['passed'] and not control['events']
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,comparisons=comparisons,
        passed=failure is None and bool(checks) and all(checks.values()),window_s=[.006,.026],
        qualified_event='condensate_check_opened',rtol=1e-10,subdivisions=[8,16,32],
        feed_internal_opening_qualified=False,simultaneous_physical_scenario_qualified=False,
        plant_calibrated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Internal reopening qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
