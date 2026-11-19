"""Short pressure-relief-opening audit, retaining longer-window timing refusal."""
import argparse
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRBypassSession,BypassConfig
from atomic_model_station_hybrid_heat import record_hybrid_window
from atomic_model_station_hybrid_heat.model import identity
from atomic_model_station_hybrid_heat.scenarios import prepare_relief_opening


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    rows={};checks={};failure=None;comparisons=[]
    try:
        session,patch,recipe=prepare_relief_opening();saved=session.checkpoint()
        atomic_json(out/'recipe.json',seal(recipe))
        checks['initially_unlatched']=not session.relief_latched
        for count in (8,16,32):
            row=record_hybrid_window(session,recipe['duration_s'],patch,subdivisions=count);rows[count]=row
            atomic_json(out/f'n{count}.json',seal(row))
        checks['balances']=all(r['passed'] for r in rows.values())
        checks['relief_root']=all(len(r['events'])==1 and r['events'][0]['kind']=='bypass_relief_open' and
            session.time<r['events'][0]['time_s']<session.time+recipe['duration_s'] and
            abs(r['events'][0]['event']['pressure_pa']-recipe['bypass']['relief_open_pa'])<=.05 for r in rows.values())
        checks['readonly']=session.checkpoint()==saved
        checks['restart']=record_hybrid_window(PWRBypassSession.restore(saved),recipe['duration_s'],subdivisions=16)==rows[16]
        for a,b in ((8,16),(16,32)):
            x=rows[a]['accounting']['sampled_integrals_j'];y=rows[b]['accounting']['sampled_integrals_j']
            comparisons.append({k:abs(x[k]-y[k])/max(1.,abs(y[k]),abs(y['nuclear_heat_w']) if k=='external_net_w' else 0.) for k in x})
        checks['refinement']=max(comparisons[-1].values())<=2e-6
        checks['trend']=all(comparisons[1][k]<=max(comparisons[0][k],1e-8) for k in comparisons[0])
        def fresh():return PWRBypassSession(bypass=BypassConfig(**recipe['bypass']),rtol=1e-10,max_step_s=.01)
        control=record_hybrid_window(fresh(),.02,subdivisions=8)
        atomic_json(out/'no-trip-control.json',seal(control))
        checks['no_trip_no_relief']=control['passed'] and not control['events']
        refused=record_hybrid_window(fresh(),.02,{'turbine_trip':True},subdivisions=16)
        atomic_json(out/'long-window-refusal.json',seal(refused))
        checks['long_window_refused']=not refused['passed'] and not refused['checks']['root_match']
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    report=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,comparisons=comparisons,
        passed=failure is None and bool(checks) and all(checks.values()),window_s=[.016,.020],
        qualified_event='bypass_relief_open',rtol=1e-10,subdivisions=[8,16,32],
        full_pressure_cycle_qualified=False,station_env_relief_qualified=False,
        simultaneous_physical_scenario_qualified=False,plant_calibrated=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(report))
    if not report['passed']:raise ValueError('Relief heat qualification failed; evidence retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
