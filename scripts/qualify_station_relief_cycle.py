"""Complete declared short pressure cycle, including actuator arrivals and rollback."""
import argparse
from copy import deepcopy
from pathlib import Path
from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station import PWRBypassSession
from atomic_model_station_event_heat import HeatEventError
from atomic_model_station_hybrid_heat import advance_hybrid_schedule
from atomic_model_station_hybrid_heat.model import identity
from atomic_model_station_hybrid_heat.scenarios import prepare_relief_cycle


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    source=identity();driver=sha(__file__);out.mkdir(parents=True)
    checks={};failure=None
    try:
        session,schedule,recipe=prepare_relief_cycle();native=deepcopy(session)
        atomic_json(out/'recipe.json',seal(recipe))
        for i,row in enumerate(schedule):
            native.advance(row['duration_s'],row['patch'])
            if i==2:midpoint=native.checkpoint()
        report=advance_hybrid_schedule(session,schedule)
        atomic_json(out/'cycle.json',seal(report))
        atomic_json(out/'final-checkpoint.json',session.checkpoint())
        checks['native_equivalence']=session.checkpoint()==native.checkpoint()
        events=[e for w in report['windows'] for packet in w['replays'][-1]['events'] for e in packet['journal_events']]
        pressure=[e for e in events if e['kind'] in ('bypass_relief_open','bypass_relief_close')]
        checks['pressure_cycle']=[e['kind'] for e in pressure]==recipe['expected_pressure_events']
        checks['pressure_residuals']=all(abs(e['pressure_pa']-recipe['bypass']['relief_open_pa' if e['kind']=='bypass_relief_open' else 'relief_close_pa'])<=.05 for e in pressure)
        checks['actuator_return']=not session.relief_latched and float(session.state[session.bypass_index])==0. and any(e['kind']=='bypass_position_arrival' for e in events)
        checks['steam_arrivals']=sum(e['kind']=='steam_valve_arrived' for e in events)==2
        checks['all_windows']=report['passed'] and len(report['windows'])==6
        restored=PWRBypassSession.restore(midpoint)
        continued=advance_hybrid_schedule(restored,schedule[3:])
        atomic_json(out/'restart.json',seal(continued))
        checks['restart']=restored.checkpoint()==session.checkpoint() and all(a['replays']==b['replays'] for a,b in zip(continued['windows'],report['windows'][3:]))
        coarse,_,_=prepare_relief_cycle();before=coarse.checkpoint()
        try:
            advance_hybrid_schedule(coarse,[dict(duration_s=.01,patch={'steam_admission':.99}),dict(duration_s=.012,patch={'steam_admission':1.})])
            checks['coarse_refused']=False
        except HeatEventError as error:
            atomic_json(out/'coarse-refusal.json',seal(error.receipt))
            checks['coarse_refused']=error.receipt['reason']=='schedule_heat_refused' and coarse.checkpoint()==before
    except Exception as error:failure=dict(type=type(error).__name__,message=str(error),receipt=getattr(error,'receipt',None))
    if source!=identity() or driver!=sha(__file__):raise ValueError('Source changed')
    result=dict(source=source,driver_sha256=driver,checks=checks,failure=failure,
        passed=failure is None and bool(checks) and all(checks.values()),duration_s=.022,
        specific_pressure_cycle_qualified=failure is None and bool(checks) and all(checks.values()),
        general_pressure_cycle_qualified=False,full_actuator_stroke_qualified=False,
        station_env_cycle_qualified=False,simultaneous_physical_roots_qualified=False,closed_tasks=[])
    atomic_json(out/'report.json',seal(result))
    if not result['passed']:raise ValueError('Pressure cycle qualification failed; evidence retained')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output)['checks'])
