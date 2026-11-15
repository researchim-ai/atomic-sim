"""Frozen R1 research matrix. A catalog or short prefix is never acceptance."""
from pathlib import Path
import argparse
import json

from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_training.spec import atomic_json
from atomic_model_station_faults import Fault
from .spec import Episode,Limits
from .engine import signature
from .runner import run_campaign

DURATIONS=dict(nominal=3600,load_cycle=1800,load_rejection=7200,scram=600,
    primary_coastdown=1800,feed_loss=600,cooling_loss=600,grid_loss=600,
    sensor_bias=1800,sensor_drift=1800,sensor_stuck=1800,coincident=120)
SETTINGS=((1.,1e-7),(.5,2e-8),(.25,5e-9))


def catalog():
    families=[]
    for name,duration in DURATIONS.items():
        ticks=duration//5;commands=[];faults=[];react=False
        stop=dict(scram=True,turbine_trip=True,dhr_requested=True)
        if name=='load_cycle':
            commands=[dict(tick=t,operator=dict(load_fraction=f),targets=dict(rod_target=3.15e9*f))
                      for t,f in ((90,.98),(180,1.02),(270,1.))]
        elif name=='load_rejection':commands=[dict(tick=0,operator=dict(generator_connected=False,dhr_requested=True))]
        elif name=='scram':commands=[dict(tick=0,operator=stop)]
        elif name=='primary_coastdown':commands=[dict(tick=0,operator=dict(stop,primary_pump_on=False))]
        elif name=='feed_loss':faults=[Fault('feed_drive_loss','pump_off','feed',1)]
        elif name=='cooling_loss':faults=[Fault('external_supply_loss','cooling_loss','external_water',1,value=.5)]
        elif name=='grid_loss':commands=[dict(tick=0,operator=dict(grid_available=False,dhr_requested=True,backup_enabled=True))]
        elif name.startswith('sensor_'):
            kind=name.removeprefix('sensor_');value=150000. if kind=='bias' else 1000. if kind=='drift' else 0.
            faults=[Fault('hidden_measurement_fault',kind,'pressure',1,value=value)];react=True
        elif name=='coincident':
            faults=[Fault('hidden_measurement_fault','bias','pressure',1,value=150000.),
                    Fault('feed_limit','drive_limit','feed',1,value=.72)]
            commands=[dict(tick=1,operator=dict(stop,steam_bypass=.8,steam_admission=1.))];react=True
        variants=[]
        for index,(step,rtol) in enumerate(SETTINGS):
            e=Episode(name+'_'+str(index),ticks,tick_s=5.,faults=tuple(faults),commands=tuple(commands),
                react_on_alarm=react,max_step_s=step,rtol=rtol,record_every=max(1,(ticks+1023)//1024),
                checkpoint_ticks=(0,1,ticks//2,ticks))
            variants.append(e.payload())
        families.append(dict(name=name,duration_s=duration,variants=variants,restart_tick=1))
    return seal(dict(format='station-r1-matrix/v1',source=signature(),families=families,
        criteria=dict(continuous_relative=1e-3,event_time_s=.05,
            tail_sg_level_m=.02,tail_hotwell_level_m=.005,tail_feed_level_m=.005,
            tail_power_relative=.02,tail_speed_relative=.01,tail_pressure_relative=.02),
        full_r1_accepted=False,scope='Declared hot research PWR; all durations and all three numerical settings are required'))


def _events(science):
    state=science['final']['payload']['control']['payload']['state']
    physical=state['plant']['payload']['snapshot']['events']
    # Numerical actuator arrivals are diagnostics; retain physical/protection switches.
    ignore={'rod_arrived','bypass_position_arrival','steam_inlet_arrival','steam_outlet_arrival'}
    selected=[(e['kind'],e['time_s']) for e in physical if 'command' not in e['kind'] and e['kind'] not in ignore]
    selected.extend((e['kind'],e['time_s']) for e in state['events'] if e['kind'] in ('protection_latched','pressure_interlock_entered','pressure_interlock_released'))
    return sorted(selected,key=lambda x:(x[1],x[0]))


def _nominal(science,criteria):
    tail=[s for s in science['samples'] if s['physical']['time_s']>=3000.]
    if not tail:return False
    specs=[('sg_collapsed_level_m','tail_sg_level_m'),('hotwell_level_m','tail_hotwell_level_m'),('feed_tank_level_m','tail_feed_level_m')]
    levels=all(max(s['physical']['levels'][k] for s in tail)-min(s['physical']['levels'][k] for s in tail)<=criteria[c] for k,c in specs)
    return bool(levels and all(not s['result']['receipt']['trip_latched']
        and abs(s['result']['observation']['measurements']['power']['value']/3.15e9-1)<=criteria['tail_power_relative']
        and abs(s['physical']['turbine_speed_rad_s']/157.-1)<=criteria['tail_speed_relative']
        and abs(s['physical']['pressure_pa']/6.3e6-1)<=criteria['tail_pressure_relative']
        and s['physical']['powers']['gross_electrical_w']>s['physical']['powers']['own_needs_w']>=0 for s in tail))


def run_matrix(output,*,families=None,limits=None):
    manifest=catalog();p=unseal(manifest);known={f['name']:f for f in p['families']}
    names=list(known) if families is None else list(families)
    if not names or len(names)!=len(set(names)) or not set(names)<=set(known):raise ValueError('Matrix family selection')
    output=Path(output).absolute()
    if any(q.is_symlink() for q in (output,*output.parents)):raise ValueError('Symlink output refused')
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'matrix.json',manifest)
    results=[]
    for name in names:
        family=known[name];episodes=[Episode.from_payload(e) for e in family['variants']]
        normal=run_campaign(output/name/'normal',episodes,limits=limits)
        row=dict(family=name,normal_accepted=normal['accepted'],restart_accepted=False,refinement_relative=None,event_agreement=False)
        if normal['accepted']:
            baseline={e.name:unseal(json.loads((output/name/'normal'/(e.name+'.json')).read_text()))['scientific'] for e in episodes}
            anchors={n:unseal(s)['checkpoints'][str(family['restart_tick'])] for n,s in baseline.items()}
            replay=run_campaign(output/name/'restart',episodes,limits=limits,resumes=anchors)
            exact=replay['accepted'] and all(unseal(json.loads((output/name/'restart'/(n+'.json')).read_text()))['scientific']==s for n,s in baseline.items())
            row['restart_accepted']=bool(exact)
            sciences=[unseal(baseline[e.name]) for e in episodes]
            x=sciences[-1]['physical_state']
            row['refinement_relative']=max(abs(a-b)/max(abs(b),1.) for s in sciences[:-1] for a,b in zip(s['physical_state'],x))
            events=[_events(s) for s in sciences]
            row['event_agreement']=all(len(a)==len(events[-1]) and all(k==j and abs(t-u)<=p['criteria']['event_time_s'] for (k,t),(j,u) in zip(a,events[-1])) for a in events[:-1])
            row['nominal_stationary']=all(_nominal(s,p['criteria']) for s in sciences) if name=='nominal' else None
        row['accepted']=bool(row['normal_accepted'] and row['restart_accepted'] and row['event_agreement']
            and row['refinement_relative']<=p['criteria']['continuous_relative'] and row.get('nominal_stationary') is not False)
        results.append(row)
        atomic_json(output/(name+'.json'),seal(row))
    full=set(names)==set(known);accepted=full and all(r['accepted'] for r in results) and p['source']==signature()
    report=dict(completed=len(results)==len(names),selected_families=names,full_matrix_attempted=full,
        matrix_numerically_accepted=bool(accepted),families=results,closed_tasks=[],r1_accepted=False,validated=False,
        qualification_scope='Numerical matrix only; task publication and physical domain review remain separate')
    atomic_json(output/'report.json',seal(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--family',action='append');parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--wall-s',type=int,default=7200);args=parser.parse_args()
    result=run_matrix(args.output,families=args.family,limits=Limits(workers=args.workers,wall_s=args.wall_s,cpu_s=args.wall_s))
    print(json.dumps(result,sort_keys=True));return 0 if all(r['accepted'] for r in result['families']) else 1


if __name__=='__main__':raise SystemExit(main())
