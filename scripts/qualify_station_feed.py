"""Local finite feed-train qualification with source-bound installed replay."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
from core.status import DomainError,SolverError
from atomic_model_station import PWRFeedSession,FeedConfig
from atomic_model_station.pwr_train import physical_signature
from atomic_model_multiphysics.provenance import sha
from scripts import qualify_station_actuators as evidence

TASKS=['NPP-07.02','NPP-07.03','NPP-07.04']
SETTINGS=[(.2,1e-7),(.1,2e-8),(.05,2e-9)]
CRITERIA=dict(energy_relative=1e-6,mass_relative=1e-10,boron_relative=1e-10,instantaneous_relative=1e-8,
    feed_power_residual_w=1e-4,condenser_power_residual_w=1e-5,refinement_relative=1e-4,tail_drift_relative=1e-5,root_residual_kg_s=1e-6)


def vector(r):return np.r_[r['pwr_state'],r['turbine_energy_j'],r['clad_temperature_k'],r['cooling_energy_j'],r['cooling']['flow_kg_s'],r['feed_inventory']]


def admit(r):
    return bool(r['energy_relative']<=CRITERIA['energy_relative'] and r['mass_relative']<=CRITERIA['mass_relative'] and r['boron_relative']<=CRITERIA['boron_relative']
        and abs(r['instantaneous_energy_defect_w'])/max(abs(r['powers']['nuclear_heat_w']),1.)<=CRITERIA['instantaneous_relative']
        and abs(r['feed_train']['internal_energy_residual_w'])<=CRITERIA['feed_power_residual_w']
        and abs(r['condenser_balance']['energy_residual_w'])<=CRITERIA['condenser_power_residual_w'])


def run(output,reference=None):
    output=Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Symlink output')
    output.mkdir(parents=True,exist_ok=False)
    source=physical_signature();driver=sha(__file__);serializer=sha(evidence.__file__)
    s=PWRFeedSession(max_step_s=1.);initial=s.snapshot()
    evidence.save(output/'protocol.json',dict(source=source,driver_sha256=driver,serializer_sha256=serializer,criteria=CRITERIA,settings=SETTINGS,
        initial=initial,parameters=s.parameter_registry(),nominal_duration_s=3600.,candidate_tasks=TASKS,
        start='Declared hot start with physical thermal settling from extra pump losses, not projected station equilibrium',
        topology='Hotwell -> condensate pump -> sealed gas-cushioned feed tank -> feed pump/check/valve -> SG; finite makeup and drain vessels',
        gas='Ideal adiabatic cushions gamma1.4; water pressure work and stored gas energy cancel internally',
        hydraulics='Finite pipe momenta, capped controller heads, exact closed check modes, explicit root residual energy returned to tank',
        scope='Reduced hot feed scheme; no deaerator/heaters, steam lines/MSR or global blackout admission'))
    print('finite feed protocol sealed',flush=True)
    rows=[]
    for i in range(60):
        rows.append(s.advance(60.))
        if i%5==0:print('finite feed nominal',i+1,'/60',flush=True)
    tail=float(np.max(abs(vector(rows[-1])-vector(rows[-3]))/np.maximum(abs(vector(rows[-3])),1.)))
    evidence.save(output/'nominal.json',dict(trace=rows,tail_relative_drift=tail,checkpoint=s.checkpoint()))
    observations=[initial,*rows];cases=[]
    for name in ('feed_trip','condensate_trip','transfer','head_saturation'):
        trials=[]
        for step,rtol in SETTINGS:
            t=PWRFeedSession(max_step_s=step,rtol=rtol);start=t.snapshot()
            if name=='feed_trip':
                a=t.advance(.2,{'feed_pump_on':False});b=t.advance(.5,{'feed_pump_on':True})
                behavior=a['feed_train']['feed_flow_kg_s']==0 and not a['feed_train']['feed_check_open'] and b['feed_train']['feed_flow_kg_s']>0
            elif name=='condensate_trip':
                a=t.advance(2.,{'condensate_pump_on':False});b=t.advance(1.,{'condensate_pump_on':True})
                behavior=a['feed_train']['condensate_flow_kg_s']==0 and a['feed_train']['feed_flow_kg_s']>0 and b['feed_train']['condensate_flow_kg_s']>0
            elif name=='transfer':
                a=t.advance(2.,{'makeup_valve':1.,'drain_valve':.4});b=t.advance(1.,{'makeup_valve':0.,'drain_valve':0.})
                behavior=a['feed_train']['tanks']['makeup']['mass_kg']<start['feed_train']['tanks']['makeup']['mass_kg'] and a['feed_train']['tanks']['drain']['mass_kg']>start['feed_train']['tanks']['drain']['mass_kg']
            else:
                a=t.advance(.1,{'feed_pump_fraction':.1});b=t.advance(.2,{'feed_pump_fraction':1.})
                behavior=a['feed_train']['feed_controller']['saturated'] and a['feed_train']['feed_flow_kg_s']==0 and b['feed_train']['feed_flow_kg_s']>0
            events=[e for e in b['events'] if e['kind'].endswith('check_closed')]
            roots=all(abs(e['root_residual_kg_s'])<=CRITERIA['root_residual_kg_s'] for e in events)
            trials.append(dict(max_step_s=step,rtol=rtol,initial=start,disturbed=a,recovered=b,behavior=bool(behavior),roots_admitted=roots))
            observations.extend([a,b]);print('finite feed case',name,step,flush=True)
        fine=vector(trials[-1]['recovered']);errors=[float(np.max(abs(vector(t['recovered'])-fine)/np.maximum(abs(fine),1.))) for t in trials]
        cases.append(dict(name=name,trials=trials,refinement_errors=errors))
    evidence.save(output/'transitions.json',dict(cases=cases))
    t=PWRFeedSession(max_step_s=.2);t.advance(.2,{'feed_pump_on':False});saved=json.loads(json.dumps(t.checkpoint()))
    restored=PWRFeedSession.restore(saved);resumed=t.advance(.5,{'feed_pump_on':True});exact=restored.advance(.5,{'feed_pump_on':True})==resumed
    observations.append(resumed)
    depleted=PWRFeedSession(feed=FeedConfig(makeup_mass_kg=21.),max_step_s=.2);before=depleted.checkpoint();rollback=False
    try:depleted.advance(1.,{'makeup_valve':1.})
    except (ValueError,DomainError,SolverError):rollback=depleted.checkpoint()==before
    suction=False
    try:PWRFeedSession(feed=FeedConfig(suction_static_head_m=1.))
    except ValueError:suction=True
    evidence.save(output/'guards.json',dict(checkpoint=saved,resumed=resumed,restart_exact=exact,depletion_rollback=rollback,suction_refused=suction))
    all_admitted=all(admit(r) for r in observations);behavior=all(t['behavior'] and t['roots_admitted'] for c in cases for t in c['trials'])
    checks={'NPP-07.02':bool(all_admitted and behavior and exact),'NPP-07.03':bool(all_admitted and behavior and rollback),
            'NPP-07.04':bool(all_admitted and behavior and rollback and suction)}
    unchanged=source==physical_signature() and driver==sha(__file__) and serializer==sha(evidence.__file__)
    passed=bool(unchanged and all(checks.values()) and tail<=CRITERIA['tail_drift_relative'] and all(max(c['refinement_errors'])<=CRITERIA['refinement_relative'] for c in cases))
    report=dict(completed=True,passed=passed,source_unchanged=unchanged,task_checks=checks,closed_tasks=TASKS if passed else [],
        nominal_duration_s=rows[-1]['time_s'],nominal_tail_relative_drift=tail,refinement_errors={c['name']:c['refinement_errors'] for c in cases},
        max_energy_relative=max(r['energy_relative'] for r in observations),max_mass_relative=max(r['mass_relative'] for r in observations),
        max_boron_relative=max(r['boron_relative'] for r in observations),restart_exact=exact,depletion_rollback=rollback,suction_refused=suction,
        complete_package_b=False,complete_package_c=False,r1_accepted=False,validated=False,plant_referenced=False,full_nominal_checkpoint_replay=False,
        limitations=['Research hot feed train, constant liquid density/cp and ideal adiabatic gas cushions',
            'Suction threshold illustrative, not a manufacturer NPSH/cavitation map; cavitation is rejected',
            'No deaerator or feed heaters, no surveyed equipment or calibrated pump curves',
            'Finite pure-water makeup/drain; no chemical-volume-control boron supply',
            'No station-wide electrical blackout, reserve supply, autonomous protection or long loss-of-feed safety admission',
            'Original phase, density, inactive spray and turbine moisture limits remain'])
    evidence.save(output/'report.json',report)
    if reference is not None:
        reference=Path(reference);files={p.relative_to(output) for p in output.rglob('*.json')}
        if files!={p.relative_to(reference) for p in reference.rglob('*.json')}:raise ValueError('Evidence file sets differ')
        for path in sorted(files):
            if evidence.read(output/path)!=evidence.read(reference/path):raise ValueError('Installed feed evidence differs:'+str(path))
        print('All installed finite feed evidence matches exactly',flush=True)
    if not passed:raise ValueError('Finite feed qualification failed; no tasks accepted')
    print(json.dumps(dict(passed=passed,closed_tasks=report['closed_tasks'])),flush=True)
    return report


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    args=p.parse_args();run(args.output,args.reference)
