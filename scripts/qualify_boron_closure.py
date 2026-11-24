"""Finite closing tail after one minute of liquid boron exchange."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from scripts import qualify_boron_ports as ports
from scripts import qualify_boron_duration as duration

CASES=ports.CASES
PRESETS=dict(coarse=dict(max_step_s=.2,rtol=2e-8),fine=dict(max_step_s=.1,rtol=2e-9))
TIMES=(*duration.TIMES,60.1,60.2,60.5,61.)
STEPS=[*deepcopy(duration.STEPS),dict(duration_s=.1,patch={'boron_flow_target_kg_s':0.}),
    dict(duration_s=.1,patch=None),dict(duration_s=.3,patch=None),dict(duration_s=.5,patch=None)]
CRITERIA=dict(temporal_relative=1e-4,reference_max_nodes=8192,reference_normalized_tolerance=1e-11,
    reference_heat_relative=1e-8,minimum_primary_boron_change_kg=.001,minimum_power_effect_relative=1e-4,numerical_margin=20.)


def source():
    return dict(model=identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (duration,ports,ports.regen,ports.regen.chemical,ports.regen.chemical.budgets,ports.refinement)})


def tail_reference(x,flow,*,max_nodes=8192):
    """Independent normalized ODE shooting, integrated in the stable direction.

    Solve e'=-abs(k)*e and J'=e from (1,0). Rescale the resulting
    gap profile to satisfy theta_h(0)=1 and theta_h(1)=d(1).
    No effectiveness or analytical exponential is used for the heat reference.
    """
    if type(max_nodes) is not int or not 8<=max_nodes<=10000:
        raise ValueError('Bounded reference budget required')
    hot=x['hot_inlet_temperature_k'];cold=x['cold_inlet_temperature_k'];ua=x['ua_w_k']
    cp_h=x['hot_cp_j_kg_k'];cp_c=x['cold_cp_j_kg_k']
    if (any(type(v) not in (float,int) or not math.isfinite(v) for v in (hot,cold,ua,cp_h,cp_c,flow))
            or not (0<=flow<=1e6 and 0<=ua<=1e8 and 1<=hot<=2000 and 1<=cold<=2000 and 1<=cp_h<=1e5 and 1<=cp_c<=1e5)):
        raise ValueError('Finite physical capacity reference inputs required')
    if flow==0 or ua==0 or hot==cold:return dict(heat_w=0.,hot_outlet_k=hot,cold_outlet_k=cold)
    ch=flow*cp_h;cc=flow*cp_c
    if min(ch,cc)<=0 or not math.isfinite(ch+cc):raise ValueError('Finite positive reference capacities required')
    a=ua/ch;k=ua*(1./cc-1./ch)
    if not math.isfinite(a) or not math.isfinite(k):raise ValueError('Finite reference differential coefficients required')
    def rhs(z,y):return [-abs(k)*y[0],y[0]]
    result=solve_ivp(rhs,(0.,1.),[1.,0.],method='Radau',rtol=CRITERIA['reference_normalized_tolerance'],atol=1e-13)
    if not result.success or len(result.t)>max_nodes:raise ValueError('Stable temperature-gap reference did not converge within budget')
    end,integral=map(float,result.y[:,-1])
    denominator=a*integral+(1. if k>=0 else end)
    if not all(math.isfinite(v) for v in (end,integral,denominator)) or integral<=0 or denominator<=0:
        raise ValueError('Invalid integrated gap reference')
    gap0=(end if k>=0 else 1.)/denominator
    gap1=(1. if k>=0 else end)/denominator
    recovered=a*integral/denominator
    # Boundary and integral conservation residuals; never project a failed solve.
    if abs(1.-recovered-gap1)>1e-10 or abs(gap1-gap0-k*integral/denominator)>1e-10:
        raise ValueError('Temperature-gap reference residual failed')
    difference=hot-cold
    return dict(heat_w=ch*difference*recovered,hot_outlet_k=cold+difference*(1.-recovered),
        cold_outlet_k=hot-difference*gap0)


def reference_mass(time):
    if type(time) not in (int,float) or not math.isfinite(time) or not 0<=time<=61.:
        raise ValueError('Closing protocol time must be in 0..61 seconds')
    return ports.regen.chemical.transferred_mass(time,flow=1.,tau=.2,close=60.)


def audit(row,metadata):
    return ports.regen.audit(row,metadata,reference_max_nodes=CRITERIA['reference_max_nodes'],
        reference_solver=tail_reference,expected_transferred_mass=reference_mass(row['time_s']))


def assess(cases):
    if set(cases)!=set(CASES) or any(set(t)!=set(PRESETS) for t in cases.values()):raise ValueError('Complete minute-and-closing-tail chemical/time matrix required')
    gates=dict(configuration=True,timeline=True,commands=True,budgets=True,transfer=True,eos=True,restart=True,
        rollback=True,reference_frozen=True,matched_initial=True,direction=True,closing_flow=True)
    errors=[];margins=[];receiver_temperatures=[];changes={};remaining={}
    for case,trajectories in cases.items():
        for preset,t in trajectories.items():
            if len(t['snapshots'])!=len(TIMES):raise ValueError('Complete minute-and-closing-tail observations required')
            gates['configuration'] &= t['checkpoint']['payload']['config']==asdict(BoronExchangeConfig(supply_boron_ppm=CASES[case],pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True))
            gates['commands'] &= t['steps']==STEPS and t['checkpoint']['payload']['history']==STEPS
            gates['timeline'] &= [r['time_s'] for r in t['snapshots']]==list(TIMES)
            gates['rollback'] &= t['rollback'] is True
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            for row in [t['initial'],*t['snapshots']]:
                gates['budgets'] &= all(ports.regen.chemical.budgets.checks(row).values())
                gates['transfer'] &= all(audit(row,t['metadata']).values())
                gates['eos'] &= ports.audit_ports(row)
                time=row['time_s'];x=row['finite_boron_exchange']
                expected_q=(1.-math.exp(-min(time,60.)/.2))*math.exp(-max(0.,time-60.)/.2)
                gates['closing_flow'] &= math.isclose(x['flow_kg_s'],expected_q,rel_tol=1e-12,abs_tol=1e-14)
                gates['closing_flow'] &= x['target_kg_s']==(1. if 0<time<=60. else 0.)
                gates['reference_frozen'] &= row['local_moderator']['reference']==t['initial']['local_moderator']['reference']
                g=row['finite_boron_exchange']['liquid_ports']
                if g['checked']:margins.extend(x['subcooling_k'] for x in g['states'].values())
                receiver_temperatures.append(row['finite_boron_exchange']['tanks']['receiver']['temperature_k'])
            delta=ports.regen.chemical.primary_boron(t['snapshots'][-1])-ports.regen.chemical.primary_boron(t['initial'])
            changes[case+'_'+preset]=delta
            gates['direction'] &= delta<-CRITERIA['minimum_primary_boron_change_kg'] if case=='dilution' else delta>CRITERIA['minimum_primary_boron_change_kg']
            remaining[case+'_'+preset]=t['snapshots'][-1]['finite_boron_exchange']['tanks']['supply']['mass_kg']
        errors.extend(ports.regen.chemical.budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots']))
    a,b=(cases[c]['fine'] for c in CASES);base=a['metadata']['base_size']
    gates['matched_initial'] &= a['initial']['coupled_state'][:base]==b['initial']['coupled_state'][:base]
    p0=a['initial']['powers']['nuclear_fission_w']
    effect=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in zip(a['snapshots'],b['snapshots']))
    noise=max(abs(a['powers']['nuclear_fission_w']-b['powers']['nuclear_fission_w'])/p0 for t in cases.values() for a,b in zip(t['coarse']['snapshots'],t['fine']['snapshots']))
    gates.update(temporal=max(errors)<=CRITERIA['temporal_relative'],effect_resolved=effect>=CRITERIA['minimum_power_effect_relative'] and effect>=CRITERIA['numerical_margin']*noise)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=max(errors),
        minimum_subcooling_k=min(margins),maximum_receiver_temperature_k=max(receiver_temperatures),
        remaining_supply_mass_kg=remaining,primary_boron_changes_kg=changes,power_effect_relative=effect,numerical_power_relative=noise,
        exact_transferred_mass_kg=reference_mass(61.),
        residual_flow_kg_s=math.exp(-5.),untransferred_tail_to_infinity_kg=.2*math.exp(-5.))


def negative_controls(cases):
    checks={}
    def reject(name,modify,gate):
        copied=deepcopy(cases);modify(copied['dilution']['fine']);r=assess(copied)
        checks[name]=not r['passed'] and not r['gates'][gate]
    def last(t):return t['snapshots'][-1]
    reject('supply_mass_drift',lambda t:last(t)['finite_boron_exchange']['tanks']['supply'].__setitem__('mass_kg',5000.),'transfer')
    reject('receiver_mass_drift',lambda t:last(t)['finite_boron_exchange']['tanks']['receiver'].__setitem__('mass_kg',5000.),'transfer')
    reject('wrong_heat_recovery',lambda t:last(t)['finite_boron_exchange']['regenerator'].__setitem__('heat_w',0.),'transfer')
    reject('disabled_liquid_check',lambda t:last(t)['finite_boron_exchange']['liquid_ports'].__setitem__('enabled',False),'eos')
    reject('wrong_end_time',lambda t:last(t).__setitem__('time_s',60.9),'timeline')
    reject('tail_clipped_to_zero',lambda t:last(t)['finite_boron_exchange'].__setitem__('flow_kg_s',0.),'closing_flow')
    reject('closing_command_lost',lambda t:t['steps'][5].__setitem__('patch',None),'commands')
    reject('failed_late_restart',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('energy_budget_failure',lambda t:last(t).__setitem__('energy_relative',1.),'budgets')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies, not additional physical trajectories')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for case,ppm in CASES.items():
        cases[case]={}
        for preset,options in PRESETS.items():
            print(case,preset,flush=True)
            config=BoronExchangeConfig(supply_boron_ppm=ppm,pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True)
            s=BoronExchangePWRSession(**ports.refinement.coupled_pwr_mesh(2),boron=config,**options)
            first=s.snapshot();rows=[];saved=None
            meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,
                supply_mass=config.supply_mass_kg,receiver_mass=config.receiver_mass_kg,supply_fraction=ppm*1e-6,ua_w_k=config.regenerator_ua_w_k,base_size=s.boron_index)
            for i,step in enumerate(STEPS):
                rows.append(s.advance(**step));print(case,preset,s.time,flush=True)
                if i==6:saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False))
            restart=None
            if preset=='fine':
                restored=BoronExchangePWRSession.restore(saved)
                restart=all(restored.advance(**step)==row for step,row in zip(STEPS[7:],rows[7:])) and restored.checkpoint()==s.checkpoint()
            before=s.checkpoint()
            try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            t=dict(initial=first,snapshots=rows,steps=deepcopy(STEPS),metadata=meta,restart_exact=restart,rollback=rollback,checkpoint=before)
            cases[case][preset]=t;atomic_json(out/f'{case}_{preset}.json',seal(t))
    assessment=assess(cases);negative=negative_controls(cases)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,negative_controls=negative,
        passed=assessment['passed'] and negative['passed'] and source()==frozen,trajectories=4,restart_suffixes=2,
        horizon_s=61.,restart_time_s=60.2,closure_command_time_s=60.,experimental_validation=False,hour_qualified=False,spatial_qualification=False,
        station_env_connected=False,closed_tasks=[],scope='1 kg/s command for 60 seconds then 1-second finite closing tail; no exact zero-flow, depletion-event or plant-long-duration qualification')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Minute-and-closing-tail chemical qualification failed; attempts retained')
    return dict(passed=True,assessment=assessment)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
