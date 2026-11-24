"""One-minute liquid boron exchange; no physical source changes or hour claim."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from scripts import qualify_boron_ports as ports

CASES=ports.CASES
PRESETS=dict(coarse=dict(max_step_s=.2,rtol=2e-8),fine=dict(max_step_s=.1,rtol=2e-9))
TIMES=(.1,1.,10.,30.,60.)
STEPS=[dict(duration_s=.1,patch={'boron_flow_target_kg_s':1.}),dict(duration_s=.9,patch=None),
    dict(duration_s=9.,patch=None),dict(duration_s=20.,patch=None),dict(duration_s=30.,patch=None)]
CRITERIA=dict(temporal_relative=1e-4,reference_max_nodes=8192,reference_normalized_tolerance=1e-10,
    reference_heat_relative=1e-8,minimum_primary_boron_change_kg=.001,minimum_power_effect_relative=1e-4,numerical_margin=20.)


def source():
    return dict(model=identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (ports,ports.regen,ports.regen.chemical,ports.regen.chemical.budgets,ports.refinement)})


def gap_reference(x,flow,*,max_nodes=8192):
    """Independent BVP in normalized hot temperature and temperature difference.

    Avoid subtracting almost equal hot/cold temperatures inside the ODE.
    theta_h'=-UA/Ch*d; d'=UA*(1/Cc-1/Ch)*d,
    theta_h(0)=1; theta_h(1)-d(1)=0. No effectiveness formula.
    """
    if type(max_nodes) is not int or not 8<=max_nodes<=10000:raise ValueError('Bounded reference node budget required')
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
    def rhs(z,y):return np.vstack((-a*y[1],k*y[1]))
    def boundary(left,right):return np.array([left[0]-1.,right[0]-right[1]])
    z=np.linspace(0.,1.,8);guess=np.vstack((np.linspace(1.,.5,8),np.full(8,.5)))
    solution=solve_bvp(rhs,boundary,z,guess,tol=CRITERIA['reference_normalized_tolerance'],max_nodes=max_nodes)
    if not solution.success:raise ValueError('Normalized temperature-gap reference did not converge')
    difference=hot-cold;hout=cold+difference*solution.y[0,-1];cout=hot-difference*solution.y[1,0]
    return dict(heat_w=float(ch*(hot-hout)),hot_outlet_k=float(hout),cold_outlet_k=float(cout))


def reference_mass(time):
    return ports.regen.chemical.transferred_mass(time,flow=1.,tau=.2,close=60.)


def audit(row,metadata):
    return ports.regen.audit(row,metadata,reference_max_nodes=CRITERIA['reference_max_nodes'],
        reference_solver=gap_reference,expected_transferred_mass=reference_mass(row['time_s']))


def assess(cases):
    if set(cases)!=set(CASES) or any(set(t)!=set(PRESETS) for t in cases.values()):raise ValueError('Complete one-minute chemical/time matrix required')
    gates=dict(configuration=True,timeline=True,commands=True,budgets=True,transfer=True,eos=True,restart=True,
        rollback=True,reference_frozen=True,matched_initial=True,direction=True)
    errors=[];margins=[];receiver_temperatures=[];changes={};remaining={}
    for case,trajectories in cases.items():
        for preset,t in trajectories.items():
            if len(t['snapshots'])!=len(TIMES):raise ValueError('Complete one-minute observations required')
            gates['configuration'] &= t['checkpoint']['payload']['config']==asdict(BoronExchangeConfig(supply_boron_ppm=CASES[case],pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True))
            gates['commands'] &= t['steps']==STEPS and t['checkpoint']['payload']['history']==STEPS
            gates['timeline'] &= [r['time_s'] for r in t['snapshots']]==list(TIMES)
            gates['rollback'] &= t['rollback'] is True
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            for row in [t['initial'],*t['snapshots']]:
                gates['budgets'] &= all(ports.regen.chemical.budgets.checks(row).values())
                gates['transfer'] &= all(audit(row,t['metadata']).values())
                gates['eos'] &= ports.audit_ports(row)
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
        exact_transferred_mass_kg=reference_mass(60.))


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
    reject('wrong_end_time',lambda t:last(t).__setitem__('time_s',59.),'timeline')
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
                if i==3:saved=json.loads(json.dumps(s.checkpoint(),allow_nan=False))
            restart=None
            if preset=='fine':
                restored=BoronExchangePWRSession.restore(saved)
                restart=restored.advance(**STEPS[-1])==rows[-1] and restored.checkpoint()==s.checkpoint()
            before=s.checkpoint()
            try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
            except ValueError:rollback=s.checkpoint()==before
            else:rollback=False
            t=dict(initial=first,snapshots=rows,steps=deepcopy(STEPS),metadata=meta,restart_exact=restart,rollback=rollback,checkpoint=before)
            cases[case][preset]=t;atomic_json(out/f'{case}_{preset}.json',seal(t))
    assessment=assess(cases);negative=negative_controls(cases)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,negative_controls=negative,
        passed=assessment['passed'] and negative['passed'] and source()==frozen,trajectories=4,restart_suffixes=2,
        horizon_s=60.,restart_time_s=30.,experimental_validation=False,hour_qualified=False,spatial_qualification=False,
        station_env_connected=False,closed_tasks=[],scope='One-minute continuous 1 kg/s exchange; no closing-tail, depletion-event or plant-long-duration qualification')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('One-minute chemical qualification failed; attempts retained')
    return dict(passed=True,assessment=assessment)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
