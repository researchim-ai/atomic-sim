"""Two consecutive finite mass doses with inventory-aware atomic commands."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from coupling import boron_dosing as dosing
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from scripts import qualify_boron_closure as closure

CASES=closure.CASES
PRESETS=dict(coarse=dict(max_step_s=.05,rtol=2e-8),fine=dict(max_step_s=.025,rtol=2e-9))
MASSES=(.3,.2)
CRITERIA=dict(temporal_relative=1e-4,mass_absolute_kg=1e-7,minimum_power_effect_relative=1e-6,numerical_margin=20.)


def source():
    return dict(model=identity(),dosing_sha256=hashlib.sha256(Path(dosing.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (closure,closure.duration,closure.ports,closure.ports.regen,closure.ports.regen.chemical,closure.ports.regen.chemical.budgets,closure.ports.refinement)})


def independent_transfer(plan,q0,tau,closing):
    """Numerical quadrature of actuator rate, independent of planner mass formula."""
    target=plan['target_flow_kg_s'];opened=plan['opening_duration_s']
    def rate(t):return target+(q0-target)*math.exp(-t/tau)
    mass=quad(rate,0.,opened,epsabs=1e-13,epsrel=1e-12)[0]
    if closing:mass+=quad(lambda t:rate(opened)*math.exp(-t/tau),0.,plan['closing_duration_s'],epsabs=1e-13,epsrel=1e-12)[0]
    return mass


def assess(cases):
    if set(cases)!=set(CASES) or any(set(t)!=set(PRESETS) for t in cases.values()):raise ValueError('Complete dose chemical/time matrix required')
    gates=dict(configuration=True,commands=True,plans=True,budgets=True,exchange=True,eos=True,restart=True,
        matched_initial=True,direction=True,finite_tail=True,reference_frozen=True)
    errors=[];margins=[];changes={}
    for case,trajectories in cases.items():
        for preset,t in trajectories.items():
            if len(t['doses'])!=2 or len(t['snapshots'])!=4:raise ValueError('Two complete dose cycles required')
            config=BoronExchangeConfig(supply_boron_ppm=CASES[case],pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True)
            gates['configuration'] &= t['checkpoint']['payload']['config']==asdict(config)
            gates['commands'] &= t['checkpoint']['payload']['history']==[step for dose in t['doses'] for step in dose['steps']]
            q0=0.;mass0=0.;time0=t['initial']['time_s']
            rows=[(t['initial'],0.)]
            for wanted,dose in zip(MASSES,t['doses']):
                plan=dose['plan'];opening,closing=dose['snapshots'];tau=config.actuator_tau_s
                gates['plans'] &= plan['requested_mass_kg']==wanted and plan['target_flow_kg_s']==1. and plan['tail_time_constants']==5.
                gates['plans'] &= math.isclose(independent_transfer(plan,q0,tau,True),wanted,abs_tol=1e-12)
                gates['plans'] &= dose['whole_dose_committed'] is True and abs(dose['transferred_mass_kg']-wanted)<=CRITERIA['mass_absolute_kg'] and abs(dose['received_mass_kg']-wanted)<=CRITERIA['mass_absolute_kg']
                gates['commands'] &= dose['steps']==[dict(duration_s=plan['opening_duration_s'],patch={'boron_flow_target_kg_s':1.}),dict(duration_s=plan['closing_duration_s'],patch={'boron_flow_target_kg_s':0.})]
                gates['commands'] &= opening['time_s']==time0+plan['opening_duration_s'] and closing['time_s']==opening['time_s']+plan['closing_duration_s']
                qclose=1.+(q0-1.)*math.exp(-plan['opening_duration_s']/tau);end=qclose*math.exp(-5.)
                gates['finite_tail'] &= math.isclose(closing['finite_boron_exchange']['flow_kg_s'],end,rel_tol=1e-10,abs_tol=1e-12) and end>0.
                gates['finite_tail'] &= math.isclose(plan['future_tail_mass_kg'],tau*end,rel_tol=1e-10)
                gates['finite_tail'] &= math.isclose(plan['eventual_transfer_mass_kg'],wanted+tau*end,rel_tol=1e-12)
                rows.extend([(opening,mass0+independent_transfer(plan,q0,tau,False)),(closing,mass0+independent_transfer(plan,q0,tau,True))])
                q0=end;mass0+=wanted;time0=closing['time_s']
            gates['commands'] &= t['snapshots']==[row for row,_ in rows[1:]]
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            for row,mass in rows:
                gates['budgets'] &= all(closure.ports.regen.chemical.budgets.checks(row).values())
                gates['exchange'] &= all(closure.ports.regen.audit(row,t['metadata'],reference_solver=closure.tail_reference,reference_max_nodes=8192,expected_transferred_mass=mass).values())
                gates['eos'] &= closure.ports.audit_ports(row)
                gates['reference_frozen'] &= row['local_moderator']['reference']==t['initial']['local_moderator']['reference']
                g=row['finite_boron_exchange']['liquid_ports']
                if g['checked']:margins.extend(v['subcooling_k'] for v in g['states'].values())
            delta=closure.ports.regen.chemical.primary_boron(rows[-1][0])-closure.ports.regen.chemical.primary_boron(t['initial'])
            changes[case+'_'+preset]=delta;gates['direction'] &= delta<-.0001 if case=='dilution' else delta>.0001
        errors.extend(closure.ports.regen.chemical.budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots']))
    a,b=(cases[c]['fine'] for c in CASES);base=a['metadata']['base_size'];p0=a['initial']['powers']['nuclear_fission_w']
    gates['matched_initial'] &= a['initial']['coupled_state'][:base]==b['initial']['coupled_state'][:base]
    effect=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for x,y in zip(a['snapshots'],b['snapshots']))
    noise=max(abs(x['powers']['nuclear_fission_w']-y['powers']['nuclear_fission_w'])/p0 for t in cases.values() for x,y in zip(t['coarse']['snapshots'],t['fine']['snapshots']))
    gates.update(temporal=max(errors)<=CRITERIA['temporal_relative'],effect_resolved=effect>=CRITERIA['minimum_power_effect_relative'] and effect>=CRITERIA['numerical_margin']*noise)
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=max(errors),
        minimum_subcooling_k=min(margins),power_effect_relative=effect,numerical_power_relative=noise,
        primary_boron_changes_kg=changes,requested_transfer_mass_kg=sum(MASSES))


def negative_controls(cases):
    checks={}
    def reject(name,mutate,gate):
        copied=deepcopy(cases);mutate(copied['dilution']['fine']);r=assess(copied);checks[name]=not r['passed'] and not r['gates'][gate]
    reject('wrong_dose_mass',lambda t:t['doses'][1]['plan'].__setitem__('requested_mass_kg',.3),'plans')
    reject('tail_reserve_omitted',lambda t:t['doses'][1]['plan'].__setitem__('future_tail_mass_kg',0.),'finite_tail')
    reject('command_lost',lambda t:t['doses'][1]['steps'][1].__setitem__('patch',None),'commands')
    reject('restart_failed',lambda t:t.__setitem__('restart_exact',False),'restart')
    reject('commit_missing',lambda t:t['doses'][1].__setitem__('whole_dose_committed',False),'plans')
    reject('energy_failure',lambda t:t['snapshots'][-1].__setitem__('energy_relative',1.),'budgets')
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies; native artifacts retained')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={}
    for case,ppm in CASES.items():
        cases[case]={}
        for preset,options in PRESETS.items():
            print(case,preset,flush=True)
            config=BoronExchangeConfig(supply_boron_ppm=ppm,pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True)
            s=BoronExchangePWRSession(**closure.ports.refinement.coupled_pwr_mesh(2),boron=config,**options)
            first=s.snapshot();doses=[];rows=[];saved=None
            meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,supply_mass=config.supply_mass_kg,
                receiver_mass=config.receiver_mass_kg,supply_fraction=ppm*1e-6,ua_w_k=config.regenerator_ua_w_k,base_size=s.boron_index)
            for mass in MASSES:
                plan=dosing.plan_boron_dose(s,mass_kg=mass,flow_kg_s=1.);dose=dosing.execute_boron_dose(s,plan)
                doses.append(dose);rows.extend(dose['snapshots']);print(case,preset,s.time,flush=True)
            restart=None
            if preset=='fine':
                # Native restart inside the second planned dose. Atomic execution
                # itself is intentionally all-or-nothing, not a resumable operation.
                history=list(s.history)
                replay=BoronExchangePWRSession(**closure.ports.refinement.coupled_pwr_mesh(2),boron=config,**options)
                for step in history[:3]:replay.advance(**step)
                saved=json.loads(json.dumps(replay.checkpoint(),allow_nan=False));restored=BoronExchangePWRSession.restore(saved)
                restart=restored.advance(**history[3])==rows[-1] and restored.checkpoint()==s.checkpoint()
            t=dict(initial=first,doses=doses,snapshots=rows,metadata=meta,restart_exact=restart,checkpoint=s.checkpoint())
            cases[case][preset]=t;atomic_json(out/f'{case}_{preset}.json',seal(t))
    assessment=assess(cases);negative=negative_controls(cases)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,negative_controls=negative,
        passed=assessment['passed'] and negative['passed'] and source()==frozen,trajectories=4,native_doses=8,restart_suffixes=2,
        experimental_validation=False,hour_qualified=False,station_env_connected=False,closed_tasks=[],
        scope='Atomic 0.3+0.2 kg dose commands with 5-tau tails; reserve forecast is not thermal or plant qualification')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases)))
    if not report['passed']:raise ValueError('Finite-dose qualification failed; attempts retained')
    return dict(passed=True,assessment=assessment)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);print(run(p.parse_args().output))
