"""Opt-in sampled IF97 liquid-port qualification for finite chemical exchange."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from physics.materials.iapws import properties_pt,saturation_temperature_k
from coupling import boron_ports,refinement
from coupling.pwr_boron import BoronExchangeConfig,BoronExchangePWRSession,identity
from scripts import qualify_boron_regenerator as regen

CASES=regen.CASES
PRESETS=regen.chemical.PRESETS
STEPS=regen.chemical.STEPS
CRITERIA=dict(temporal_relative=1e-4,enthalpy_abs_j_kg=.01,minimum_subcooling_k=1.,entropy_min_w_k=-1e-6,reference_max_nodes=4096)
NAMES=('supply','pump_outlet','primary_return','primary_donor','letdown_inlet','letdown_outlet')


def source():
    return dict(model=identity(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helpers_sha256={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (regen,regen.chemical,regen.chemical.budgets,refinement)})


def audit_ports(row):
    x=row['finite_boron_exchange'];g=x['liquid_ports'];q=x['flow_kg_s'];p=x['exchange_ports'];t=x['tanks']
    if q==0:return bool(g['enabled'] and not g['active'] and not g['checked'] and g['passed'] and g['states']=={})
    expected_gates={'liquid','enthalpy','endpoint_pinches','pump_entropy','exchanger_entropy','valve_entropy'}
    if not (g['enabled'] and g['active'] and g['checked'] and g['passed'] and set(g['gates'])==expected_gates and all(g['gates'].values()) and set(g['states'])==set(NAMES)):return False
    pp=p['pump_head_pa']+t['supply']['pressure_pa'];hs=p['supply_enthalpy_j_kg'];hp=hs+p['pump_specific_work_j_kg'];hh=p['primary_donor_enthalpy_j_kg'];heat=x['regenerator']['heat_w']/q
    expected=dict(supply=(t['supply']['pressure_pa'],hs),pump_outlet=(pp,hp),primary_return=(pp,hp+heat),
        primary_donor=(pp,hh),letdown_inlet=(pp,hh-heat),letdown_outlet=(t['receiver']['pressure_pa'],hh-heat))
    valid=True;water={}
    for name,(pressure,enthalpy) in expected.items():
        s=g['states'][name];forward=properties_pt(s['pressure_pa'],s['temperature_k']);water[name]=forward
        subcool=saturation_temperature_k(s['pressure_pa']*1e-6)-s['temperature_k']
        valid &= (s['region']==forward.region==1 and subcool>=CRITERIA['minimum_subcooling_k']
            and regen.close(s['pressure_pa'],pressure) and regen.close(s['requested_enthalpy_j_kg'],enthalpy)
            and abs(forward.h_j_kg-enthalpy)<=CRITERIA['enthalpy_abs_j_kg']
            and abs(s['enthalpy_j_kg']-enthalpy)<=CRITERIA['enthalpy_abs_j_kg']
            and abs(s['enthalpy_residual_j_kg'])<=CRITERIA['enthalpy_abs_j_kg']
            and regen.close(s['entropy_j_kg_k'],float(forward.s_j_kg_k)) and regen.close(s['subcooling_k'],float(subcool)))
    entropy=dict(pump_entropy_w_k=q*(water['pump_outlet'].s_j_kg_k-water['supply'].s_j_kg_k),
        exchanger_entropy_w_k=q*(water['primary_return'].s_j_kg_k+water['letdown_inlet'].s_j_kg_k-water['pump_outlet'].s_j_kg_k-water['primary_donor'].s_j_kg_k),
        valve_entropy_w_k=q*(water['letdown_outlet'].s_j_kg_k-water['letdown_inlet'].s_j_kg_k))
    valid &= all(v>=CRITERIA['entropy_min_w_k'] and regen.close(g[k],float(v),1e-8) for k,v in entropy.items())
    approaches=dict(hot_end_k=water['primary_donor'].T_k-water['primary_return'].T_k,
        cold_end_k=water['letdown_inlet'].T_k-water['pump_outlet'].T_k)
    valid &= all(v>=-1e-7 and regen.close(g['endpoint_approaches'][k],float(v),1e-8) for k,v in approaches.items())
    return bool(valid and g['margin_k']==1.)


def assess(cases,legacy,rejection):
    if set(cases)!=set(CASES) or set(legacy)!=set(CASES) or any(set(t)!=set(PRESETS) for t in cases.values()):
        raise ValueError('Complete strict and legacy chemical matrices required')
    gates=dict(configuration=True,commands=True,timeline=True,budgets=True,transfer=True,eos=True,restart=True,
        rollback=True,reference_frozen=True,matched_initial=True,legacy_policy=True,rejection=True)
    errors=[];margins=[];entropy=[];pinches=[];enthalpy=[]
    for case,trajectories in cases.items():
        for preset,t in trajectories.items():
            if len(t['snapshots'])!=5:raise ValueError('Complete strict timeline required')
            gates['configuration'] &= t['checkpoint']['payload']['config']==asdict(BoronExchangeConfig(supply_boron_ppm=CASES[case],pressure_pa=1e6,regenerator_ua_w_k=500000.,require_liquid_ports=True))
            gates['commands'] &= t['steps']==STEPS and t['checkpoint']['payload']['history']==STEPS
            gates['timeline'] &= [r['time_s'] for r in t['snapshots']]==[.1,.5,1.,2.,2.5]
            gates['rollback'] &= t['rollback'] is True
            if preset=='fine':gates['restart'] &= t['restart_exact'] is True
            for row in [t['initial'],*t['snapshots']]:
                gates['budgets'] &= all(regen.chemical.budgets.checks(row).values())
                gates['transfer'] &= all(regen.audit(row,t['metadata'],reference_max_nodes=CRITERIA['reference_max_nodes']).values())
                gates['eos'] &= audit_ports(row)
                gates['reference_frozen'] &= row['local_moderator']['reference']==t['initial']['local_moderator']['reference']
                g=row['finite_boron_exchange']['liquid_ports']
                if g['checked']:
                    margins.extend(x['subcooling_k'] for x in g['states'].values());enthalpy.extend(x['enthalpy_residual_j_kg'] for x in g['states'].values())
                    pinches.extend(g['endpoint_approaches'].values());entropy.extend(g[k] for k in ('pump_entropy_w_k','exchanger_entropy_w_k','valve_entropy_w_k'))
        errors.extend(regen.chemical.budgets.relative(a['coupled_state'],np.asarray(b['coupled_state'])) for a,b in zip(trajectories['coarse']['snapshots'],trajectories['fine']['snapshots']))
        old=legacy[case]
        gates['legacy_policy'] &= old['checkpoint']['payload']['config']==asdict(BoronExchangeConfig(supply_boron_ppm=CASES[case],regenerator_ua_w_k=200000.))
        for row in [old['initial'],*old['snapshots']]:
            g=row['finite_boron_exchange']['liquid_ports']
            gates['legacy_policy'] &= not g['enabled'] and not g['checked'] and not g['passed']
            gates['budgets'] &= all(regen.chemical.budgets.checks(row).values())
            gates['transfer'] &= all(regen.audit(row,old['metadata'],reference_max_nodes=CRITERIA['reference_max_nodes']).values())
    gates['matched_initial'] &= cases['dilution']['fine']['initial']['coupled_state'][:cases['dilution']['fine']['metadata']['base_size']]==cases['boration']['fine']['initial']['coupled_state'][:cases['boration']['fine']['metadata']['base_size']]
    gates['rejection'] &= (rejection['rejected'] is True and rejection['rollback_exact'] is True and rejection['start_s']==.1
        and rejection['requested_end_s']==.5 and 'liquid' in rejection['error'] and audit_ports(rejection['accepted_prefix'])
        and rejection['before_checkpoint']['sha256']==rejection['after_checkpoint_sha256'])
    gates['temporal']=max(errors)<=CRITERIA['temporal_relative']
    return dict(passed=all(gates.values()),gates={k:bool(v) for k,v in gates.items()},temporal_maximum_relative=max(errors),
        minimum_subcooling_k=min(margins),minimum_endpoint_approach_k=min(pinches),minimum_port_entropy_w_k=min(entropy),maximum_enthalpy_residual_j_kg=max(enthalpy))


def trajectory(ppm,options,*,strict):
    config=BoronExchangeConfig(supply_boron_ppm=ppm,pressure_pa=1e6 if strict else 2e5,
        regenerator_ua_w_k=500000. if strict else 200000.,require_liquid_ports=strict)
    s=BoronExchangePWRSession(**refinement.coupled_pwr_mesh(2),boron=config,**options)
    first=s.snapshot();rows=[];saved=None
    meta=dict(cold_index=13+3*s.plant.network.volume_names().index('cold_leg'),rho=s.boron_rho,
        supply_mass=config.supply_mass_kg,receiver_mass=config.receiver_mass_kg,supply_fraction=ppm*1e-6,ua_w_k=config.regenerator_ua_w_k,base_size=s.boron_index)
    for i,step in enumerate(STEPS):
        rows.append(s.advance(**step));print(s.time,flush=True)
        if i==1:saved=json.loads(json.dumps(s.checkpoint()))
    restart=None
    if strict and options==PRESETS['fine']:
        restored=BoronExchangePWRSession.restore(saved)
        restart=[restored.advance(**step) for step in STEPS[2:]]==rows[2:] and restored.checkpoint()==s.checkpoint()
    before=s.checkpoint()
    try:s.advance(.01,{'boron_flow_target_kg_s':1000.})
    except ValueError:rollback=s.checkpoint()==before
    else:rollback=False
    return dict(initial=first,snapshots=rows,steps=deepcopy(STEPS),metadata=meta,restart_exact=restart,rollback=rollback,checkpoint=before)


def rejected_interval():
    s=BoronExchangePWRSession(**refinement.coupled_pwr_mesh(2),boron=BoronExchangeConfig(
        pressure_pa=1e6,regenerator_ua_w_k=200000.,require_liquid_ports=True),**PRESETS['fine'])
    prefix=s.advance(.1,{'boron_flow_target_kg_s':50.});before=s.checkpoint();error='';rejected=False
    try:s.advance(.4)
    except ValueError as exc:error=str(exc);rejected=True
    after=s.checkpoint()
    return dict(rejected=rejected,rollback_exact=before==after,error=error,start_s=.1,requested_end_s=.5,
        accepted_prefix=prefix,before_checkpoint=before,after_checkpoint_sha256=after['sha256'])


def negative_controls(cases,legacy,rejection):
    checks={}
    def reject(name,modify,gate):
        copied=deepcopy(cases);modify(copied['dilution']['fine']);a=assess(copied,legacy,rejection)
        checks[name]=not a['passed'] and not a['gates'][gate]
    def ports(t):return t['snapshots'][-1]['finite_boron_exchange']['liquid_ports']
    reject('disabled_policy',lambda t:ports(t).__setitem__('enabled',False),'eos')
    reject('wrong_region',lambda t:ports(t)['states']['letdown_outlet'].__setitem__('region',4),'eos')
    reject('wrong_enthalpy',lambda t:ports(t)['states']['primary_return'].__setitem__('requested_enthalpy_j_kg',0.),'eos')
    reject('wrong_valve_pressure',lambda t:ports(t)['states']['letdown_outlet'].__setitem__('pressure_pa',2e6),'eos')
    reject('negative_valve_entropy',lambda t:ports(t).__setitem__('valve_entropy_w_k',-1.),'eos')
    reject('wrong_pinch',lambda t:ports(t)['endpoint_approaches'].__setitem__('cold_end_k',-10.),'eos')
    reject('failed_restart',lambda t:t.__setitem__('restart_exact',False),'restart')
    broken=deepcopy(rejection);broken['rollback_exact']=False
    checks['failed_rollback']=not assess(cases,legacy,broken)['gates']['rejection']
    return dict(passed=all(checks.values()),checks=checks,scope='Mutated copies only; native records unchanged')


def run(output):
    out=Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True);frozen=source();cases={};legacy={}
    for case,ppm in CASES.items():
        cases[case]={}
        for preset,options in PRESETS.items():
            print(case,preset,flush=True);t=trajectory(ppm,options,strict=True);cases[case][preset]=t
            atomic_json(out/f'{case}_{preset}.json',seal(t))
        print(case,'legacy',flush=True);legacy[case]=trajectory(ppm,PRESETS['fine'],strict=False)
        atomic_json(out/f'{case}_legacy.json',seal(legacy[case]))
    print('phase rejection',flush=True);rejection=rejected_interval();atomic_json(out/'rejection.json',seal(rejection))
    assessment=assess(cases,legacy,rejection);negative=negative_controls(cases,legacy,rejection)
    report=dict(source=frozen,criteria=CRITERIA,assessment=assessment,negative_controls=negative,
        passed=assessment['passed'] and negative['passed'] and source()==frozen,
        strict_trajectories=4,legacy_trajectories=2,rejected_intervals=1,restart_suffixes=2,horizon_s=2.5,
        experimental_validation=False,long_horizon_accepted=False,spatial_qualification=False,station_env_connected=False,closed_tasks=[],
        scope='Opt-in sampled liquid IF97 ports, entropy and endpoint pinches; rejection rolls back entire interval, no phase-event localization')
    atomic_json(out/'report.json',seal(report));atomic_json(out/'scientific.json',seal(dict(report=report,cases=cases,legacy=legacy,rejection=rejection)))
    if not report['passed']:raise ValueError('Liquid-port qualification failed; attempts retained')
    return dict(passed=True,assessment=assessment)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
